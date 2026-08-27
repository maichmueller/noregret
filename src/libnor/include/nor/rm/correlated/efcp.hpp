
#ifndef NOR_RM_CORRELATED_EFCP_HPP
#define NOR_RM_CORRELATED_EFCP_HPP

#include <algorithm>
#include <array>
#include <cassert>
#include <cmath>
#include <cstdint>
#include <functional>
#include <limits>
#include <numeric>
#include <optional>
#include <ranges>
#include <stdexcept>
#include <unordered_map>
#include <unordered_set>
#include <utility>
#include <vector>

#include "common/common.hpp"
#include "nor/concepts.hpp"
#include "nor/fosg_traits.hpp"
#include "nor/game_defs.hpp"
#include "nor/rm/correlated/joint_distribution.hpp"
#include "nor/rm/correlated/sequence_form.hpp"
#include "nor/type_defs.hpp"
#include "nor/utils/utils.hpp"

namespace nor::rm::correlated {

/**
 * @brief EFCP -- Extensive-Form Correlation Plan regret minimization
 *        (Farina, Ling, Fang & Sandholm, "Efficient Regret Minimization
 *        Algorithm for Extensive-Form Correlated Equilibrium", NeurIPS 2019,
 *        arXiv:1910.12450).
 *
 * The solver searches directly over the polytope Xi of sequence-form
 * correlation plans of a two-player perfect-recall extensive-form game WITHOUT
 * chance moves (von Stengel & Forges 2008): Xi = {xi >= 0 : xi[0,0] = 1,
 * sum_a xi[(I,a),tau] = xi[I_bar,tau] for every relevant (information set,
 * opponent sequence) combination}. The majority of those flow constraints is
 * redundant; the paper's DECOMPOSE procedure identifies a non-redundant subset
 * expressible hierarchically (its Theorem 1: Xi is a CHAIN of scaled
 * extensions of action simplexes Delta^{|A_I|} and singleton sets {1}), from
 * which a global regret minimizer over Xi is composed from local
 * regret-matching-plus leaves -- an instance of the Farina-Kroer-Sandholm
 * (ICML 2019) regret-circuit framework. Iterates are always feasible; no
 * projection is ever taken.
 *
 * Self-play protocol (paper Section 5): the mediator minimizer over Xi faces a
 * deviator minimizer over the convex hull
 *     Y = conv_union_k lambda_k * Qtld_k,  lambda in Delta^{|Sigma|-1},
 * where Qtld_k is the responding player's sequence-form polytope rooted below
 * the trigger sequence sigma_k = (I_k, a_k) ("treeplex rooted at
 * sigma(I_k)"). Alternating updates and linear averaging drive the average
 * correlation plan to a minimax point of the saddle-point reformulation of
 * Farina et al. 2019c, i.e. an extensive-form correlated equilibrium; running
 * it on the raw bilinear payoff selects a high-social-welfare mediation on the
 * repo's general-sum beds.
 *
 * SCOPE CAVEATS (explicit, deliberate): losses are evaluated through
 * terminal and reduced-plan enumeration (polynomial objects, quadratic in the
 * terminal count per trigger per iteration) rather than through bottom-up
 * sequence-form propagation; fine for validation-scale games, replaceable
 * without touching the decomposition/circuit layers. Chance moves throw.
 */
inline constexpr int32_t k_empty_sequence_id = 0;

/// CFR+-style regret-matching-plus kernel over abstract simplex slots
/// (identical update arithmetic to rm::RegretMatchingPlus -- cumulative-table
/// clamping at recommendation time -- keyed by position instead of actions)
struct SimplexRMPlus {
   std::vector< double > regret;

   [[nodiscard]] size_t size() const { return regret.size(); }
   /// appends 'n' fresh (zero-regret) simplex slots
   void register_slots(size_t n) { regret.resize(regret.size() + n, 0.); }

   void observe(size_t idx, double increment) { regret.at(idx) += increment; }

   /// folds one batch of instantaneous increments (aligned with the table)
   void observe_span(const double* increments, size_t n)
   {
      assert(n == regret.size());
      for(auto idx : std::views::iota(size_t{0}, n)) {
         regret[idx] += increments[idx];
      }
   }

   /// clamps in place and matches on positive cumulative mass
   [[nodiscard]] std::vector< double > recommend()
   {
      double pos_sum = 0.;
      for(double& value : regret) {
         value = std::max(0., value);
         pos_sum += value;
      }
      std::vector< double > policy(regret.size());
      if(pos_sum > 0.) {
         for(auto idx : std::views::iota(size_t{0}, regret.size())) {
            policy[idx] = regret[idx] / pos_sum;
         }
      } else {
         std::ranges::fill(policy, 1. / double(regret.size()));
      }
      return policy;
   }
};

/**
 * @brief one operation of the scaled-extension chain realizing Xi (the paper's
 *        FillSimplex / SumSimplex primitives).
 *
 * A 'fill' partitions the current value held at 'source' among 'children'
 * through local leaf minimizer 'rm_index'; a 'sum' writes the aggregate of
 * 'sources' into 'target' (singleton-set scaled extension, no minimizer --
 * Decompose lines 20-29). Coordinates address flat indices of the relevant-
 * pair grid managed by CorrelationPlanSpace.
 */
struct CircuitOp {
   enum class Kind : uint8_t { fill, sum };
   Kind kind = Kind::fill;
   /// decomposition frame of the emitting Decompose call (diagnostics)
   int32_t frame_s1 = k_empty_sequence_id;
   int32_t frame_s2 = k_empty_sequence_id;
   /// fill: coordinate whose stored value is partitioned
   int32_t source = -1;
   /// sum: coordinate receiving the aggregate
   int32_t target = -1;
   /// fill: freshly-filled coordinates
   std::vector< int32_t > children{};
   /// sum: aggregate operands (previously-filled coordinates)
   std::vector< int32_t > sources{};
   /// fill: index into the circuit's leaf-simplex table (-1 otherwise)
   int32_t rm_index = -1;
};

namespace detail {

struct PairKeyHash {
   [[nodiscard]] size_t operator()(const std::pair< uint32_t, uint32_t >& key) const noexcept
   {
      size_t seed = 0;
      common::hash_combine(seed, key.first);
      common::hash_combine(seed, key.second);
      return seed;
   }
};

}  // namespace detail

/**
 * @brief correlation-matrix layout and the structural DECOMPOSITION of Xi for
 *        EFCP: sequence registries (with empty-sequence sentinel 0), relevant-
 *        pair indexing, information-set connectivity, the full Definition-3
 *        constraint system (kept for feasibility auditing -- most of it is
 *        redundant and intentionally NOT enforced by the circuit) and the
 *        paper's Algorithm-2 decomposition emitted as CircuitOp chain.
 */
template < concepts::fosg Env >
class CorrelationPlanSpace {
   using Oracle = SequenceFormOracle< Env >;

  public:
   explicit CorrelationPlanSpace(const Oracle& oracle) : m_oracle(oracle)
   {
      const auto& roster = oracle.players();
      if(roster.size() != 2) {
         throw std::invalid_argument("CorrelationPlanSpace: exactly two players required");
      }
      m_players = {roster.at(0), roster.at(1)};
      if constexpr(requires { Env::stochasticity(); }) {
         static_assert(
            Env::stochasticity() == Stochasticity::deterministic,
            "EFCP requires a chance-free environment (the von Stengel-Forges "
            "correlation-plan polytope characterization holds only without "
            "chance moves)"
         );
      }
      _build_sequences_and_chains();
      _build_connectivity_and_relevance();
      _build_terminal_indexes();
      _build_constraints();
      _decompose_root();
      _validate_decomposition();
   }

   ////////////////////////////
   /// structure accessors  ///
   ////////////////////////////

   [[nodiscard]] const Oracle& oracle() const { return m_oracle; }
   [[nodiscard]] Player player(size_t slot) const { return m_players[slot]; }
   [[nodiscard]] size_t player_slot(Player player) const { return m_oracle.player_slot(player); }
   /// sequence registry size of 'player' INCLUDING empty-sequence id 0
   [[nodiscard]] size_t sequence_count(Player player) const
   {
      return m_sequences[player_slot(player)].size();
   }

   struct SequenceRecord {
      Player player = Player::unknown;
      /// owner infoset of sigma=(I,a); sentinel for the empty sequence
      uint32_t owner_infoset = std::numeric_limits< uint32_t >::max();
      uint32_t action_idx = 0;
      /// preceding own sequence id (0 == empty / root)
      int32_t parent = k_empty_sequence_id;
      /// deferred registration key of 'parent', resolved post-registration
      std::optional< std::pair< uint32_t, uint32_t > > _parent_key{};
      bool empty() const { return owner_infoset == std::numeric_limits< uint32_t >::max(); }
   };

   [[nodiscard]] const SequenceRecord& sequence(Player player, int32_t seq) const
   {
      return m_sequences[player_slot(player)].at(size_t(seq));
   }
   [[nodiscard]] int32_t sequence_id(Player player, uint32_t infoset, uint32_t action_idx) const
   {
      return m_seq_ids[player_slot(player)].at(std::pair{infoset, action_idx});
   }
   [[nodiscard]] bool has_sequence_id(Player player, uint32_t infoset, uint32_t action_idx) const
   {
      return m_seq_ids[player_slot(player)].contains(std::pair{infoset, action_idx});
   }
   /// unique id of the relevant pair (s1, s2); -1 marks irrelevance
   [[nodiscard]] int32_t index_of(int32_t s1, int32_t s2) const
   {
      return m_rel_index.at(size_t(s1)).at(size_t(s2));
   }
   [[nodiscard]] size_t relevant_pair_count() const { return m_rel_coords.size(); }
   [[nodiscard]] std::pair< int32_t, int32_t > pair_of(int32_t coord) const
   {
      return m_rel_coords.at(size_t(coord));
   }
   [[nodiscard]] bool relevant(int32_t s1, int32_t s2) const { return index_of(s1, s2) >= 0; }
   /// number of committed own actions encoded in the sequence (empty == 0)
   [[nodiscard]] size_t sequence_depth(Player player, int32_t seq) const
   {
      return m_seq_depths[player_slot(player)][size_t(seq)];
   }
   /// strict ancestor steps of the sequence, root-first, as (owner infoset,
   /// action) decision pairs (excludes the sequence's own final step)
   [[nodiscard]] const std::vector< std::pair< uint32_t, uint32_t > >&
   sequence_chain(Player player, int32_t seq) const
   {
      return m_seq_chains[player_slot(player)][size_t(seq)];
   }
   /// ancestor DECISIONS of 'infoset', root-first: every own (infoset, action)
   /// commitment that precedes acting at 'infoset' (excludes acting there)
   [[nodiscard]] const std::vector< std::pair< uint32_t, uint32_t > >&
   infoset_chain(Player player, uint32_t infoset) const
   {
      return m_infostate_chains[player_slot(player)][infoset];
   }
   /// parent sequence of 'infoset' (0 for root-level sets)
   [[nodiscard]] int32_t infoset_parent_sequence(Player player, uint32_t infoset) const
   {
      return m_infoset_parents[player_slot(player)][infoset];
   }
   /// information-set connectedness (paper Definition 1)
   [[nodiscard]] bool connected(size_t infoset_p1, size_t infoset_p2) const
   {
      return m_connected.at(infoset_p1).at(infoset_p2);
   }
   /// terminals passing through the exact decision step (I, a) of 'player'
   [[nodiscard]] const std::vector< uint32_t >&
   terminals_by_step(Player player, uint32_t infoset, uint32_t action_idx) const
   {
      return m_terminals_by_step[player_slot(player)][infoset][action_idx];
   }
   /// terminals whose 'player'-signature passes through 'infoset'
   [[nodiscard]] const std::vector< uint32_t >& terminals_through(Player player, uint32_t infoset)
      const
   {
      return m_terminals_through[player_slot(player)][infoset];
   }
   /// relevant-pair coordinate carrying terminal z (chance-free: xi[z] realizes
   /// the terminal mass directly)
   [[nodiscard]] int32_t terminal_pair_coordinate(size_t z) const { return m_terminal_coords[z]; }
   /// 'iid' at-or-descendant-of 'trigger' within player's own tree (plans that
   /// fired the trigger control exactly these sets)
   [[nodiscard]] bool infoset_at_or_below_trigger(Player player, uint32_t iid, uint32_t trigger)
      const
   {
      return m_below[player_slot(player)][trigger].contains(iid);
   }
   /// response-subtree of trigger sequence 'root' (= the supporting infosets of
   /// Q~_root): own infosets whose parent sequence lies at-or-below 'root'
   [[nodiscard]] std::vector< uint32_t > response_infosets(Player player, int32_t root) const
   {
      std::vector< uint32_t > out;
      const auto n = m_oracle.structure(player).size();
      for(auto iid : std::views::iota(uint32_t{0}, uint32_t(n))) {
         const int32_t parent = infoset_parent_sequence(player, iid);
         if(sequence_is_at_or_below(player, parent, root)) {
            out.push_back(iid);
         }
      }
      return out;
   }
   /// descendant-or-equal test between sequences of the same player
   [[nodiscard]] bool sequence_is_at_or_below(Player player, int32_t descendant, int32_t ancestor)
      const
   {
      if(ancestor == k_empty_sequence_id) {
         return true;
      }
      const auto slot = player_slot(player);
      for(const auto& [anc_iid, anc_aid] : m_seq_chains[slot][size_t(descendant)]) {
         auto found = m_seq_ids[slot].find({anc_iid, anc_aid});
         if(found != m_seq_ids[slot].end() and found->second == ancestor) {
            return true;
         }
      }
      return false;
   }

   ////////////////////////////////////
   /// Definition-3 constraint set ///
   ///////////////////////////////////

   struct ConstraintEntry {
      /// coefficients over relevant-pair coordinates; enforce <lhs> == xi[rhs]
      std::vector< std::pair< int32_t, double > > lhs_terms;
      int32_t rhs = -1;
   };
   [[nodiscard]] const std::vector< ConstraintEntry >& constraints() const { return m_constraints; }
   /// max_violation and L1 residual of the full Def.-3 system at 'plan'
   struct FeasibilityResidual {
      double linf = 0.;
      double l1 = 0.;
   };
   [[nodiscard]] FeasibilityResidual feasibility_residual(const std::vector< double >& plan) const
   {
      FeasibilityResidual out{};
      for(const auto& constraint : m_constraints) {
         double lhs = 0.;
         for(const auto& [coord, coefficient] : constraint.lhs_terms) {
            lhs += coefficient * plan.at(size_t(coord));
         }
         const double diff = std::abs(lhs - plan.at(size_t(constraint.rhs)));
         out.linf = std::max(out.linf, diff);
         out.l1 += diff;
      }
      return out;
   }

   ////////////////////////////
   /// decomposition output ///
   ////////////////////////////

   [[nodiscard]] const std::vector< CircuitOp >& decomposition_ops() const { return m_ops; }
   [[nodiscard]] size_t leaf_unit_count() const { return m_leaf_counter; }

  private:
   const Oracle& m_oracle;
   std::array< Player, 2 > m_players{Player::alex, Player::bob};
   std::array< std::vector< SequenceRecord >, 2 > m_sequences{};
   std::array<
      std::unordered_map< std::pair< uint32_t, uint32_t >, int32_t, detail::PairKeyHash >,
      2 >
      m_seq_ids{};
   std::array< std::vector< size_t >, 2 > m_seq_depths{};
   std::array< std::vector< std::vector< std::pair< uint32_t, uint32_t > > >, 2 > m_seq_chains{};
   std::array< std::vector< std::vector< std::pair< uint32_t, uint32_t > > >, 2 >
      m_infostate_chains{};
   std::array< std::vector< int32_t >, 2 > m_infoset_parents{};
   std::vector< std::vector< char > > m_connected{};
   std::vector< std::vector< int32_t > > m_rel_index{};
   std::vector< std::pair< int32_t, int32_t > > m_rel_coords{};
   std::vector< ConstraintEntry > m_constraints{};
   std::vector< CircuitOp > m_ops{};
   size_t m_leaf_counter = 0;
   std::array< std::vector< std::vector< std::vector< uint32_t > > >, 2 > m_terminals_by_step{};
   std::array< std::vector< std::vector< uint32_t > >, 2 > m_terminals_through{};
   std::vector< int32_t > m_terminal_coords{};
   std::array< std::vector< std::unordered_set< uint32_t > >, 2 > m_below{};
   /// decomposition ledger of the construction (paper set S)
   std::vector< char > m_filled{};

   /////////////////////////////////////////
   /// static-structure construction    ///
   ///////////////////////////////////////

   void _build_sequences_and_chains()
   {
      const size_t n_terms = m_oracle.terminal_count();

      for(auto slot : std::views::iota(size_t{0}, size_t{2})) {
         SequenceRecord empty_rec{};
         empty_rec.player = m_players[slot];
         m_sequences[slot].push_back(empty_rec);
      }

      // registration pass: a sequence (prev_i, prev_a) appears whenever a
      // signature continues from the (prev_i, prev_a) decision to any next own
      // decision (or terminates); perfect recall makes occurrences agree
      for(auto z : std::views::iota(size_t{0}, n_terms)) {
         for(auto slot : std::views::iota(size_t{0}, size_t{2})) {
            const auto& sig = m_oracle.terminal(z).signatures[slot];
            for(auto step : std::views::iota(size_t{0}, sig.size())) {
               const auto key = sig[step];
               if(m_seq_ids[slot].contains(key)) {
                  continue;
               }
               SequenceRecord rec{};
               rec.player = m_players[slot];
               rec.owner_infoset = key.first;
               rec.action_idx = key.second;
               if(step > 0) {
                  rec._parent_key = sig[step - 1];
               }
               const int32_t sid = int32_t(m_sequences[slot].size());
               m_sequences[slot].push_back(rec);
               m_seq_ids[slot].emplace(key, sid);
            }
         }
      }
      // resolve parent keys into registry ids
      for(auto slot : std::views::iota(size_t{0}, size_t{2})) {
         for(auto& rec : m_sequences[slot]) {
            if(rec.empty()) {
               continue;
            }
            if(rec._parent_key.has_value()) {
               rec.parent = m_seq_ids[slot].at(*rec._parent_key);
            }
         }
         // depths + strict-ancestor chains by upward walking (the registry is
         // a forest rooted at the empty sequence)
         m_seq_depths[slot].assign(m_sequences[slot].size(), 0);
         m_seq_chains[slot].assign(m_sequences[slot].size(), {});
         for(auto sid : std::views::iota(size_t{1}, m_sequences[slot].size())) {
            std::vector< std::pair< uint32_t, uint32_t > > chain;
            int64_t walk = int64_t(sid);
            while(walk > 0) {
               const auto& rec = m_sequences[slot][size_t(walk)];
               chain.emplace_back(rec.owner_infoset, rec.action_idx);
               walk = rec.parent;
            }
            std::ranges::reverse(chain);
            m_seq_depths[slot][sid] = chain.size();
            m_seq_chains[slot][sid] = std::move(chain);
         }
      }
      // per-infoset parent sequence + ancestor decision list
      for(auto slot : std::views::iota(size_t{0}, size_t{2})) {
         const size_t n = m_oracle.structure(m_players[slot]).size();
         m_infoset_parents[slot].assign(n, k_empty_sequence_id);
         m_infostate_chains[slot].assign(n, {});
         for(const auto& [key, sid] : m_seq_ids[slot]) {
            // every infoset with outgoing actions owns >= 1 registered sequence
            m_infoset_parents[slot][key.first] = m_sequences[slot][size_t(sid)].parent;
         }
         for(auto iid : std::views::iota(uint32_t{0}, uint32_t(n))) {
            const int32_t parent = m_infoset_parents[slot][iid];
            if(parent == k_empty_sequence_id) {
               continue;
            }
            const auto& chain = m_seq_chains[slot][size_t(parent)];
            m_infostate_chains[slot][iid].assign(chain.begin(), chain.end());
         }
      }
   }

   void _build_connectivity_and_relevance()
   {
      const size_t n1 = m_oracle.structure(m_players[0]).size();
      const size_t n2 = m_oracle.structure(m_players[1]).size();
      const size_t n_terms = m_oracle.terminal_count();

      // connected iff SOME path co-visits both sets in either direction --
      // every co-path pairing qualifies, so marking all cross products over
      // per-terminal signature keys reproduces Definition 1 exactly
      m_connected.assign(n1, std::vector< char >(n2, 0));
      for(auto z : std::views::iota(size_t{0}, n_terms)) {
         for(const auto& [i1, _] : m_oracle.terminal(z).signatures[0]) {
            for(const auto& [i2, __] : m_oracle.terminal(z).signatures[1]) {
               m_connected[i1][i2] = 1;
            }
         }
      }

      const size_t n_seqs1 = m_sequences[0].size();
      const size_t n_seqs2 = m_sequences[1].size();
      m_rel_index.assign(n_seqs1, std::vector< int32_t >(n_seqs2, -1));
      for(auto s1 : std::views::iota(int32_t{0}, int32_t(n_seqs1))) {
         for(auto s2 : std::views::iota(int32_t{0}, int32_t(n_seqs2))) {
            const bool relevant = [&] {
               if(s1 == k_empty_sequence_id or s2 == k_empty_sequence_id) {
                  return true;
               }
               const auto& r1 = m_sequences[0][size_t(s1)];
               const auto& r2 = m_sequences[1][size_t(s2)];
               return m_connected[r1.owner_infoset][r2.owner_infoset] != 0;
            }();
            if(relevant) {
               m_rel_index[size_t(s1)][size_t(s2)] = int32_t(m_rel_coords.size());
               m_rel_coords.emplace_back(s1, s2);
            }
         }
      }
      for(auto& row : m_rel_index) {
         if(row.front() < 0) {
            throw std::logic_error(
               "EFCP: pairs involving the empty sequence are always "
               "relevant"
            );
         }
      }
   }

   void _build_terminal_indexes()
   {
      const size_t n_terms = m_oracle.terminal_count();
      for(auto slot : std::views::iota(size_t{0}, size_t{2})) {
         const size_t n = m_oracle.structure(m_players[slot]).size();
         m_terminals_by_step[slot].assign(n, {});
         m_terminals_through[slot].assign(n, {});
         m_below[slot].assign(n, {});
         for(auto iid : std::views::iota(uint32_t{0}, uint32_t(n))) {
            const size_t n_actions = m_oracle.structure(m_players[slot]).actions[iid].size();
            m_terminals_by_step[slot][iid].assign(n_actions, {});
         }
         std::unordered_map< uint32_t, std::vector< uint32_t > > child_map;
         for(auto z : std::views::iota(size_t{0}, n_terms)) {
            const auto& sig = m_oracle.terminal(z).signatures[slot];
            for(auto step : std::views::iota(size_t{0}, sig.size())) {
               const auto [iid, aid] = sig[step];
               m_terminals_by_step[slot][iid][aid].push_back(uint32_t(z));
               m_terminals_through[slot][iid].push_back(uint32_t(z));
               if(step > 0) {
                  child_map[sig[step - 1].first].push_back(iid);
               }
            }
         }
         // closure: descendants-or-self per infoset (drives the deviation
         // suffix-consistency tests of the trigger-gap evaluator)
         for(auto root : std::views::iota(uint32_t{0}, uint32_t(n))) {
            std::vector< uint32_t > stack{root};
            while(not stack.empty()) {
               const uint32_t cur = stack.back();
               stack.pop_back();
               auto& below = m_below[slot][root];
               if(below.contains(cur)) {
                  continue;
               }
               below.insert(cur);
               for(uint32_t child : child_map[cur]) {
                  stack.push_back(child);
               }
            }
         }
      }
      m_terminal_coords.resize(n_terms);
      for(auto z : std::views::iota(size_t{0}, n_terms)) {
         const auto& sig1 = m_oracle.terminal(z).signatures[0];
         const auto& sig2 = m_oracle.terminal(z).signatures[1];
         const int32_t s1 = sig1.empty() ? k_empty_sequence_id : m_seq_ids[0].at(sig1.back());
         const int32_t s2 = sig2.empty() ? k_empty_sequence_id : m_seq_ids[1].at(sig2.back());
         const int32_t coord = index_of(s1, s2);
         if(coord < 0) {
            throw std::logic_error("EFCP: terminal judged irrelevant to both roots");
         }
         m_terminal_coords[z] = coord;
      }
   }

   void _build_constraints()
   {
      for(auto slot : std::views::iota(size_t{0}, size_t{2})) {
         const Player player = m_players[slot];
         const size_t n = m_oracle.structure(player).size();
         for(auto iid : std::views::iota(uint32_t{0}, uint32_t(n))) {
            const size_t n_actions = m_oracle.structure(player).actions[iid].size();
            std::vector< int32_t > child_seqs;
            for(auto aid : std::views::iota(uint32_t{0}, uint32_t(n_actions))) {
               if(has_sequence_id(player, iid, aid)) {
                  child_seqs.push_back(sequence_id(player, iid, aid));
               }
            }
            if(child_seqs.empty()) {
               continue;
            }
            const int32_t parent = m_sequences[slot][size_t(child_seqs.front())].parent;
            const auto other_slot = size_t{1} - slot;
            for(auto tau : std::views::iota(
                   int32_t{k_empty_sequence_id}, int32_t(m_sequences[other_slot].size())
                )) {
               ConstraintEntry entry{};
               entry.rhs = slot == 0 ? index_of(parent, tau) : index_of(tau, parent);
               if(entry.rhs < 0) {
                  continue;  // off-diagonal redundant pair outside Xi's support
               }
               for(int32_t child : child_seqs) {
                  const int32_t lhs_coord = slot == 0 ? index_of(child, tau) : index_of(tau, child);
                  if(lhs_coord < 0) {
                     continue;
                  }
                  entry.lhs_terms.emplace_back(lhs_coord, 1.);
               }
               m_constraints.push_back(std::move(entry));
            }
         }
      }
   }

   ///////////////////////////////////////
   /// DECOMPOSE (paper Algorithm 2)   ///
   //////////////////////////////////////

   void _decompose_root()
   {
      m_filled.assign(m_rel_coords.size(), 0);
      const int32_t root = index_of(k_empty_sequence_id, k_empty_sequence_id);
      m_filled[size_t(root)] = 1;
      _decompose(k_empty_sequence_id, k_empty_sequence_id);
   }

   /// existence of an opposing infoset of parent sequence 'tau' connected to
   /// 'iid' -- the criticality notion of Definition 5
   [[nodiscard]] bool _is_critical(size_t slot, uint32_t iid, int32_t tau) const
   {
      const size_t other = size_t{1} - slot;
      const Player other_player = m_players[other];
      for(auto j :
          std::views::iota(uint32_t{0}, uint32_t(m_oracle.structure(other_player).size()))) {
         if(m_infoset_parents[other][j] != tau) {
            continue;
         }
         const bool hit = slot == 0 ? m_connected[iid][j] : m_connected[j][iid];
         if(hit) {
            return true;
         }
      }
      return false;
   }

   /// all own children sequences of 'infoset' that carry relevant pair entries
   /// ('tau' being the fixed opponent sequence of the caller's frame)
   [[nodiscard]] std::vector< int32_t >
   _relevant_children_sequences(size_t slot, uint32_t iid, int32_t tau) const
   {
      const Player player = m_players[slot];
      std::vector< int32_t > out;
      const size_t n_actions = m_oracle.structure(player).actions[iid].size();
      for(auto aid : std::views::iota(uint32_t{0}, uint32_t(n_actions))) {
         if(not has_sequence_id(player, iid, aid)) {
            continue;
         }
         const int32_t child = sequence_id(player, iid, aid);
         if(slot == 0 ? relevant(child, tau) : relevant(tau, child)) {
            out.push_back(child);
         }
      }
      return out;
   }

   void _mark_filled(int32_t coord) { m_filled[size_t(coord)] = 1; }
   void _assert_fresh_write(int32_t coord)
   {
      if(m_filled[size_t(coord)] != 0) {
         throw std::logic_error("EFCP decomposition: duplicate write into a filled entry");
      }
   }
   void _assert_readable(int32_t coord)
   {
      if(coord < 0 or m_filled[size_t(coord)] == 0) {
         throw std::logic_error("EFCP decomposition: read of an unfilled entry");
      }
   }

   /// infosets of 'slot' hanging directly below sequence 'parent'
   [[nodiscard]] std::vector< uint32_t > _infosets_with_parent(size_t slot, int32_t parent) const
   {
      std::vector< uint32_t > out;
      const size_t n = m_oracle.structure(m_players[slot]).size();
      for(auto iid : std::views::iota(uint32_t{0}, uint32_t(n))) {
         if(m_infoset_parents[slot][iid] == parent) {
            out.push_back(iid);
         }
      }
      return out;
   }

   /// loop-18 candidates: opponent infosets whose parent sequence lies at or
   /// below 'base_tau', returned top-down (ascending parent depth, stable ids)
   [[nodiscard]] std::vector< std::pair< uint32_t, int32_t > >
   _opponent_candidates_top_down(size_t slot, int32_t base_tau) const
   {
      std::vector< std::pair< uint32_t, int32_t > > cand;
      const Player player = m_players[slot];
      const size_t n = m_oracle.structure(player).size();
      for(auto iid : std::views::iota(uint32_t{0}, uint32_t(n))) {
         const int32_t parent = m_infoset_parents[slot][iid];
         if(not sequence_is_at_or_below(player, parent, base_tau)) {
            continue;
         }
         cand.emplace_back(iid, parent);
      }
      std::ranges::stable_sort(cand, [&](const auto& a, const auto& b) {
         return sequence_depth(player, a.second) < sequence_depth(player, b.second);
      });
      return cand;
   }

   void _decompose(int32_t s1, int32_t s2)
   {
      // ---- critical player selection (corollary of paper Proposition 2) ----
      std::array< std::vector< uint32_t >, 2 > critical{};
      for(auto side : std::views::iota(size_t{0}, size_t{2})) {
         const int32_t own_tau = side == 0 ? s1 : s2;
         const int32_t opp_tau = side == 0 ? s2 : s1;
         for(uint32_t iid : _infosets_with_parent(side, own_tau)) {
            if(_is_critical(side, iid, opp_tau)) {
               critical[side].push_back(iid);
            }
         }
      }
      size_t star = 0;
      if(critical[0].size() > 1) {
         if(critical[1].size() > 1) {
            throw std::logic_error(
               "EFCP: neither player is critical for a relevant sequence pair "
               "-- impossible without chance moves (paper Proposition 2)"
            );
         }
         star = 1;
      }
      const size_t other = size_t{1} - star;

      // ---- loop 1 (lines 5-17): split the star side below the frame ----
      {
         const int32_t own_tau = star == 0 ? s1 : s2;
         const int32_t opp_tau = star == 0 ? s2 : s1;
         for(uint32_t iid : _infosets_with_parent(star, own_tau)) {
            auto children = _relevant_children_sequences(star, iid, opp_tau);
            if(children.empty()) {
               continue;
            }
            CircuitOp op{};
            op.kind = CircuitOp::Kind::fill;
            op.frame_s1 = s1;
            op.frame_s2 = s2;
            op.source = index_of(s1, s2);
            _assert_readable(op.source);
            op.rm_index = int32_t(m_leaf_counter++);
            for(int32_t child : children) {
               const int32_t coord = star == 0 ? index_of(child, s2) : index_of(s1, child);
               _assert_fresh_write(coord);
               op.children.push_back(coord);
            }
            m_ops.push_back(std::move(op));
            for(int32_t coord : m_ops.back().children) {
               _mark_filled(coord);
            }
            for(int32_t child : children) {
               if(star == 0) {
                  _decompose(child, s2);
               } else {
                  _decompose(s1, child);
               }
            }
         }
      }

      // ---- loop 2 (lines 18-39): fill opponent-descendant columns ----
      {
         const int32_t sigma_star = star == 0 ? s1 : s2;
         const int32_t sigma_other = star == 0 ? s2 : s1;
         const uint32_t sigma_star_owner = sigma_star == k_empty_sequence_id
                                              ? std::numeric_limits< uint32_t >::max()
                                              : m_sequences[star][size_t(sigma_star)].owner_infoset;
         for(const auto& [jid, j_parent] : _opponent_candidates_top_down(other, sigma_other)) {
            // sigma_star rightleftarrows-J relevance (Definition 2, I-vs-seq form)
            if(sigma_star != k_empty_sequence_id) {
               const bool hit = star == 0 ? m_connected[sigma_star_owner][jid]
                                          : m_connected[jid][sigma_star_owner];
               if(not hit) {
                  continue;
               }
            }
            const Player opp = m_players[other];
            std::vector< int32_t > children;
            const size_t n_actions = m_oracle.structure(opp).actions[jid].size();
            for(auto aid : std::views::iota(uint32_t{0}, uint32_t(n_actions))) {
               if(has_sequence_id(opp, jid, aid)) {
                  children.push_back(sequence_id(opp, jid, aid));
               }
            }
            if(children.empty()) {
               continue;
            }
            const bool sum_branch = critical[star].size() == 1u and [&] {
               const uint32_t star_iid = critical[star].front();
               return star == 0 ? m_connected[star_iid][jid] : m_connected[jid][star_iid];
            }();
            if(sum_branch) {
               const uint32_t star_iid = critical[star].front();
               // operand generator: star-side children of the critical infoset
               auto make_source = [&](int32_t star_child, int32_t opp_child) {
                  return star == 0 ? index_of(star_child, opp_child)
                                   : index_of(opp_child, star_child);
               };
               const Player star_player = m_players[star];
               for(int32_t opp_child : children) {
                  CircuitOp op{};
                  op.kind = CircuitOp::Kind::sum;
                  op.frame_s1 = s1;
                  op.frame_s2 = s2;
                  op.target = star == 0 ? index_of(s1, opp_child) : index_of(opp_child, s2);
                  _assert_fresh_write(op.target);
                  const size_t
                     n_star_actions = m_oracle.structure(star_player).actions[star_iid].size();
                  for(auto bid : std::views::iota(uint32_t{0}, uint32_t(n_star_actions))) {
                     if(not has_sequence_id(star_player, star_iid, bid)) {
                        continue;
                     }
                     const int32_t star_child = sequence_id(star_player, star_iid, bid);
                     const int32_t src = make_source(star_child, opp_child);
                     _assert_readable(src);
                     op.sources.push_back(src);
                  }
                  if(op.sources.empty()) {
                     throw std::logic_error("EFCP decomposition: SumSimplex without operands");
                  }
                  m_ops.push_back(std::move(op));
                  _mark_filled(m_ops.back().target);
               }
            } else {
               CircuitOp op{};
               op.kind = CircuitOp::Kind::fill;
               op.frame_s1 = s1;
               op.frame_s2 = s2;
               op.source = star == 0 ? index_of(s1, j_parent) : index_of(j_parent, s2);
               _assert_readable(op.source);
               op.rm_index = int32_t(m_leaf_counter++);
               for(int32_t opp_child : children) {
                  const int32_t coord = star == 0 ? index_of(s1, opp_child)
                                                  : index_of(opp_child, s2);
                  _assert_fresh_write(coord);
                  op.children.push_back(coord);
               }
               m_ops.push_back(std::move(op));
               for(int32_t coord : m_ops.back().children) {
                  _mark_filled(coord);
               }
            }
         }
      }
   }

   void _validate_decomposition()
   {
      if(size_t(std::ranges::count(m_filled, char(1))) != m_rel_coords.size()) {
         throw std::logic_error("EFCP: decomposition finished before covering every relevant pair");
      }
   }
};

/**
 * @brief quality report of an EFCP correlation plan.
 *
 * 'efce_gap' equals v*(x̄) -- the largest utility increase any trigger agent
 * could realize by defecting once against the mediation induced by the
 * reported correlation plan; zero means an exact EFCE. 'feasibility_residual'
 * audits the FULL (redundant-inclusive) von Stengel-Forges system, measuring
 * how completely the reduced constraint set of the decomposition implies it.
 */
struct EFCPMetrics {
   double feasibility_residual_linf = 0.;
   double feasibility_residual_l1 = 0.;
   double efce_gap = 0.;
   double social_welfare = 0.;
   player_hashmap< double > player_values{};

   struct TriggerGap {
      size_t player_slot = 0;
      uint32_t infoset = 0;
      uint32_t action = 0;
      double triggered_value = 0.;
      double obedient_value = 0.;
   };
   std::vector< TriggerGap > triggers{};
};

/**
 * @brief EFCP self-play solver: mediator regret circuit over Xi vs deviator
 *        hull-circuit over trigger responses, alternating-updates CFR+
 *        (regret-matching-plus leaves everywhere, linear averaging), exactly
 *        following Farina et al. NeurIPS 2019 Sections 3-5.
 */
template < concepts::fosg Env >
class EFCP {
   using Oracle = SequenceFormOracle< Env >;
   using Space = CorrelationPlanSpace< Env >;

   /// one deviator component Q~_sigma : the responding player's treeplex
   /// restricted to infosets whose parent sequence lies at-or-below the
   /// trigger's parent, expressed as its own FillSimplex chain (Section 3.1)
   struct TriggerBlock {
      size_t slot = 0;
      uint32_t infoset = 0;
      uint32_t action = 0;
      int32_t sigma_sid = k_empty_sequence_id;
      int32_t root_sid = k_empty_sequence_id;

      struct ChainOp {
         int32_t source_sid = k_empty_sequence_id;
         std::vector< int32_t > child_sids{};
         uint32_t rm_index = 0;
      };
      std::vector< ChainOp > chain{};
      std::vector< SimplexRMPlus > kernels{};
      /// latest recommendation per kernel (aligned with 'chain')
      std::vector< std::vector< double > > cached_policy{};
      /// current realization masses, flat over the responder's sequence registry
      std::vector< double > realization{};
      /// dense position of the component inside the hull mixing simplex
      size_t lambda_index = 0;
   };

  public:
   EFCP(const Env& env, const typename Env::world_state_type& root_state)
       : m_env(env),
         m_root(utils::static_unique_ptr_downcast< typename Env::world_state_type >(
            utils::clone_any_way(root_state)
         )),
         m_oracle(m_env, *m_root),
         m_space(m_oracle)
   {
      constexpr bool chance_free = not requires {
         typename Env::chance_outcome_type;
      } or std::is_same_v< typename Env::chance_outcome_type, void > or (requires {
         Env::stochasticity();
      } and Env::stochasticity() == Stochasticity::deterministic);
      if constexpr(! chance_free) {
         throw std::invalid_argument(
            "EFCP: environments with chance moves are outside the scope of "
            "extensive-form correlation-plan solvers"
         );
      }
      _initialize_reward_scale();
      _build_mediator_circuit();
      _build_deviator_circuit();
   }

   ///////////////////////////////
   /// training API           ///
   //////////////////////////////

   void iterate(size_t n_iters)
   {
      for([[maybe_unused]] auto cycle : std::views::iota(size_t{0}, n_iters)) {
         ++m_iteration;
         const double avg_weight = double(m_iteration);  // linear averaging
         // ---- mediator move against the currently played deviation ----
         auto mediator_loss = _mediator_losses(m_lambda_weights);
         for(double& entry : mediator_loss) {
            entry /= m_reward_scale;
         }
         _recommend_mediator();  // materializes xi before observing
         _observe_mediator(mediator_loss);
         _accumulate_average(avg_weight);

         // ---- alternating deviator refresh against the new mediator plan ----
         const auto& xi_next = _recommend_mediator();
         for(auto block_idx : std::views::iota(size_t{0}, m_blocks.size())) {
            _observe_trigger_component(block_idx, xi_next);
            _refresh_trigger_component(block_idx);
         }
         // hull mixing weights move once per completed round
         m_lambda_weights = m_lambda_kernel.recommend();
      }
   }

   [[nodiscard]] size_t iteration() const { return m_iteration; }

   /**
    * @brief linearly averaged correlation plan xi_bar^T (dimension:
    *        space().relevant_pair_count(), coordinate convention inherited).
    */
   [[nodiscard]] std::vector< double > average_plan() const
   {
      std::vector< double > out;
      if(m_average_weight_sum <= 0.) {
         out.assign(m_space.relevant_pair_count(), 0.);
         return out;
      }
      out.reserve(m_avg_plan.size());
      for(double acc : m_avg_plan) {
         out.push_back(acc / m_average_weight_sum);
      }
      return out;
   }

   /// the most recent recommended iterate xi^T (always feasible by construction)
   [[nodiscard]] std::vector< double > current_plan() const { return m_current_plan; }

   [[nodiscard]] const Space& space() const { return m_space; }

   /**
    * @brief trains a synthetic loss vector through the mediator circuit WITHOUT
    *        touching the averaging accumulator -- the deterministic hook used by
    *        unit tests to probe circuit-level scaled-extension invariants
    *        (feasibility of iterates, conservation of simplex mass flow).
    */
   void debug_observe_synthetic_losses(const std::vector< double >& loss)
   {
      (void) _recommend_mediator();
      _observe_mediator(loss);
   }

   ////////////////////////////////////
   /// evaluation                  ///
   /////////////////////////////////

   /// evaluates either the averaged plan ('averaged') or the latest iterate
   [[nodiscard]] EFCPMetrics evaluate(bool averaged = true) const
   {
      const std::vector< double > plan = averaged ? average_plan() : [&] {
         auto copy = m_current_plan;
         return copy;
      }();
      return evaluate_plan(plan);
   }

   [[nodiscard]] EFCPMetrics evaluate_plan(const std::vector< double >& plan) const
   {
      EFCPMetrics metrics{};
      const auto residual = m_space.feasibility_residual(plan);
      metrics.feasibility_residual_linf = residual.linf;
      metrics.feasibility_residual_l1 = residual.l1;

      const size_t n_terminals = m_oracle.terminal_count();
      for(auto slot : std::views::iota(size_t{0}, size_t{2})) {
         double value = 0.;
         for(auto z : std::views::iota(size_t{0}, n_terminals)) {
            const double prob = plan.at(size_t(m_space.terminal_pair_coordinate(z)));
            value += prob * m_oracle.terminal_reward(z, m_space.player(slot));
         }
         metrics.player_values[m_space.player(slot)] = value;
         metrics.social_welfare += value;
      }

      for(auto slot : std::views::iota(size_t{0}, size_t{2})) {
         for(const auto& [iid, aid] : m_triggers[slot]) {
            const auto entry = _evaluate_trigger_gap(plan, slot, iid, aid);
            metrics.triggers.push_back(EFCPMetrics::TriggerGap{
               .player_slot = slot,
               .infoset = iid,
               .action = aid,
               .triggered_value = entry.first,
               .obedient_value = entry.second});
            metrics.efce_gap = std::max(metrics.efce_gap, entry.first - entry.second);
         }
      }
      return metrics;
   }

  private:
   Env m_env;
   uptr< typename Env::world_state_type > m_root;
   Oracle m_oracle;
   Space m_space;

   // ---------------- mediator circuit ----------------
   std::vector< CircuitOp > m_ops{};
   std::vector< SimplexRMPlus > m_leaves{};
   std::vector< std::vector< double > > m_cached_leaf_policies{};
   std::vector< double > m_current_plan{};
   std::vector< double > m_avg_plan{};
   double m_average_weight_sum = 0.;
   size_t m_iteration = 0;

   // ---------------- deviator circuit ----------------
   std::array< std::vector< std::pair< uint32_t, uint32_t > >, 2 > m_triggers{};
   std::vector< TriggerBlock > m_blocks{};  // registry order matches m_triggers
   SimplexRMPlus m_lambda_kernel{};
   std::vector< double > m_lambda_weights{};

   /// shared scalars / traversal caches
   double m_reward_scale = 1.;
   /// per player-slot, per terminal: the terminal's LAST own sequence id
   std::array< std::vector< int32_t >, 2 > m_terminal_last_sequence{};

   void _initialize_reward_scale()
   {
      double scale = 1e-12;
      for(auto z : std::views::iota(size_t{0}, m_oracle.terminal_count())) {
         for(auto slot : std::views::iota(size_t{0}, size_t{2})) {
            scale = std::max(scale, std::abs(m_oracle.terminal_reward(z, m_space.player(slot))));
         }
         // stash terminal last-sequences while traversing anyway
      }
      m_reward_scale = scale;
      for(auto slot : std::views::iota(size_t{0}, size_t{2})) {
         m_terminal_last_sequence[slot].assign(m_oracle.terminal_count(), k_empty_sequence_id);
         for(auto z : std::views::iota(size_t{0}, m_oracle.terminal_count())) {
            const auto& sig = m_oracle.terminal(z).signatures[slot];
            m_terminal_last_sequence[slot][z] = sig.empty() ? k_empty_sequence_id
                                                            : m_space.sequence_id(
                                                               m_space.player(slot),
                                                               sig.back().first,
                                                               sig.back().second
                                                            );
         }
      }
   }

   void _build_mediator_circuit()
   {
      m_ops = m_space.decomposition_ops();
      m_current_plan.assign(m_space.relevant_pair_count(), 0.);
      m_avg_plan.assign(m_space.relevant_pair_count(), 0.);
      m_leaves.reserve(m_space.leaf_unit_count());
      m_cached_leaf_policies.reserve(m_space.leaf_unit_count());
      for(const auto& op : m_ops) {
         if(op.kind != CircuitOp::Kind::fill) {
            continue;
         }
         auto& kernel = m_leaves.emplace_back();
         kernel.register_slots(op.children.size());
         m_cached_leaf_policies.push_back(kernel.recommend());
      }
   }

   void _build_deviator_circuit()
   {
      for(auto slot : std::views::iota(size_t{0}, size_t{2})) {
         const Player player = m_space.player(slot);
         const size_t n_infosets = m_oracle.structure(player).size();
         for(auto iid : std::views::iota(uint32_t{0}, uint32_t(n_infosets))) {
            const size_t n_actions = m_oracle.structure(player).actions[iid].size();
            for(auto aid : std::views::iota(uint32_t{0}, uint32_t(n_actions))) {
               if(not m_space.has_sequence_id(player, iid, aid)) {
                  continue;
               }
               m_triggers[slot].emplace_back(iid, aid);
            }
         }
      }
      m_lambda_kernel.register_slots(_total_trigger_count());

      size_t lambda_running = 0;
      for(auto slot : std::views::iota(size_t{0}, size_t{2})) {
         const Player player = m_space.player(slot);
         const size_t n_sequences = m_space.sequence_count(player);
         for(const auto& [iid, aid] : m_triggers[slot]) {
            const int32_t sigma_sid = m_space.sequence_id(player, iid, aid);
            auto& block = m_blocks.emplace_back();
            block.slot = slot;
            block.infoset = iid;
            block.action = aid;
            block.sigma_sid = sigma_sid;
            block.root_sid = m_space.sequence(player, sigma_sid).parent;
            block.realization.assign(n_sequences, 0.);
            block.lambda_index = lambda_running++;

            // response subtree of Q~_root: infosets whose parent lies below the
            // root, laid down top-down as a pure FillSimplex chain (Section 3.1)
            auto subtree = m_space.response_infosets(player, block.root_sid);
            std::ranges::stable_sort(subtree, [&](uint32_t lhs, uint32_t rhs) {
               return m_space.infoset_chain(player, lhs).size()
                      < m_space.infoset_chain(player, rhs).size();
            });
            for(uint32_t member : subtree) {
               typename TriggerBlock::ChainOp op{};
               op.source_sid = m_space.infoset_parent_sequence(player, member);
               const size_t n_actions = m_oracle.structure(player).actions[member].size();
               for(auto mid : std::views::iota(uint32_t{0}, uint32_t(n_actions))) {
                  if(not m_space.has_sequence_id(player, member, mid)) {
                     continue;
                  }
                  op.child_sids.push_back(m_space.sequence_id(player, member, mid));
               }
               if(op.child_sids.empty()) {
                  continue;
               }
               op.rm_index = uint32_t(block.kernels.size());
               auto& kernel = block.kernels.emplace_back();
               kernel.register_slots(op.child_sids.size());
               block.cached_policy.push_back(kernel.recommend());
               block.chain.push_back(std::move(op));
            }
            _refresh_trigger_component(m_blocks.size() - 1);
         }
      }
      // seed the hull mixing weights uniformly
      m_lambda_weights = m_lambda_kernel.recommend();
   }

   [[nodiscard]] size_t _total_trigger_count() const
   {
      return m_triggers[0].size() + m_triggers[1].size();
   }

   // ---------------------------------------------------------------
   /// mediator forward pass: materializes xi from the operation chain
   // ---------------------------------------------------------------
   [[nodiscard]] const std::vector< double >& _recommend_mediator()
   {
      std::ranges::fill(m_current_plan, 0.);
      const int32_t root = m_space.index_of(k_empty_sequence_id, k_empty_sequence_id);
      m_current_plan[size_t(root)] = 1.;
      for(const auto& op : m_ops) {
         switch(op.kind) {
            case CircuitOp::Kind::fill: {
               const double source_value = m_current_plan[size_t(op.source)];
               const auto& policy = m_cached_leaf_policies[size_t(op.rm_index)];
               for(auto idx : std::views::iota(size_t{0}, op.children.size())) {
                  m_current_plan[size_t(op.children[idx])] = source_value * policy[idx];
               }
               break;
            }
            case CircuitOp::Kind::sum: {
               double total = 0.;
               for(int32_t src : op.sources) {
                  total += m_current_plan[size_t(src)];
               }
               m_current_plan[size_t(op.target)] = total;
               break;
            }
         }
      }
      return m_current_plan;
   }

   /// backward fold of Algorithm 3 ObserveLoss over the shared loss buffer
   void _observe_mediator(const std::vector< double >& loss)
   {
      if(loss.size() != m_current_plan.size()) {
         throw std::invalid_argument("EFCP: synthetic loss dimension mismatch");
      }
      std::vector< double > buffer = loss;
      std::vector< double > child_scratch;
      for(size_t op_idx = m_ops.size(); op_idx-- > 0;) {
         const CircuitOp& op = m_ops[op_idx];
         if(op.kind == CircuitOp::Kind::fill) {
            auto& kernel = m_leaves[size_t(op.rm_index)];
            const auto& policy = m_cached_leaf_policies[size_t(op.rm_index)];
            double weighted_child_mass = 0.;
            child_scratch.clear();
            for(auto idx : std::views::iota(size_t{0}, op.children.size())) {
               const double child_loss = buffer[size_t(op.children[idx])];
               weighted_child_mass += policy[idx] * child_loss;
               child_scratch.push_back(child_loss);
            }
            kernel.observe_span(child_scratch.data(), child_scratch.size());
            buffer[size_t(op.source)] += weighted_child_mass;
         } else {
            for(int32_t src : op.sources) {
               buffer[size_t(src)] += buffer[size_t(op.target)];
            }
         }
      }
      // refresh recommendations (CFR+ clamp happens here) and their caches
      for(auto k : std::views::iota(size_t{0}, m_leaves.size())) {
         m_cached_leaf_policies[k] = m_leaves[k].recommend();
      }
   }

   void _accumulate_average(double weight)
   {
      for(auto idx : std::views::iota(size_t{0}, m_current_plan.size())) {
         m_avg_plan[idx] += weight * m_current_plan[idx];
      }
      m_average_weight_sum += weight;
   }

   // ---------------------------------------------------------------
   /// deviator side
   // ---------------------------------------------------------------

   /// scatter-cache buffer over one player's sequence registry
   [[nodiscard]] std::vector< double > _zero_gradient_buffer(size_t slot) const
   {
      return std::vector< double >(m_space.sequence_count(m_space.player(slot)), 0.);
   }

   /**
    * folds the component's counterfactual losses (gain-side scatter only -- the
    * obedient side is constant in the deviation strategy) into its treeplex
    * kernels, and reports the trigger's total value into the hull mixing
    * kernel. The component gradient is compensated by its own current mixing
    * mass (losses are consumed in q-space rather than lambda*q-space -- the
    * convex-hull circuit's rescale-to-simplex compensation); starved components
    * freeze until their mass revives.
    */
   void _observe_trigger_component(size_t block_idx, const std::vector< double >& xi)
   {
      TriggerBlock& block = m_blocks[block_idx];
      const Player player = m_space.player(block.slot);
      const size_t other_slot = size_t{1} - block.slot;
      const size_t lam_idx = block.lambda_index;

      auto gradient = _zero_gradient_buffer(block.slot);
      double total_value = 0.;

      for(auto z : m_space.terminals_through(player, block.infoset)) {
         const double reward = m_oracle.terminal_reward(z, m_space.player(block.slot));
         if(reward == 0.) {
            continue;
         }
         const int32_t tau = m_terminal_last_sequence[block.slot][z];
         const int32_t tau_other = m_terminal_last_sequence[other_slot][z];
         const int32_t gain_coord = block.slot == 0 ? m_space.index_of(block.sigma_sid, tau_other)
                                                    : m_space.index_of(tau_other, block.sigma_sid);
         gradient[size_t(tau)] += reward * xi[size_t(gain_coord)];
         total_value += reward * xi[size_t(gain_coord)] * block.realization[size_t(tau)];
      }
      for(auto z : m_space.terminals_by_step(player, block.infoset, block.action)) {
         const double reward = m_oracle.terminal_reward(z, m_space.player(block.slot));
         total_value -= reward * xi[size_t(m_space.terminal_pair_coordinate(z))];
      }

      m_lambda_kernel.observe(lam_idx, -(total_value / m_reward_scale));

      const double lambda_mass = m_lambda_weights[lam_idx];
      if(lambda_mass <= 0.) {
         return;  // hull-circuit corner: starved component freezes
      }
      for(double& entry : gradient) {
         entry /= (m_reward_scale * lambda_mass);
      }
      for(size_t op_idx = block.chain.size(); op_idx-- > 0;) {
         const auto& op = block.chain[op_idx];
         auto& kernel = block.kernels[op.rm_index];
         const auto& policy = block.cached_policy[op.rm_index];
         double weighted = 0.;
         for(auto idx : std::views::iota(size_t{0}, op.child_sids.size())) {
            weighted += policy[idx] * gradient[size_t(op.child_sids[idx])];
            kernel.observe(idx, gradient[size_t(op.child_sids[idx])]);
         }
         gradient[size_t(op.source_sid)] += weighted;
      }
   }

   /// recompute recommendation + realization snapshot of one component
   void _refresh_trigger_component(size_t block_idx)
   {
      TriggerBlock& block = m_blocks[block_idx];
      std::ranges::fill(block.realization, 0.);
      block.realization[size_t(block.root_sid)] = 1.;
      for(auto& op : block.chain) {
         auto policy = block.kernels[op.rm_index].recommend();
         for(auto idx : std::views::iota(size_t{0}, op.child_sids.size())) {
            block.realization[size_t(op.child_sids[idx])] = block.realization[size_t(op.source_sid)]
                                                            * policy[idx];
         }
         block.cached_policy[op.rm_index] = std::move(policy);
      }
   }

   // ---------------------------------------------------------------
   /// mediator loss vector assembly for the current deviator play
   // ---------------------------------------------------------------
   [[nodiscard]] std::vector< double > _mediator_losses(const std::vector< double >& lambda_weights
   ) const
   {
      std::vector< double > buffer(m_space.relevant_pair_count(), 0.);
      for(const auto& block : m_blocks) {
         const Player player = m_space.player(block.slot);
         const size_t other_slot = size_t{1} - block.slot;
         const double lam = lambda_weights[block.lambda_index];
         if(lam == 0.) {
            continue;
         }
         for(auto z : m_space.terminals_through(player, block.infoset)) {
            const double reward = m_oracle.terminal_reward(z, m_space.player(block.slot));
            if(reward == 0.) {
               continue;
            }
            const int32_t tau = m_terminal_last_sequence[block.slot][z];
            const int32_t tau_other = m_terminal_last_sequence[other_slot][z];
            const int32_t gain_coord = block.slot == 0
                                          ? m_space.index_of(block.sigma_sid, tau_other)
                                          : m_space.index_of(tau_other, block.sigma_sid);
            buffer[size_t(gain_coord)] += lam * reward * block.realization[size_t(tau)];
         }
         for(auto z : m_space.terminals_by_step(player, block.infoset, block.action)) {
            const double reward = m_oracle.terminal_reward(z, m_space.player(block.slot));
            buffer[size_t(m_space.terminal_pair_coordinate(z))] -= lam * reward;
         }
      }
      return buffer;
   }

   /**
    * paper Equation (2) split for one trigger sigma_hat=(I,a): returns
    * (best triggered continuation value, obedient continuation value) evaluated
    * against the given correlation plan, deviating continuations optimized over
    * the reduced plans that can still reach I (plan enumeration reuses the
    * SequenceFormOracle's reduced plan space).
    */
   [[nodiscard]] std::pair< double, double > _evaluate_trigger_gap(
      const std::vector< double >& plan,
      size_t slot,
      uint32_t infoset,
      uint32_t action
   ) const
   {
      const Player player = m_space.player(slot);
      const size_t other_slot = size_t{1} - slot;
      const int32_t sigma_sid = m_space.sequence_id(player, infoset, action);

      double obedient_value = 0.;
      for(auto z : m_space.terminals_by_step(player, infoset, action)) {
         obedient_value += m_oracle.terminal_reward(z, player)
                           * plan[size_t(m_space.terminal_pair_coordinate(z))];
      }

      // plans able to receive the sigma_hat recommendation: they agree with
      // every strict ancestor decision leading to I (behavior at I is REPLACED
      // by the deviation)
      const auto& ancestor_chain = m_space.infoset_chain(player, infoset);
      double best_triggered = -std::numeric_limits< double >::infinity();
      for(const Plan& candidate : m_oracle.reduced_plans(player)) {
         const bool reaches = std::ranges::all_of(
            ancestor_chain,
            [&](const std::pair< uint32_t, uint32_t >& step) {
               return candidate.at(step.first) == step.second;
            }
         );
         if(not reaches) {
            continue;
         }
         double value = 0.;
         for(auto z : m_space.terminals_through(player, infoset)) {
            bool consistent = true;
            for(const auto& [iid, aid] : m_oracle.terminal(z).signatures[slot]) {
               if(m_space.infoset_at_or_below_trigger(player, iid, infoset)
                  and candidate.at(iid) != aid) {
                  consistent = false;
                  break;
               }
            }
            if(not consistent) {
               continue;
            }
            const int32_t tau_other = m_terminal_last_sequence[other_slot][z];
            const int32_t gain_coord = slot == 0 ? m_space.index_of(sigma_sid, tau_other)
                                                 : m_space.index_of(tau_other, sigma_sid);
            value += m_oracle.terminal_reward(z, player) * plan[size_t(gain_coord)];
         }
         best_triggered = std::max(best_triggered, value);
      }
      if(best_triggered == -std::numeric_limits< double >::infinity()) {
         best_triggered = 0.;
      }
      return {best_triggered, obedient_value};
   }
};

}  // namespace nor::rm::correlated

#endif  // NOR_RM_CORRELATED_EFCP_HPP
