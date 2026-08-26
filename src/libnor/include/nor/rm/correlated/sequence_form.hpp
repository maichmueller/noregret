
#ifndef NOR_RM_CORRELATED_SEQUENCE_FORM_HPP
#define NOR_RM_CORRELATED_SEQUENCE_FORM_HPP

#include <algorithm>
#include <cassert>
#include <cstdint>
#include <functional>
#include <limits>
#include <ranges>
#include <stdexcept>
#include <unordered_map>
#include <unordered_set>
#include <utility>
#include <variant>
#include <vector>

#include "common/common.hpp"
#include "nor/concepts.hpp"
#include "nor/fosg_traits.hpp"
#include "nor/game_defs.hpp"
#include "nor/rm/forest.hpp"
#include "nor/type_defs.hpp"
#include "nor/utils/utils.hpp"

namespace nor::rm::correlated {

/// a normal-form plan: local action index per registered infoset of the owning player
/// (indexed by infoset id). Canonicalized plans carry 'unassigned_plan_entry' at
/// infosets that are unreachable under the plan itself.
using Plan = std::vector< uint32_t >;

inline constexpr uint32_t unassigned_plan_entry = std::numeric_limits< uint32_t >::max();

struct plan_hasher {
   [[nodiscard]] size_t operator()(const Plan& plan) const noexcept
   {
      size_t seed = 0;
      for(uint32_t entry : plan) {
         common::hash_combine(seed, entry);
      }
      return seed;
   }
};

/**
 * @brief Per-player sequence-form machinery for correlated-equilibrium learning
 *        (Celli, Marchesi, Bianchi & Gatti, "Learning to Correlate in Multi-Player
 *        General-Sum Sequential Games", NeurIPS 2019, arXiv:1910.06228).
 *
 * A single full-tree traversal lazily builds, per instance:
 *   - the registry of every player's information sets (in precedence order, i.e. the
 *     linear extension of the infoset precedence relation induced by DFS first
 *     encounter) together with each infoset's legal actions,
 *   - the terminal history registry holding per-player rewards, the accumulated chance
 *     probability and every player's DECISION SIGNATURE: the ordered (infoset, local
 *     action index) pairs that player realizes on the path to that terminal.
 *
 * On top of these structures the class implements
 *   - Algorithm 2 of the paper (NF-Strategy-Reconstruction): the polynomial-time
 *     greedy reconstruction of a normal-form strategy over reduced plans that is
 *     REALIZATION EQUIVALENT to a given behavioral strategy,
 *   - verification of realization equivalence between a behavioral strategy and its
 *     reconstruction,
 *   - enumeration of the reduced normal-form plan space of each player (the deviation
 *     candidate set of the CCE-gap evaluator),
 *   - memoized terminal-membership bitmasks of plans (used to turn joint distributions
 *     over plan tuples into expectations over terminal histories).
 *
 * MEMORY NOTES: the averaged joint distribution over plan tuples lives in JointDistribution
 * and is sparse -- keyed by tuples that received nonzero mass -- but its worst-case
 * support is the product |S_1| x ... x |S_n| of the per-player REDUCED plan spaces and
 * each iteration may add up to prod_i supp(x_i^t) fresh tuples. This is harmless for
 * the targeted testbeds (kuhn poker, small goofspiel, shapley's game, centipede) but
 * the class enforces a hard cap on per-player reduced-plan counts so oversized games
 * fail loudly instead of exhausting memory.
 */
template < typename Env >
   requires concepts::fosg< Env >
class SequenceFormOracle {
  public:
   ////////////////////////////
   /// public typedefs      ///
   ////////////////////////////

   using env_type = Env;
   using world_state_type = auto_world_state_type< Env >;
   using info_state_type = auto_info_state_type< Env >;
   using observation_type = auto_observation_type< Env >;
   using action_type = auto_action_type< Env >;
   using chance_outcome_type = std::conditional_t<
      std::is_same_v< auto_chance_outcome_type< Env >, void >,
      std::monostate,
      auto_chance_outcome_type< Env > >;
   using action_variant_type = auto_action_variant_type< Env >;

   /// alias re-export of the namespace-level plan representation
   using Plan = correlated::Plan;
   static constexpr uint32_t unassigned_plan_entry = correlated::unassigned_plan_entry;

   /// decision-point registry of one player
   struct PlayerStructure {
      /// infosets in precedence order (DFS first encounter == valid linear extension)
      std::vector< info_state_type > infostates;
      /// legal actions per infoset, aligned with 'infostates'
      std::vector< std::vector< action_type > > actions;

      [[nodiscard]] size_t size() const { return infostates.size(); }
   };

   /// one terminal history of the game tree
   struct TerminalRecord {
      /// reward of every root participant (aligned with players(); chance excluded)
      std::vector< double > rewards;
      /// product of all chance probabilities along the path (1 for deterministic games)
      double chance_prob = 1.;
      /// per root player: the (infoset id, local action index) decisions realized on
      /// the path, in path order
      std::vector< std::vector< std::pair< uint32_t, uint32_t > > > signatures;
   };

   /// a normal-form strategy over (reduced) plans: (plan, probability mass) pairs
   using NormalFormStrategy = std::vector< std::pair< Plan, double > >;

   /// hard cap on enumerated reduced plans per player (memory guard, see class docs)
   static constexpr size_t max_reduced_plans_per_player = size_t(1) << 20;

   ////////////////////////////
   /// construction         ///
   ////////////////////////////

   SequenceFormOracle(const Env& env, const world_state_type& root)
       : m_env(env),
         m_root_state(
            utils::static_unique_ptr_downcast< world_state_type >(utils::clone_any_way(root))
         ),
         m_players(std::invoke([&] {
            auto roster = env.players(root);
            std::erase(roster, Player::chance);
            return roster;
         }))
   {
      for(auto player : m_players) {
         m_player_slot.emplace(player, m_structures.size());
         m_structures.emplace_back();
         m_infostate_ids.emplace_back();
         m_plan_masks.emplace_back();
         m_reduced_plans.emplace_back();
         m_cover.emplace_back();
      }
      build_tree(*m_root_state);
      for(size_t slot : std::views::iota(size_t{0}, m_structures.size())) {
         build_cover_lists(slot);
         m_reduced_plans.at(slot) = enumerate_reduced_plans_of(slot);
         if(m_reduced_plans.at(slot).size() > max_reduced_plans_per_player) {
            throw std::length_error(
               "SequenceFormOracle: player " + common::to_string(m_players.at(slot))
               + " has more than the supported number of reduced normal-form plans ("
               + std::to_string(m_reduced_plans.at(slot).size()) + " > "
               + std::to_string(max_reduced_plans_per_player)
               + "). The joint plan space of CFR-Jr/CFR-S averaging is the PRODUCT of the "
               + "per-player reduced plan spaces and exceeds the supported memory guard."
            );
         }
      }
   }

   SequenceFormOracle(const SequenceFormOracle&) = delete;
   SequenceFormOracle(SequenceFormOracle&&) = default;
   SequenceFormOracle& operator=(const SequenceFormOracle&) = delete;
   SequenceFormOracle& operator=(SequenceFormOracle&&) = default;
   ~SequenceFormOracle() = default;

   ////////////////////////////
   /// accessors            ///
   ////////////////////////////

   [[nodiscard]] const std::vector< Player >& players() const { return m_players; }
   [[nodiscard]] size_t player_count() const { return m_structures.size(); }
   [[nodiscard]] size_t player_slot(Player player) const { return m_player_slot.at(player); }
   [[nodiscard]] const PlayerStructure& structure(Player player) const
   {
      return m_structures.at(player_slot(player));
   }
   [[nodiscard]] size_t terminal_count() const { return m_terminals.size(); }
   [[nodiscard]] const TerminalRecord& terminal(size_t z) const { return m_terminals.at(z); }
   [[nodiscard]] double terminal_reward(size_t z, Player player) const
   {
      return m_terminals.at(z).rewards.at(player_slot(player));
   }
   /// the reduced normal-form plans of 'player'; built eagerly at construction
   [[nodiscard]] const std::vector< Plan >& reduced_plans(Player player) const
   {
      return m_reduced_plans.at(player_slot(player));
   }

   /**
    * @brief terminal-membership bitmask of 'plan' (bit z set iff z lies in Z(plan)).
    *        Memoized per player.
    */
   [[nodiscard]] const std::vector< uint64_t >& plan_mask(Player player, const Plan& plan) const
   {
      auto& cache = m_plan_masks.at(player_slot(player));
      if(auto found = cache.find(plan); found != cache.end()) {
         return found->second;
      }
      const size_t slot = player_slot(player);
      std::vector< uint64_t > mask(words_for(m_terminals.size()), 0);
      for(auto z : std::views::iota(size_t{0}, m_terminals.size())) {
         if(covers_plan(m_terminals.at(z).signatures.at(slot), plan)) {
            mask.at(z / 64) |= (uint64_t(1) << (z % 64));
         }
      }
      return cache.emplace(plan, std::move(mask)).first->second;
   }

   ////////////////////////////
   /// Algorithm 2          ///
   ////////////////////////////

   /**
    * @brief NF-Strategy-Reconstruction (Celli et al. 2019, Algorithm 2): computes a
    *        normal-form strategy that is realization equivalent to the given
    *        behavioral strategy.
    *
    * @param player the player whose strategy is reconstructed
    * @param behavioral invocable (Player, infoset id) -> action distribution aligned
    *        with the player's registered actions at that infoset
    *
    * The greedy plan selection expands the paper's argmax recursively as prescribed
    * by Appendix D (Lemmas 9-10): infosets are decided BOTTOM-UP in reverse
    * precedence order, scoring every candidate action by the minimum remaining omega
    * over Z(sigma-bar, I, a) -- the terminals realizing (I, a) that follow the
    * already-fixed decisions at all own infosets strictly below I. Dead terminals
    * inside that set disqualify the action, which keeps omega componentwise
    * nonnegative; dead terminals outside it are invisible (prefix-free scoring).
    * Every round assigns omega-bar = min_{z in Z(sigma-bar)} omega_z > 0 to the
    * selected plan and zeroes at least one terminal, so the loop terminates within
    * |Z| rounds (Theorem 4).
    *
    * NOTE: the returned plans are TOTAL assignments (every registered infoset carries a
    * local action index). Entries at infosets unreachable under the plan itself are
    * irrelevant: terminal coverage (covers_plan) only consults realized decisions, so
    * total assignments induce exactly the same coverage as their canonical reduced
    * counterparts without paying a canonicalization traversal per support element.
    */
   template < typename Behavioral >
   [[nodiscard]] NormalFormStrategy reconstruct(Player player, Behavioral&& behavioral) const
   {
      const size_t slot = player_slot(player);
      const auto& player_structure = m_structures.at(slot);
      const size_t n_terminals = m_terminals.size();

      // cache the queried behavioral distributions (one row per infoset)
      std::vector< std::vector< double > > rows(player_structure.size());
      for(auto iid : std::views::iota(size_t{0}, player_structure.size())) {
         rows.at(iid) = behavioral(player, iid);
         if(rows.at(iid).size() != player_structure.actions.at(iid).size()) {
            throw std::invalid_argument(
               "SequenceFormOracle::reconstruct: behavioral distribution size mismatch at "
               "infoset "
               + std::to_string(iid)
            );
         }
      }

      // tolerance for floating-point dust on the realization vector; products of many
      // behavioral probabilities leave ~1e-13 residues that must neither keep the
      // reconstruction loop alive nor poison plans as fake dead terminals
      constexpr double epsilon = 1e-12;

      // line 3: omega_z <- rho^pi_z -- the player-i-only realization factors: the
      // probability contributions of 'player's own decisions along z ("all other
      // players -- chance included -- play so as to reach z"). Chance weights cancel
      // on both sides of the realization-equivalence equation and are therefore NOT
      // part of omega; the reconstructed plan MASSES still sum to one by flow
      // conservation, but omega itself is a realization-form vector (Farina et al.
      // 2018, Definition 2), NOT a distribution over Z: its entries sum to
      // sum_z prod_{(I,a) in sig(z)} pi(I,a), which generally exceeds one whenever
      // several terminals share the player's decision prefix.
      std::vector< double > omega(n_terminals, 0.);
      for(auto z : std::views::iota(size_t{0}, n_terminals)) {
         double w = 1.;
         for(const auto& [iid, aid] : m_terminals.at(z).signatures.at(slot)) {
            w *= rows.at(iid).at(aid);
         }
         if(not (w >= 0.) or w > 1. + 1e-9) {
            throw std::domain_error(
               "SequenceFormOracle::reconstruct: malformed realization factor " + std::to_string(w)
               + " encountered (products of behavioral probabilities must lie in [0, 1])"
            );
         }
         omega.at(z) = w <= epsilon ? 0. : w;
      }
      double total = std::ranges::fold_left(omega, double(0.), std::plus{});

      NormalFormStrategy out;
      // Theorem 4: support size <= |Z|; slack for floating-point dust
      const size_t max_rounds = n_terminals + 16;
      // per-round agreement flags of every terminal with the plan currently being
      // built: 'consistent' holds iff the terminal realizes every already-decided
      // plan entry at its own visited infosets; after the bottom-up pass it flags
      // exactly Z(sigma-bar), the selected plan's terminal coverage
      std::vector< char > consistent(n_terminals, 1);

      for([[maybe_unused]] auto _ : std::views::iota(size_t{0}, max_rounds)) {
         if(total <= epsilon) {
            return out;
         }
         // lines 4-5: the argmax expanded recursively per Appendix D (Lemmas 9 and
         // 10): infosets are decided BOTTOM-UP in reverse precedence order; the score
         // of action a at infoset I is the minimum remaining omega over the terminals
         // that realize (I, a) and follow the already-fixed plan decisions at every
         // own infoset STRICTLY BELOW I -- the paper's set Z(sigma-bar, I, a). Dead
         // terminals inside that subtree disqualify the action (score zero), while
         // dead terminals outside it are invisible; this prefix-freeness is what
         // makes the greedy selection exact rather than heuristic.
         Plan assignment(player_structure.size(), 0);
         std::ranges::fill(consistent, 1);
         for(size_t step = player_structure.size(); step-- > 0;) {
            const uint32_t infoset_id = uint32_t(step);
            const auto& infoset_covers = m_cover.at(slot).at(infoset_id);
            const size_t n_actions = player_structure.actions.at(infoset_id).size();
            uint32_t best = 0;
            double best_score = -1.;
            for(auto a : std::views::iota(size_t{0}, n_actions)) {
               // minimum omega over the surviving witnesses of (I, a); witnesses
               // killed by a below-decision mismatch drop out (their disagreement
               // with sigma-bar below I excludes them from Z(sigma-bar, I, a))
               double score = std::numeric_limits< double >::infinity();
               for(uint32_t z : infoset_covers.at(a)) {
                  if(consistent.at(z) != 0) {
                     score = std::min(score, omega.at(z));
                  }
               }
               if(score > best_score) {
                  best_score = score;
                  best = uint32_t(a);
               }
            }
            assignment.at(infoset_id) = best;
            // kill the witnesses of every non-selected action: they contradict the
            // plan at 'infoset_id' and are therefore unreachable under any completion
            // for all REMAINING (higher) infosets
            for(auto a : std::views::iota(size_t{0}, n_actions)) {
               if(uint32_t(a) == best) {
                  continue;
               }
               for(uint32_t z : infoset_covers.at(a)) {
                  consistent.at(z) = 0;
               }
            }
         }
         // line 6: omega_bar = min omega over Z(selected plan) -- exactly the
         // terminals still flagged as consistent (they agree with the plan
         // everywhere); Lemma 10 guarantees positivity whenever omega != 0
         double omega_bar = std::numeric_limits< double >::infinity();
         size_t coverage = 0;
         for(auto z : std::views::iota(size_t{0}, n_terminals)) {
            if(consistent.at(z) != 0) {
               omega_bar = std::min(omega_bar, omega.at(z));
               ++coverage;
            }
         }
         if(coverage == 0 or not (omega_bar > epsilon)) {
            throw std::runtime_error(
               "SequenceFormOracle::reconstruct: greedy selection stalled at total residual "
               + std::to_string(total)
               + ". The behavioral strategy is malformed or numerical breakdown occurred."
            );
         }
         // lines 7-8: assign the mass to the selected plan and decrease omega
         out.emplace_back(assignment, omega_bar);
         for(auto z : std::views::iota(size_t{0}, n_terminals)) {
            if(consistent.at(z) != 0) {
               omega.at(z) = std::max(0., omega.at(z) - omega_bar);
               if(omega.at(z) <= epsilon) {
                  omega.at(z) = 0.;
               }
            }
         }
         total -= omega_bar * double(coverage);
      }
      throw std::runtime_error(
         "SequenceFormOracle::reconstruct: failed to zero omega within the Theorem-4 support "
         "bound; this indicates a numerical breakdown."
      );
   }

   /**
    * @brief verifies realization equivalence between a behavioral strategy and a
    *        reconstructed normal-form strategy: for every terminal z, the mass accumulated
    *        by plans covering z must equal the behavioral realization factor rho^pi_z,
    *        and the masses must form a probability distribution.
    */
   template < typename Behavioral >
   [[nodiscard]] bool verify_realization_equivalence(
      Player player,
      Behavioral&& behavioral,
      const NormalFormStrategy& strategy,
      double tolerance = 1e-9
   ) const
   {
      const size_t slot = player_slot(player);
      const size_t n_terminals = m_terminals.size();

      std::vector< std::vector< double > > rows(m_structures.at(slot).size());
      for(auto iid : std::views::iota(size_t{0}, rows.size())) {
         rows.at(iid) = behavioral(player, iid);
      }

      const double mass_sum = std::ranges::fold_left(
         strategy | std::views::values, double(0.), std::plus{}
      );
      if(std::abs(mass_sum - 1.) > tolerance) {
         return false;
      }
      if(std::ranges::any_of(strategy | std::views::values, [&](double mass) {
            return mass < -tolerance;
         })) {
         return false;
      }
      for(auto z : std::views::iota(size_t{0}, n_terminals)) {
         const auto& signature = m_terminals.at(z).signatures.at(slot);
         double lhs = 0.;
         for(const auto& [plan, mass] : strategy) {
            if(covers_plan(signature, plan)) {
               lhs += mass;
            }
         }
         double rhs = 1.;
         for(const auto& [iid, aid] : signature) {
            rhs *= rows.at(iid).at(aid);
         }
         if(std::abs(lhs - rhs) > tolerance) {
            return false;
         }
      }
      return true;
   }

   ////////////////////////////
   /// helpers              ///
   ////////////////////////////

   /// true iff the terminal signature (the player's own decisions realizing z) agrees
   /// with 'plan'. Sentinel entries never agree with a realized decision.
   [[nodiscard]] static bool
   covers_plan(const std::vector< std::pair< uint32_t, uint32_t > >& signature, const Plan& plan)
   {
      for(const auto& [iid, aid] : signature) {
         if(plan.at(iid) != aid) {
            return false;
         }
      }
      return true;
   }

   [[nodiscard]] static constexpr size_t words_for(size_t bits) { return (bits + 63) / 64; }

  private:
   ////////////////////////////
   /// traversal context    ///
   ////////////////////////////

   /// observation buffers plus per-player current infostates, maintained along a
   /// descent with the same mechanics the solvers use (so infostate VALUES coincide
   /// with the solver policy tables' keys)
   struct TraversalContext {
      player_hashmap< std::vector< std::pair< observation_type, observation_type > > >
         obs_buffers{};
      player_hashmap< sptr< info_state_type > > infostates{};

      [[nodiscard]] TraversalContext deep_copy() const
      {
         TraversalContext copy{};
         for(const auto& [player, buffer] : obs_buffers) {
            copy.obs_buffers.emplace(player, buffer);
         }
         for(const auto& [player, istate] : infostates) {
            copy.infostates.emplace(player, std::make_shared< info_state_type >(*istate));
         }
         return copy;
      }

      static TraversalContext initial(const Env& env, const world_state_type& root)
      {
         TraversalContext ctx{};
         for(auto player : env.players(root)) {
            if(player != Player::chance) {
               ctx.obs_buffers.emplace(
                  player, std::vector< std::pair< observation_type, observation_type > >{}
               );
               ctx.infostates.emplace(player, std::make_shared< info_state_type >(player));
            }
         }
         return ctx;
      }

      template < typename ActionOrOutcome >
      void advance(
         const Env& env,
         const world_state_type& curr_state,
         const ActionOrOutcome& action_or_outcome,
         const world_state_type& next_state
      )
      {
         next_infostate_and_obs_buffers_inplace(
            env, obs_buffers, infostates, curr_state, action_or_outcome, next_state
         );
      }
   };

   [[nodiscard]] uptr< world_state_type >
   child_state(const world_state_type& state, const action_variant_type& action_variant) const
   {
      auto next = utils::static_unique_ptr_downcast< world_state_type >(utils::clone_any_way(state)
      );
      std::visit(
         common::Overload{
            [&](const action_type& action) { m_env.transition(*next, action); },
            [&](const chance_outcome_type& outcome) {
               if constexpr(concepts::stochastic_env< Env >) {
                  m_env.transition(*next, outcome);
               } else {
                  throw std::logic_error(
                     "SequenceFormOracle: chance outcome in a deterministic environment"
                  );
               }
            }},
         action_variant
      );
      return next;
   }

   /// advances the traversal context across one edge, dispatching on the unwrapped
   /// variant payload (typed overloads keep deterministic environments instantiable:
   /// their 'monostate' chance payload must never reach the env API)
   void advance_variant(
      TraversalContext& ctx,
      const world_state_type& curr_state,
      const action_variant_type& action_variant,
      const world_state_type& next_state
   ) const
   {
      std::visit(
         common::Overload{
            [&](const action_type& action) { ctx.advance(m_env, curr_state, action, next_state); },
            [&](const chance_outcome_type& outcome) {
               if constexpr(concepts::stochastic_env< Env >) {
                  ctx.advance(m_env, curr_state, outcome, next_state);
               } else {
                  throw std::logic_error(
                     "SequenceFormOracle: chance outcome in a deterministic environment"
                  );
               }
            }},
         action_variant
      );
   }

   [[nodiscard]] std::vector< action_variant_type > child_actions(const world_state_type& state
   ) const
   {
      Player actor = m_env.active_player(state);
      if constexpr(concepts::stochastic_env< Env >) {
         if(actor == Player::chance) {
            return std::ranges::to< std::vector< action_variant_type > >(
               m_env.chance_actions(state)
               | std::views::transform([](const chance_outcome_type& outcome) {
                    return action_variant_type(outcome);
                 })
            );
         }
      }
      return std::ranges::to< std::vector< action_variant_type > >(
         m_env.actions(actor, state) | std::views::transform([](const action_type& action) {
            return action_variant_type(action);
         })
      );
   }

   ////////////////////////////
   /// tree registration    ///
   ////////////////////////////

   struct BuildData {
      TraversalContext ctx{};
      player_hashmap< std::vector< std::pair< uint32_t, uint32_t > > > paths{};
      double chance_prob = 1.;
   };

   void build_tree(const world_state_type& root)
   {
      BuildData init_data{
         .ctx = TraversalContext::initial(m_env, root), .paths = {}, .chance_prob = 1.};
      for(auto player : m_players) {
         init_data.paths.emplace(player, std::vector< std::pair< uint32_t, uint32_t > >{});
      }

      auto child_hook = [&](
                           const BuildData& data,
                           const action_variant_type* action_variant,
                           world_state_type* curr_state,
                           world_state_type* next_state
                        ) -> BuildData {
         // DEEP-copy the traversal context: the infostate holders are shared pointers
         // and 'update_infostate' mutates through them, so a shallow copy would let
         // sibling branches contaminate each other's observation histories
         BuildData next_data{
            .ctx = data.ctx.deep_copy(), .paths = data.paths, .chance_prob = data.chance_prob};
         Player actor = m_env.active_player(*curr_state);
         if(actor != Player::chance) {
            // decision edge: intern the decision infoset and extend the signature
            uint32_t iid = intern(
               actor, *next_data.ctx.infostates.at(actor), m_env.actions(actor, *curr_state)
            );
            uint32_t aid = std::visit(
               common::Overload{
                  [&](const action_type& action) {
                     const auto& actions = m_structures.at(player_slot(actor)).actions.at(iid);
                     auto found = std::ranges::find(actions, action);
                     if(found == actions.end()) {
                        throw std::logic_error(
                           "SequenceFormOracle: traversed action is not registered at its "
                           "infoset"
                        );
                     }
                     return uint32_t(std::distance(actions.begin(), found));
                  },
                  [&](const chance_outcome_type&) -> uint32_t {
                     throw std::logic_error("SequenceFormOracle: chance outcome at a decision edge"
                     );
                  }},
               *action_variant
            );
            next_data.paths.at(actor).emplace_back(iid, aid);
         } else {
            next_data.chance_prob *= std::visit(
               common::Overload{
                  [&](const action_type&) -> double {
                     throw std::logic_error("SequenceFormOracle: regular action at a chance edge");
                  },
                  [&](const chance_outcome_type& outcome) -> double {
                     if constexpr(concepts::stochastic_env< Env >) {
                        return m_env.chance_probability(*curr_state, outcome);
                     } else {
                        throw std::logic_error(
                           "SequenceFormOracle: chance outcome in a deterministic environment"
                        );
                     }
                  }},
               *action_variant
            );
         }
         advance_variant(next_data.ctx, *curr_state, *action_variant, *next_state);
         if(m_env.is_terminal(*next_state)) {
            TerminalRecord record{};
            record.chance_prob = next_data.chance_prob;
            for(auto player : m_players) {
               record.rewards.push_back(m_env.reward(player, *next_state));
               record.signatures.push_back(next_data.paths.at(player));
            }
            m_terminals.push_back(std::move(record));
         }
         return next_data;
      };

      forest::GameTreeTraverser< Env >{m_env}.walk(
         utils::static_unique_ptr_downcast< world_state_type >(utils::clone_any_way(root)),
         std::move(init_data),
         forest::TraversalHooks{.child_hook = std::move(child_hook)}
      );
   }

   /// registers 'infostate' (and its legal actions) if unseen and returns its id
   uint32_t
   intern(Player player, const info_state_type& infostate, std::vector< action_type > actions)
   {
      auto& id_map = m_infostate_ids.at(player_slot(player));
      if(auto found = id_map.find(infostate); found != id_map.end()) {
         return found->second;
      }
      auto& player_structure = m_structures.at(player_slot(player));
      const uint32_t id = uint32_t(player_structure.infostates.size());
      player_structure.infostates.push_back(infostate);
      player_structure.actions.push_back(std::move(actions));
      id_map.emplace(infostate, id);
      return id;
   }

   /// indexes, per (infoset, local action), the terminals whose signature realizes that
   /// decision -- the witness lists of the Algorithm-2 argmax recursion. Every list is
   /// non-empty because registered actions stem from traversed edges and every tree
   /// branch ends in a terminal.
   void build_cover_lists(size_t slot)
   {
      auto& covers = m_cover.at(slot);
      const auto& player_structure = m_structures.at(slot);
      covers.assign(player_structure.size(), std::vector< std::vector< uint32_t > >{});
      for(auto iid : std::views::iota(size_t{0}, player_structure.size())) {
         covers.at(iid).assign(player_structure.actions.at(iid).size(), {});
      }
      for(auto z : std::views::iota(size_t{0}, m_terminals.size())) {
         for(const auto& [iid, aid] : m_terminals.at(z).signatures.at(slot)) {
            covers.at(iid).at(aid).push_back(uint32_t(z));
         }
      }
   }

   ////////////////////////////
   /// reduced-plan space   ///
   ////////////////////////////

   /**
    * Enumerates the REDUCED normal-form plan space of one player: a single DFS over
    * the game tree that branches over the player's legal actions upon the FIRST visit
    * of an infoset within the current descent and follows the fixed assignment on
    * every later visit (infoset consistency). Each completed descent yields one raw
    * assignment, which is canonicalized to its self-reachable infoset domain before
    * insertion into the deduplicating result set -- realization-equivalent completions
    * collapse into one reduced plan. The traversal cost is a small multiple of the
    * game-tree size (each edge is walked once per distinct assignment of the owning
    * player's infosets preceding it), which keeps even wide games such as goofspiel
    * inexpensive.
    */
   std::vector< Plan > enumerate_reduced_plans_of(size_t slot)
   {
      const Player player = m_players.at(slot);
      const size_t n_infosets = m_structures.at(slot).size();

      std::unordered_set< Plan, plan_hasher, common::value_comparator< Plan > > unique_plans;

      Plan assignment(n_infosets, unassigned_plan_entry);
      std::vector< char > decided(n_infosets, 0);

      std::function< void(uptr< world_state_type >, TraversalContext&) > descend =
         [&](uptr< world_state_type > state, TraversalContext& ctx) {
            if(m_env.is_terminal(*state)) {
               unique_plans.insert(canonicalize_plan(slot, assignment));
               return;
            }
            const Player actor = m_env.active_player(*state);
            if(actor != player) {
               for(const auto& variant : child_actions(*state)) {
                  auto next = child_state(*state, variant);
                  TraversalContext child_ctx = ctx.deep_copy();
                  advance_variant(child_ctx, *state, variant, *next);
                  descend(std::move(next), child_ctx);
               }
               return;
            }
            const uint32_t current_iid = intern(
               actor, *ctx.infostates.at(actor), m_env.actions(actor, *state)
            );
            const auto& actions = m_structures.at(slot).actions.at(current_iid);
            if(decided.at(current_iid)) {
               // follow the already-fixed decision of this descent (the context must be
               // copied: mutating 'ctx' in place would pollute pending sibling branches
               // of the ancestor frames)
               const auto& chosen = actions.at(assignment.at(current_iid));
               auto next = child_state(*state, action_variant_type{chosen});
               TraversalContext child_ctx = ctx.deep_copy();
               advance_variant(child_ctx, *state, action_variant_type{chosen}, *next);
               descend(std::move(next), child_ctx);
               return;
            }
            decided.at(current_iid) = 1;
            for(auto a : std::views::iota(size_t{0}, actions.size())) {
               assignment.at(current_iid) = uint32_t(a);
               auto next = child_state(*state, action_variant_type{actions.at(a)});
               TraversalContext child_ctx = ctx.deep_copy();
               advance_variant(child_ctx, *state, action_variant_type{actions.at(a)}, *next);
               descend(std::move(next), child_ctx);
            }
            decided.at(current_iid) = 0;
            assignment.at(current_iid) = unassigned_plan_entry;
         };

      TraversalContext ctx = TraversalContext::initial(m_env, *m_root_state);
      descend(
         utils::static_unique_ptr_downcast< world_state_type >(utils::clone_any_way(*m_root_state)),
         ctx
      );
      return std::vector< Plan >{unique_plans.begin(), unique_plans.end()};
   }

   /**
    * Restricts 'plan' to the infosets REACHABLE under the plan itself: simulates every
    * root-to-terminal path that follows the plan at the owning player's assigned
    * decisions (branching everywhere else, including at unassigned infosets) and marks
    * each visited infoset with the plan's choice. Realization-equivalent assignments
    * produce identical canonical plans.
    */
   [[nodiscard]] Plan canonicalize_plan(size_t slot, const Plan& plan)
   {
      const Player player = m_players.at(slot);
      Plan canonical(m_structures.at(slot).size(), unassigned_plan_entry);

      std::function< void(uptr< world_state_type >, TraversalContext&) > descend =
         [&](uptr< world_state_type > state, TraversalContext& ctx) {
            if(m_env.is_terminal(*state)) {
               return;
            }
            Player actor = m_env.active_player(*state);
            if(actor == player) {
               uint32_t current_iid = intern(
                  actor, *ctx.infostates.at(actor), m_env.actions(actor, *state)
               );
               canonical.at(current_iid) = plan.at(current_iid);
               if(plan.at(current_iid) != unassigned_plan_entry) {
                  const auto& chosen = m_structures.at(slot)
                                          .actions.at(current_iid)
                                          .at(plan.at(current_iid));
                  auto next = child_state(*state, action_variant_type{chosen});
                  TraversalContext child_ctx = ctx.deep_copy();
                  advance_variant(child_ctx, *state, action_variant_type{chosen}, *next);
                  descend(std::move(next), child_ctx);
               } else {
                  for(const auto& variant : child_actions(*state)) {
                     auto next = child_state(*state, variant);
                     TraversalContext child_ctx = ctx.deep_copy();
                     advance_variant(child_ctx, *state, variant, *next);
                     descend(std::move(next), child_ctx);
                  }
               }
            } else {
               for(const auto& variant : child_actions(*state)) {
                  auto next = child_state(*state, variant);
                  TraversalContext child_ctx = ctx.deep_copy();
                  advance_variant(child_ctx, *state, variant, *next);
                  descend(std::move(next), child_ctx);
               }
            }
         };

      TraversalContext ctx = TraversalContext::initial(m_env, *m_root_state);
      descend(
         utils::static_unique_ptr_downcast< world_state_type >(utils::clone_any_way(*m_root_state)),
         ctx
      );
      return canonical;
   }

   ////////////////////////////
   /// members              ///
   ////////////////////////////

   Env m_env;
   uptr< world_state_type > m_root_state;
   std::vector< Player > m_players{};
   player_hashmap< size_t > m_player_slot{};
   std::vector< PlayerStructure > m_structures{};
   std::vector< std::unordered_map<
      info_state_type,
      uint32_t,
      common::value_hasher< info_state_type >,
      common::value_comparator< info_state_type > > >
      m_infostate_ids{};
   std::vector< TerminalRecord > m_terminals{};
   /// per player, per infoset, per local action: indices of terminals whose signature
   /// realizes that decision (witness lists of the Algorithm-2 argmax recursion)
   std::vector< std::vector< std::vector< std::vector< uint32_t > > > > m_cover{};
   std::vector< std::vector< Plan > > m_reduced_plans{};
   mutable std::vector< std::unordered_map<
      Plan,
      std::vector< uint64_t >,
      plan_hasher,
      common::value_comparator< Plan > > >
      m_plan_masks{};
};

}  // namespace nor::rm::correlated

#endif  // NOR_RM_CORRELATED_SEQUENCE_FORM_HPP
