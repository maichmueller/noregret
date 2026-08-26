
#ifndef NOR_HINDSIGHT_HPP
#define NOR_HINDSIGHT_HPP

#include <algorithm>
#include <array>
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
#include "nor/policy/action_policy.hpp"
#include "nor/rm/rm_utils.hpp"
#include "nor/utils/utils.hpp"

/**
 * @file hindsight.hpp
 *
 * Hindsight-rationality / correlated-play evaluation module (Morrill, D'Orazio,
 * Sarfati, Lanctot, Wright, Greenwald, Bowling, "Hindsight and Sequential
 * Rationality of Correlated Play", AAAI 2021, arXiv:2012.05874).
 *
 * Given an EMPIRICAL CORRELATION over joint plays -- represented as a weighted
 * collection of joint behavioral profiles ("snapshots"), e.g. the iterates of
 * any solver -- the module computes, for every player, the maximum deviation
 * gain under the phi-deviation taxonomy:
 *
 *   - external                (phi^{ex};   equilibrium concept CCE)
 *   - blind causal            (phi^{bc};   EFCCE)
 *   - action                  (blind action; AFCCE)
 *   - blind counterfactual    (phi^{bcf};  CFCCE)
 *   - informed counterfactual (phi^{icf};  CFCE)
 *
 * Gains are non-negative scalars: zero means the empirical play has no
 * incentive to deviate within that family ("gap-free"), larger values mean
 * stronger incentives.
 *
 * Provable numeric relations (verified in test/libnor/test_hindsight.cpp):
 *   - every gap is non-negative;
 *   - on a single recommendation draw, gap(blind CF) >= gap(informed CF): the
 *     informed gain equals the blind one deflated by the
 *     recommendation-matching trigger weight sigma_pi(I, a) <= 1;
 *   - on single-decision games all non-informed families coincide exactly;
 *   - causal-vs-action and causal-vs-counterfactual families are provably
 *     INCOMPARABLE in general (paper Table 1; Sections 5.1/6.3 examples), so no
 *     ordering among them is asserted anywhere.
 *
 * ENGINE GENERALIZATIONS (documented deviations from the paper's pure-strategy
 * mediated-equilibrium setting):
 *   - Snapshots are BEHAVIORAL profiles rather than distributions over pure
 *     strategy profiles; trigger conditions the paper states as a realized
 *     recommendation s_i(I^!) = a^! generalize to TRIGGER WEIGHTS
 *     sigma_pi(I^!, a^!) (the probability that the recommendation matches).
 *   - Counterfactual gains use the paper's counterfactual weighting
 *       sum_{h in I} pi_-i(h) * [v(h a_odot) - sum_a sigma(I,a) v(h a)]
 *     (Def. 1/2 transformations evaluated through Theorem 2's intermediate-
 *     counterfactual-regret form), while external / blind-causal / action
 *     gains use plain expected-utility differences under mediated execution.
 *   - Deterministic replacement strategies (external / blind causal) are
 *     enumerated over the REALIZED infostate support of the recorded play;
 *     behavior outside that support falls back to the snapshot's own
 *     recommendation. A work-budget guard rejects supports whose enumeration
 *     would explode.
 */

namespace nor::hindsight {

/////////////////////////////////////////////////////////////////////////////////////////////
/////////////////////////////////////// public taxonomy /////////////////////////////////////
/////////////////////////////////////////////////////////////////////////////////////////////

/// the phi-deviation taxonomy of Morrill et al. (AAAI 2021)
enum class DeviationFamily : uint8_t {
   external = 0,  ///< constant deviations (CCE)
   blind_causal = 1,  ///< commit to a fixed strategy once triggered (EFCCE)
   action = 2,  ///< swap a single action at a trigger infoset (AFCCE)
   blind_counterfactual = 3,  ///< play to reach a target infoset, swap there (CFCCE)
   informed_counterfactual = 4  ///< ...but only if the recommendation matched (CFCE)
};

[[nodiscard]] inline constexpr size_t deviation_family_count()
{
   return 5;
}

[[nodiscard]] inline constexpr size_t family_index(DeviationFamily family)
{
   return static_cast< size_t >(family);
}

/////////////////////////////////////////////////////////////////////////////////////////////
/////////////////////////// empirical correlation representation ////////////////////////////
/////////////////////////////////////////////////////////////////////////////////////////////

/// the behavioral profile table shape used throughout this module: a map from
/// each player's infostates to (normalized) action policies
template < typename Env >
using hindsight_table = std::
   unordered_map< auto_info_state_type< Env >, HashmapActionPolicy< auto_action_type< Env > > >;

/**
 * @brief empirical correlation over joint plays: a weighted collection of
 * joint behavioral profiles ("snapshots").
 *
 * This is the recording accumulator of the module: alongside ANY solver's
 * iteration loop, call record_average()/record_current() (or add() with
 * hand-built tables) once per iterate to build the empirical distribution of
 * play mu^T (Morrill et al., sec. 3.3). Tables are expected to hold NORMALIZED
 * action probabilities; the recorder adapters normalize automatically via
 * nor::normalize_state_policy.
 */
template < typename Env >
class EmpiricalPlay {
  public:
   using env_type = std::remove_cvref_t< Env >;
   using info_state_type = auto_info_state_type< env_type >;
   using action_type = auto_action_type< env_type >;
   using table_type = hindsight_table< env_type >;

   struct Snapshot {
      double weight = 1.;
      player_hashmap< table_type > profile{};
   };

   /// add a hand-built joint behavioral profile with the given mixture weight
   void add(double weight, player_hashmap< table_type > profile)
   {
      m_snapshots.push_back(Snapshot{weight, std::move(profile)});
   }

   /**
    * @brief record one iterate of a solver exposing 'average_policy()' or
    * 'policy()' (a player_hashmap of tabular policies with '.table()' access).
    * The tables are snapshotted NORMALIZED.
    */
   template < bool average, typename Solver >
   void record(const Solver& solver, double weight = 1.)
   {
      player_hashmap< table_type > profile{};
      const auto& policy_map = [&]() -> decltype(auto) {
         if constexpr(average) {
            return solver.average_policy();
         } else {
            return solver.policy();
         }
      }();
      for(const auto& [player, table] : policy_map) {
         if(player == Player::chance) {
            continue;
         }
         profile.emplace(player, normalize_state_policy(table.table()));
      }
      add(weight, std::move(profile));
   }

   template < typename Solver >
   void record_average(const Solver& solver, double weight = 1.)
   {
      record< true >(solver, weight);
   }

   template < typename Solver >
   void record_current(const Solver& solver, double weight = 1.)
   {
      record< false >(solver, weight);
   }

   [[nodiscard]] const std::vector< Snapshot >& snapshots() const { return m_snapshots; }

   [[nodiscard]] double total_weight() const
   {
      return std::ranges::fold_left(
         m_snapshots | std::views::transform([](const Snapshot& s) { return s.weight; }),
         0.,
         std::plus{}
      );
   }

   [[nodiscard]] bool empty() const { return m_snapshots.empty(); }

  private:
   std::vector< Snapshot > m_snapshots{};
};

/////////////////////////////////////////////////////////////////////////////////////////////
//////////////////////////////////// evaluation internals ///////////////////////////////////
/////////////////////////////////////////////////////////////////////////////////////////////

namespace detail {

/// upper bound on plain expectimax evaluations per gap computation (guards
/// against exponential enumeration over large realized supports)
inline constexpr size_t kMaxProfileEvaluations = 200000;

template < typename InfoState >
[[nodiscard]] inline bool is_predecessor_or_equal(const InfoState& pre, const InfoState& cur)
{
   // perfect-recall predecessor test between two of the SAME player's
   // infostates: 'pre' precedes 'cur' iff its observation history is a prefix
   // of cur's
   const auto& pre_history = pre.history();
   const auto& cur_history = cur.history();
   return pre_history.size() <= cur_history.size()
          and std::equal(pre_history.begin(), pre_history.end(), cur_history.begin());
}

/// hash of a (target infoset, target action) deviation candidate
template < typename Env >
struct CandidateHash {
   using info_state_type = auto_info_state_type< Env >;
   using action_type = auto_action_type< Env >;

   size_t operator()(const std::pair< info_state_type, action_type >& candidate) const noexcept
   {
      size_t seed = 0;
      common::hash_combine(seed, common::value_hasher< info_state_type >{}(candidate.first));
      common::hash_combine(seed, common::value_hasher< action_type >{}(candidate.second));
      return seed;
   }
};

/// counterfactual bookkeeping accumulated over the visits to ONE infoset of
/// the target player within ONE snapshot
template < typename Env >
struct CounterfactualEntry {
   /// accumulated chance x opponents' reach pi_-i(h) of the visited histories
   double reach_minus_player = 0.;
   /// plain continuation values V(h a), pushed once per visit and aligned
   /// across actions; Morrill et al.'s counterfactual value sums these
   /// weighted by 'reach_minus_player'
   std::unordered_map< auto_action_type< Env >, std::vector< double > > action_values{};
};

/**
 * @brief single-pass root-to-leaves walker behind ALL evaluations.
 *
 * One recursive descent folds observation buffers / infostates exactly like
 * rm::policy_value, so behavioral tables are indexed by the CURRENT infostate
 * identity at every decision frame. While descending it optionally RECORDS --
 * at every infoset of the target player -- the counterfactual reach pi_-i(h)
 * and the per-action continuation values V(h a); and it evaluates the target
 * player's profile value under an OPTIONAL deterministic deviation overlay
 * (an assignment of forced actions restricted to descendants-or-self of an
 * optional trigger infoset).
 */
template < typename Env, typename Table >
class ProfileWalker {
  public:
   using info_state_type = auto_info_state_type< Env >;
   using action_type = auto_action_type< Env >;
   using observation_type = auto_observation_type< Env >;
   using EntryMap = std::unordered_map< info_state_type, CounterfactualEntry< Env > >;
   using Obsbuffer = player_hashmap<
      std::vector< std::pair< observation_type, observation_type > > >;
   using IstateMap = player_hashmap< sptr< info_state_type > >;
   using Assignment = std::unordered_map<
      info_state_type,
      action_type,
      common::value_hasher< info_state_type >,
      common::value_comparator< info_state_type > >;

   /// chain of (own infostate -> action) decisions along the current DFS path
   using OwnChain = std::vector< std::pair< info_state_type, action_type > >;
   /// collected play-to-reach chains: target infostate -> its ancestor chain
   using ChainMap = std::unordered_map<
      info_state_type,
      OwnChain,
      common::value_hasher< info_state_type >,
      common::value_comparator< info_state_type > >;

   struct Config {
      Player target_player{};
      const player_hashmap< Table >* profile = nullptr;
      bool record_entries = false;
      const Assignment* assignment = nullptr;
      const info_state_type* trigger = nullptr;
      ChainMap* chains = nullptr;
   };

   struct Result {
      double value = 0.;
      EntryMap entries{};
   };

   [[nodiscard]] static Result
   run(const Env& env, const auto_world_state_type< Env >& root, const Config& config)
   {
      Obsbuffer buffers{};
      IstateMap istates{};
      for(auto player : env.players(root)) {
         if(player == Player::chance) {
            continue;
         }
         buffers.try_emplace(player);
         istates.emplace(player, std::make_shared< info_state_type >(player));
      }
      Result result{};
      result.value = walk(
         env,
         utils::clone_any_way(root),
         /*reach_minus_player=*/1.,
         config,
         buffers,
         istates,
         result.entries
      );
      return result;
   }

  private:
   static double walk(
      const Env& env,
      uptr< auto_world_state_type< Env > > state,
      double reach_minus_player,
      const Config& config,
      Obsbuffer& buffers,
      IstateMap& istates,
      EntryMap& entries,
      const OwnChain& current_chain = {}
   )
   {
      if(env.is_terminal(*state)) {
         return env.reward(config.target_player, *state);
      }
      const Player active = env.active_player(*state);

      if constexpr(concepts::stochastic_env< Env >) {
         if(active == Player::chance) {
            double acc = 0.;
            for(const auto& outcome : env.chance_actions(*state)) {
               const double prob = env.chance_probability(*state, outcome);
               auto next = child_state(env, *state, outcome);
               auto [child_buffers, child_istates] = next_infostate_and_obs_buffers(
                  env, buffers, istates, *state, outcome, *next
               );
               acc += prob
                      * walk(
                         env,
                         std::move(next),
                         reach_minus_player * prob,
                         config,
                         child_buffers,
                         child_istates,
                         entries,
                         current_chain
                      );
            }
            return acc;
         }
      }

      const auto actions = env.actions(active, *state);

      if(active == config.target_player) {
         // this frame's infostate was advanced into place by the parent edge fold
         const info_state_type& infostate = *istates.at(active);
         const bool overlay_applies =
            config.assignment != nullptr
            and (config.trigger == nullptr || is_predecessor_or_equal(*config.trigger, infostate))
            and config.assignment->contains(infostate);
         CounterfactualEntry< Env >* entry = nullptr;
         if(config.record_entries) {
            entry = &entries[infostate];
            entry->reach_minus_player += reach_minus_player;
         }
         if(config.chains != nullptr and not config.chains->contains(infostate)) {
            // perfect recall makes the play-to-reach decision sequence of every
            // history in this infoset unique, so recording the first arrival's
            // ancestor chain suffices
            (*config.chains)[infostate] = current_chain;
         }
         OwnChain next_chain = current_chain;
         next_chain.emplace_back(infostate, actions.front());
         double node_value = 0.;
         for(const auto& action : actions) {
            double prob = std::numeric_limits< double >::quiet_NaN();
            if(overlay_applies) {
               prob = action == config.assignment->at(infostate) ? 1. : 0.;
            } else {
               prob = base_action_probability(config.profile, active, &infostate, action, actions);
            }
            auto next = child_state(env, *state, action);
            auto [child_buffers, child_istates] = next_infostate_and_obs_buffers(
               env, buffers, istates, *state, action, *next
            );
            next_chain.back().second = action;
            const double continuation = walk(
               env,
               std::move(next),
               reach_minus_player,
               config,
               child_buffers,
               child_istates,
               entries,
               next_chain
            );
            if(entry != nullptr) {
               entry->action_values[action].push_back(continuation);
            }
            node_value += prob * continuation;
         }
         return node_value;
      }

      // opponent frame: scale the counterfactual reach by the opponent's
      // snapshot action probability and continue
      const info_state_type& opponent_infostate = *istates.at(active);
      double acc = 0.;
      for(const auto& action : actions) {
         const double prob = base_action_probability(
            config.profile, active, &opponent_infostate, action, actions
         );
         if(prob <= 0.) {
            continue;
         }
         auto next = child_state(env, *state, action);
         auto [child_buffers, child_istates] = next_infostate_and_obs_buffers(
            env, buffers, istates, *state, action, *next
         );
         acc += prob
                * walk(
                   env,
                   std::move(next),
                   reach_minus_player * prob,
                   config,
                   child_buffers,
                   child_istates,
                   entries,
                   current_chain
                );
      }
      return acc;
   }

  public:
   /// normalized probability of 'action' for 'player' in the snapshot profile;
   /// 'infostate' may be null (opponent frames of the plain value recursion),
   /// and unrealized infostates fall back to the uniform distribution over
   /// 'actions'
   [[nodiscard]] static double base_action_probability(
      const player_hashmap< Table >* profile,
      Player player,
      const info_state_type* infostate,
      const action_type& action,
      const std::vector< action_type >& actions
   )
   {
      constexpr double kUniform = 0.;  // replaced below; keeps single return shape
      (void) kUniform;
      const double fallback = 1. / static_cast< double >(actions.size());
      if(profile == nullptr or infostate == nullptr) {
         return fallback;
      }
      auto profile_iter = profile->find(player);
      if(profile_iter == profile->end()) {
         return fallback;
      }
      auto row_iter = profile_iter->second.find(*infostate);
      if(row_iter == profile_iter->second.end()) {
         return fallback;
      }
      return row_iter->second.at(action);
   }
};

/// the realized infostate support of 'player': union of the table keys over all
/// snapshots, together with the union of legal actions seen per infostate
template < typename Env >
struct SupportStructure {
   using info_state_type = auto_info_state_type< Env >;
   using action_type = auto_action_type< Env >;

   std::vector< info_state_type > infostates{};
   std::unordered_map<
      info_state_type,
      std::vector< action_type >,
      common::value_hasher< info_state_type >,
      common::value_comparator< info_state_type > >
      actions_of{};
};

template < typename Env >
[[nodiscard]] SupportStructure< Env > support_of(const EmpiricalPlay< Env >& play, Player player)
{
   using Support = SupportStructure< Env >;
   using info_state_type = typename Support::info_state_type;
   using action_type = typename Support::action_type;

   std::unordered_set<
      info_state_type,
      common::value_hasher< info_state_type >,
      common::value_comparator< info_state_type > >
      seen{};
   Support support{};
   for(const auto& snapshot : play.snapshots()) {
      auto profile_iter = snapshot.profile.find(player);
      if(profile_iter == snapshot.profile.end()) {
         continue;
      }
      for(const auto& [infostate, action_policy] : profile_iter->second) {
         if(seen.insert(infostate).second) {
            support.infostates.push_back(infostate);
         }
         auto& actions = support.actions_of[infostate];
         for(const auto& [action, prob] : action_policy) {
            (void) prob;
            if(std::ranges::find(actions, action) == actions.end()) {
               actions.push_back(action);
            }
         }
      }
   }
   return support;
}

/// deterministic assignment of one legal action to selected support infostates
template < typename Env >
using DeterministicStrategy = typename ProfileWalker< Env, hindsight_table< Env > >::Assignment;

/// enumerate all deterministic strategies over 'infostates' (each choosing from
/// its legal-action list) and invoke 'sink' on each; returns false when the
/// enumeration exceeds 'budget' total invocations
template < typename Env, typename Sink >
[[nodiscard]] bool enumerate_deterministic_strategies(
   const std::vector< typename SupportStructure< Env >::info_state_type >& infostates,
   const SupportStructure< Env >& support,
   size_t& budget,
   Sink&& sink
)
{
   using info_state_type = typename SupportStructure< Env >::info_state_type;
   DeterministicStrategy< Env > assignment{};
   std::vector< size_t > cursor(infostates.size(), 0);

   auto advance = [&]() -> bool {
      for(size_t idx = 0; idx < cursor.size(); ++idx) {
         ++cursor[idx];
         if(cursor[idx] < support.actions_of.at(infostates[idx]).size()) {
            return true;
         }
         cursor[idx] = 0;
      }
      return false;
   };
   auto fill = [&] {
      for(const auto [idx, infostate] : std::views::enumerate(infostates)) {
         assignment[infostate] = support.actions_of.at(infostate).at(cursor[idx]);
      }
   };

   if(infostates.empty()) {
      // nothing to enumerate: a complete (not budget-exhausted) no-op
      return true;
   }
   do {
      if(budget == 0) {
         return false;
      }
      --budget;
      fill();
      std::invoke(sink, assignment);
   } while(advance());
   return true;
}

}  // namespace detail

/////////////////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////// public gap evaluators ////////////////////////////////
/////////////////////////////////////////////////////////////////////////////////////////////

/**
 * @brief maximum deviation gain of ONE deviation family for ONE player under
 * the empirical correlation 'play'.
 *
 * Deviation parameters are maximized against the WHOLE distribution (the
 * mediator semantics: one phi, benefit averaged over snapshots), not per
 * snapshot.
 *
 * @throws std::invalid_argument on an empty 'play'
 * @throws std::overflow_error when the deterministic-strategy enumeration of
 *         the realized support exceeds the internal work budget
 */
template < typename Env >
[[nodiscard]] double deviation_gap(
   DeviationFamily family,
   const Env& env,
   const auto_world_state_type< Env >& root_state,
   const EmpiricalPlay< Env >& play,
   Player player
)
{
   using detail::ProfileWalker;
   using info_state_type = auto_info_state_type< Env >;
   using action_type = auto_action_type< Env >;
   using table_type = hindsight_table< Env >;

   if(play.empty()) {
      throw std::invalid_argument("deviation_gap: the empirical play carries no snapshots.");
   }

   const double total_weight = play.total_weight();
   if(not (total_weight > 0.)) {
      throw std::invalid_argument("deviation_gap: the empirical play weights sum to zero.");
   }

   switch(family) {
      ///////////////////////////////////////////////////////////////////////
      case DeviationFamily::blind_counterfactual:
      case DeviationFamily::informed_counterfactual: {
         // Play-to-reach deviations evaluated through PLAIN mediated-execution
         // benefits (Morrill et al., Defs 3/4). For each candidate target
         // infoset I_odot the walker first collects the unique play-to-reach
         // decision chain (perfect recall); the deviated profile forces those
         // decisions plus the target action a_odot, and everything else
         // follows the recommendation (re-correlation). The informed variant
         // deflates the benefit by the recommendation-matching trigger weight
         // sigma_pi(I_odot, a_odot).
         using Walker = ProfileWalker< Env, table_type >;
         std::unordered_map<
            std::pair< info_state_type, action_type >,
            double,
            detail::CandidateHash< Env > >
            aggregated{};
         for(const auto& snapshot : play.snapshots()) {
            const double norm_weight = snapshot.weight / total_weight;
            const table_type empty_table{};
            const table_type& player_table = [&]() -> const table_type& {
               auto found = snapshot.profile.find(player);
               return found != snapshot.profile.end() ? found->second : empty_table;
            }();
            typename Walker::Config collect_config{
               .target_player = player, .profile = &snapshot.profile, .record_entries = true};
            typename Walker::ChainMap chains{};
            collect_config.chains = &chains;
            const double base_value = Walker::run(env, root_state, collect_config).value;

            for(const auto& [target, chain] : chains) {
               const auto row_iter = player_table.find(target);
               const HashmapActionPolicy< action_type >*
                  recommendation_row = row_iter == player_table.end() ? nullptr : &row_iter->second;
               std::vector< action_type > candidate_actions{};
               if(recommendation_row != nullptr) {
                  for(const auto& [a, p] : *recommendation_row) {
                     (void) p;
                     candidate_actions.push_back(a);
                  }
               } else {
                  for(const auto& a : detail::support_of(play, player).actions_of.at(target)) {
                     candidate_actions.push_back(a);
                  }
               }
               for(const auto& action : candidate_actions) {
                  detail::DeterministicStrategy< Env > assignment{};
                  for(const auto& [infostate, forced] : chain) {
                     assignment.emplace(infostate, forced);
                  }
                  // force the deviation action at the target itself (the last
                  // chain entry, or the ONLY entry when the target is a forest
                  // root of the player's infostate tree)
                  assignment[target] = action;
                  typename Walker::Config dev_cfg{
                     .target_player = player,
                     .profile = &snapshot.profile,
                     .record_entries = false,
                     .assignment = &assignment};
                  const double deviated = Walker::run(env, root_state, dev_cfg).value;
                  double gain = norm_weight * (deviated - base_value);
                  if(family == DeviationFamily::informed_counterfactual) {
                     const double trigger_weight = recommendation_row == nullptr
                                                      ? 1.
                                                           / static_cast< double >(
                                                              candidate_actions.size()
                                                           )
                                                      : recommendation_row->at(action);
                     gain *= trigger_weight;
                  }
                  aggregated[{target, action}] += gain;
               }
            }
         }
         double max_gain = 0.;
         for(const auto& [candidate, gain] : aggregated) {
            (void) candidate;
            max_gain = std::max(max_gain, gain);
         }
         return max_gain;
      }

      ///////////////////////////////////////////////////////////////////////
      case DeviationFamily::external:
      case DeviationFamily::blind_causal:
      case DeviationFamily::action: {
         // plain mediated-execution benefits of deterministic deviations over
         // the realized support
         const auto support = detail::support_of(play, player);

         size_t budget = detail::kMaxProfileEvaluations;
         double max_gain = 0.;
         bool exhausted = false;

         // per-snapshot base values, evaluated once
         struct BaseSnapshot {
            double norm_weight;
            const player_hashmap< table_type >* profile;
            double base_value;
         };
         std::vector< BaseSnapshot > bases{};
         bases.reserve(play.snapshots().size());
         for(const auto& snapshot : play.snapshots()) {
            typename ProfileWalker< Env, table_type >::Config config{
               .target_player = player, .profile = &snapshot.profile};
            bases.push_back(BaseSnapshot{
               snapshot.weight / total_weight,
               &snapshot.profile,
               ProfileWalker< Env, table_type >::run(env, root_state, config).value});
         }

         auto accumulate_benefit = [&](
                                      const detail::DeterministicStrategy< Env >& assignment,
                                      const std::optional< info_state_type >& trigger
                                   ) {
            double benefit = 0.;
            for(const auto& base : bases) {
               typename ProfileWalker< Env, table_type >::Config deviated_config{
                  .target_player = player,
                  .profile = base.profile,
                  .record_entries = false,
                  .assignment = &assignment,
                  .trigger = trigger.has_value() ? &trigger.value() : nullptr};
               benefit += base.norm_weight
                          * (ProfileWalker< Env, table_type >::run(env, root_state, deviated_config)
                                .value
                             - base.base_value);
            }
            max_gain = std::max(max_gain, benefit);
         };

         if(family == DeviationFamily::action) {
            // single-point swaps: (trigger, forced action) pairs only
            for(const auto& trigger : support.infostates) {
               for(const auto& action : support.actions_of.at(trigger)) {
                  if(budget == 0) {
                     exhausted = true;
                     break;
                  }
                  --budget;
                  detail::DeterministicStrategy< Env > assignment{{trigger, action}};
                  accumulate_benefit(assignment, trigger);
               }
            }
         } else if(family == DeviationFamily::blind_causal) {
            // trigger + fixed replacement restricted to the trigger's subtree
            for(const auto& trigger : support.infostates) {
               std::vector< info_state_type > descendants{};
               for(const auto& infostate : support.infostates) {
                  if(detail::is_predecessor_or_equal(trigger, infostate)) {
                     descendants.push_back(infostate);
                  }
               }
               const bool complete = detail::enumerate_deterministic_strategies< Env >(
                  descendants,
                  support,
                  budget,
                  [&](const detail::DeterministicStrategy< Env >& assignment) {
                     accumulate_benefit(assignment, trigger);
                  }
               );
               if(not complete) {
                  exhausted = true;
                  break;
               }
            }
         } else {
            // external: constant strategies over the whole realized support
            const bool complete = detail::enumerate_deterministic_strategies< Env >(
               support.infostates,
               support,
               budget,
               [&](const detail::DeterministicStrategy< Env >& assignment) {
                  accumulate_benefit(assignment, std::nullopt);
               }
            );
            if(not complete) {
               exhausted = true;
            }
         }
         if(exhausted) {
            throw std::overflow_error(
               "deviation_gap: deterministic-strategy enumeration over the realized "
               "support exceeded the work budget; restrict the recorded support or "
               "raise hindsight::detail::kMaxProfileEvaluations."
            );
         }
         return max_gain;
      }
   }
   throw std::invalid_argument("deviation_gap: unknown deviation family.");
}

/**
 * @brief all five taxonomy gaps for ONE player, indexed by family_index().
 */
template < typename Env >
[[nodiscard]] std::array< double, deviation_family_count() > deviation_gaps(
   const Env& env,
   const auto_world_state_type< Env >& root_state,
   const EmpiricalPlay< Env >& play,
   Player player
)
{
   std::array< double, deviation_family_count() > out{};
   for(const auto family :
       {DeviationFamily::external,
        DeviationFamily::blind_causal,
        DeviationFamily::action,
        DeviationFamily::blind_counterfactual,
        DeviationFamily::informed_counterfactual}) {
      out[family_index(family)] = deviation_gap(family, env, root_state, play, player);
   }
   return out;
}

/**
 * @brief hindsight-rationality report: all five taxonomy gaps for every actual
 * player of the game, indexed by family_index().
 */
template < typename Env >
[[nodiscard]] player_hashmap< std::array< double, deviation_family_count() > > hindsight_gaps(
   const Env& env,
   const auto_world_state_type< Env >& root_state,
   const EmpiricalPlay< Env >& play
)
{
   player_hashmap< std::array< double, deviation_family_count() > > report{};
   for(auto player : env.players(root_state)) {
      if(player == Player::chance) {
         continue;
      }
      report.emplace(player, deviation_gaps(env, root_state, play, player));
   }
   return report;
}

}  // namespace nor::hindsight

#endif  // NOR_HINDSIGHT_HPP
