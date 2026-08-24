
#ifndef NOR_COLONEL_BLOTTO_ENVIRONMENT_HPP
#define NOR_COLONEL_BLOTTO_ENVIRONMENT_HPP

#include <optional>
#include <utility>
#include <vector>

#include "colonel_blotto/state.hpp"
#include "colonel_blotto/utils.hpp"
#include "common/common.hpp"
#include "nor/fosg_states.hpp"
#include "nor/fosg_traits.hpp"
#include "nor/game_defs.hpp"
#include "nor/utils/player_informed_type.hpp"

namespace nor::games::colonel_blotto {

using namespace ::colonel_blotto;

inline auto to_blotto_player(const nor::Player& player)
{
   return static_cast< ::colonel_blotto::Player >(player);
}
inline auto to_nor_player(const ::colonel_blotto::Player& player)
{
   return static_cast< nor::Player >(player);
}

/**
 * @brief Compact observation of discretized Colonel Blotto.
 *
 * Information structure (per-field secret commitments, single terminal reveal):
 * - commit phases: the fact that someone committed on the current battlefield is public, the
 *   deployment size stays hidden until the fused all-field resolve;
 * - the committer privately echoes his fresh deployment;
 * - at the resolve-all transition ALL allocation vectors become public together with the
 *   per-battlefield outcomes and the terminal announcement.
 */
struct Observation {
   /// a commitment happened by this player; its value stays hidden (commit phases)
   std::optional< nor::Player > committed_by{};
   /// the battlefield the commitment event referred to
   std::optional< size_t > field{};
   /// the observer's own freshly committed deployment (private observation of the committer only)
   std::optional< Deploy > own_deploy{};
   /// both full allocation vectors in (one, two) order (public, resolve transition only)
   std::optional< std::pair<
      std::array< uint32_t, battlefield_count >,
      std::array< uint32_t, battlefield_count > > >
      revealed_allocations{};
   /// per-battlefield outcomes in field order (resolve transition only)
   std::optional< std::array< FieldOutcome, battlefield_count > > outcomes{};
   /// public announcement of how the game ended (terminal transitions only)
   std::optional< TerminalCause > terminal_cause{};

   friend bool operator==(const Observation&, const Observation&) = default;
};

}  // namespace nor::games::colonel_blotto

/// has to be visible before the DefaultPublicstate/DefaultInfostate bases below, whose
/// 'observation' concept requirement probes the availability of this specialization
namespace std {

template <>
struct hash< nor::games::colonel_blotto::Observation > {
   size_t operator()(const nor::games::colonel_blotto::Observation& obs) const noexcept
   {
      namespace cb = nor::games::colonel_blotto;
      size_t seed = 0;
      if(obs.committed_by.has_value()) {
         common::hash_combine(seed, std::hash< int >{}(1 + int(*obs.committed_by)));
      }
      if(obs.field.has_value()) {
         common::hash_combine(seed, std::hash< unsigned >{}(7u + unsigned(*obs.field)));
      }
      if(obs.own_deploy.has_value()) {
         common::hash_combine(
            seed, std::hash< unsigned >{}(17u + unsigned(obs.own_deploy->troops))
         );
      }
      if(obs.revealed_allocations.has_value()) {
         const auto& allocs = *obs.revealed_allocations;
         unsigned packed = 0;
         for(size_t j = 0; j < cb::battlefield_count; ++j) {
            packed = packed * 8u + unsigned(allocs.first.at(j));
         }
         for(size_t j = 0; j < cb::battlefield_count; ++j) {
            packed = packed * 8u + unsigned(allocs.second.at(j));
         }
         common::hash_combine(seed, std::hash< unsigned >{}(29u + packed));
      }
      if(obs.outcomes.has_value()) {
         unsigned packed = 0;
         for(size_t j = 0; j < cb::battlefield_count; ++j) {
            packed = packed * 4u + unsigned((*obs.outcomes).at(j));
         }
         common::hash_combine(seed, std::hash< unsigned >{}(97u + packed));
      }
      if(obs.terminal_cause.has_value()) {
         common::hash_combine(seed, std::hash< unsigned >{}(211u + unsigned(*obs.terminal_cause)));
      }
      return seed;
   }
};

}  // namespace std

namespace nor::games::colonel_blotto {

class Publicstate: public DefaultPublicstate< Publicstate, Observation > {
   using base = DefaultPublicstate< Publicstate, Observation >;
   using base::base;

   friend base;

   size_t _hash_impl() const
   {
      size_t seed = 0;
      for(const auto& observation : history()) {
         common::hash_combine(seed, std::hash< Observation >{}(observation));
      }
      return seed;
   }
};

class Infostate: public DefaultInfostate< Infostate, Observation > {
   using base = DefaultInfostate< Infostate, Observation >;
   using base::base;

   friend base;

   size_t _hash_impl() const
   {
      size_t seed = 0;
      for(const auto& [public_obs, private_obs] : history()) {
         common::hash_combine(seed, std::hash< Observation >{}(public_obs));
         common::hash_combine(seed, std::hash< Observation >{}(private_obs));
      }
      return seed;
   }
};

/**
 * @brief The FOSG environment adapter of discretized Colonel Blotto.
 *
 * The simultaneous vector allocation is sequentialized per battlefield as CommitP1 ->
 * CommitP2 (secret values) with the deterministic all-field resolution fused into the final
 * CommitP2 application (see State's Phase documentation).
 *
 * REWARD MODEL: CONSTANT-SUM -- reward(player) = fraction_won(player) - 1/2 sums to zero across
 * players at every terminal history. The normalized metric exploitability(...,
 * constant_sum=true) is therefore well-defined here (unlike Shapley/centipede), with nash_conv
 * coinciding since the centered rewards already sum to zero.
 */
class Environment {
  public:
   // nor fosg typedefs
   using world_state_type = State;
   using info_state_type = Infostate;
   using public_state_type = Publicstate;
   using action_type = Deploy;
   using chance_outcome_type = std::monostate;
   using observation_type = Observation;
   using action_variant_type = action_variant_type_generator_t< action_type, chance_outcome_type >;
   // nor fosg traits
   static constexpr size_t max_player_count() { return 2; }
   static constexpr size_t player_count() { return 2; }
   static constexpr bool serialized() { return true; }
   static constexpr bool unrolled() { return true; }
   static constexpr Stochasticity stochasticity() { return Stochasticity::deterministic; }

   Environment() = default;
   explicit Environment(BlottoConfig config) : m_config(config) {}

   [[nodiscard]] const BlottoConfig& config() const { return m_config; }

   ///////////////////////////////////
   /// API: transitions            ///
   ///////////////////////////////////

   [[nodiscard]] std::vector< action_type > actions(Player player, const world_state_type& wstate)
      const
   {
      return wstate.actions(to_blotto_player(player));
   }

   void transition(world_state_type& worldstate, const action_type& action) const
   {
      worldstate.apply_action(action);
   }

   void transition(world_state_type& worldstate, action_type&& action) const
   {
      worldstate.apply_action(std::move(action));
   }

   [[nodiscard]] world_state_type initial_world_state() const { return world_state_type(m_config); }

   /////////////////////////////////
   /// API: players and payoffs  ///
   /////////////////////////////////

   /**
    * NOTE: returns the Player::none sentinel (mapped to nor::Player::unknown) on terminal states.
    * Transitions INTO terminal states still query `active_player(next_wstate)` during the
    * observation-buffer flush, so this must never throw or crash there.
    */
   [[nodiscard]] Player active_player(const world_state_type& wstate) const
   {
      return to_nor_player(wstate.active_player());
   }

   static std::vector< Player > players(const world_state_type&)
   {
      return {Player::alex, Player::bob};
   }

   static bool is_terminal(const world_state_type& wstate) { return wstate.terminal(); }

   static constexpr bool is_partaking(const world_state_type&, Player) { return true; }

   /// alex is player one, bob is player two; each receives his centered value fraction
   static double reward(Player player, const world_state_type& wstate)
   {
      return wstate.payoff(to_blotto_player(player));
   }

   ////////////////////////////////
   /// API: observations        ///
   ////////////////////////////////

   observation_type private_observation(
      Player observer,
      const world_state_type& wstate,
      const action_type& action,
      const world_state_type& /*next_wstate*/
   ) const
   {
      // only the committer echoes his own fresh deployment
      if(to_blotto_player(observer) != wstate.active_player()) {
         return observation_type{};
      }
      return observation_type{.field = wstate.field(), .own_deploy = action};
   }

   observation_type public_observation(
      const world_state_type& wstate,
      const action_type& /*action*/,
      const world_state_type& next_wstate
   ) const
   {
      switch(wstate.phase()) {
         case Phase::commit_p1: {
            // commitment event is public knowledge while the value is withheld until the resolve
            return observation_type{.committed_by = Player::alex, .field = wstate.field()};
         }
         case Phase::commit_p2: {
            observation_type obs{.committed_by = Player::bob, .field = wstate.field()};
            if(next_wstate.terminal()) {
               // fused resolve-all publishes both allocations and every battlefield outcome
               obs.revealed_allocations = next_wstate.allocations();
               obs.outcomes = next_wstate.field_outcomes();
               obs.terminal_cause = next_wstate.terminal_cause();
            }
            return obs;
         }
      }
      return observation_type{};
   }

   observation_type tiny_repr(const world_state_type&) const { return observation_type{}; }

   ////////////////////////////////
   /// API: histories           ///
   ////////////////////////////////

   /**
    * chronological sequence of commitments masked to what `player` can observe: his/her own
    * deployments in full, the opponent's hidden (nullopt). Two entries per battlefield
    * (CommitP1, CommitP2); the last entry of a terminal state belongs to the fused resolve-all.
    */
   [[nodiscard]] std::vector< PlayerInformedType< std::optional< action_variant_type > > >
   private_history(Player player, const world_state_type& wstate) const
   {
      const auto self = to_blotto_player(player);
      std::vector< PlayerInformedType< std::optional< action_variant_type > > > out;
      const size_t n_field_pairs = wstate.terminal() or wstate.phase() == Phase::commit_p2
                                      ? wstate.field() + 1
                                      : wstate.field();
      out.reserve(2 * n_field_pairs);

      const auto& records = wstate.allocations();
      for(size_t j = 0; j < n_field_pairs; ++j) {
         out.emplace_back(
            self == ::colonel_blotto::Player::one
                  and _committed_on(wstate, ::colonel_blotto::Player::one, j)
               ? std::optional< action_variant_type >{action_variant_type{
                  Deploy{records.first.at(j)}}}
               : std::nullopt,
            Player::alex
         );
         out.emplace_back(
            self == ::colonel_blotto::Player::two
                  and _committed_on(wstate, ::colonel_blotto::Player::two, j)
               ? std::optional< action_variant_type >{action_variant_type{
                  Deploy{records.second.at(j)}}}
               : std::nullopt,
            Player::bob
         );
      }
      out.shrink_to_fit();
      return out;
   }

   /// chronological sequence with every commitment masked out (the public view)
   [[nodiscard]] std::vector< PlayerInformedType< std::optional< action_variant_type > > >
   public_history(const world_state_type& wstate) const
   {
      std::vector< PlayerInformedType< std::optional< action_variant_type > > > out;
      const size_t n_field_pairs = wstate.terminal() or wstate.phase() == Phase::commit_p2
                                      ? wstate.field() + 1
                                      : wstate.field();
      out.reserve(2 * n_field_pairs);

      for(size_t j = 0; j < n_field_pairs; ++j) {
         out.emplace_back(std::nullopt, Player::alex);
         out.emplace_back(std::nullopt, Player::bob);
      }
      out.shrink_to_fit();
      return out;
   }

   /// the fully open history in which even the hidden commitments are revealed
   [[nodiscard]] std::vector< PlayerInformedType< action_variant_type > > open_history(
      const world_state_type& wstate
   ) const
   {
      std::vector< PlayerInformedType< action_variant_type > > out;
      const size_t n_field_pairs = wstate.terminal() or wstate.phase() == Phase::commit_p2
                                      ? wstate.field() + 1
                                      : wstate.field();
      out.reserve(2 * n_field_pairs);

      const auto& records = wstate.allocations();
      for(size_t j = 0; j < n_field_pairs; ++j) {
         out.emplace_back(action_variant_type{Deploy{records.first.at(j)}}, Player::alex);
         out.emplace_back(action_variant_type{Deploy{records.second.at(j)}}, Player::bob);
      }
      out.shrink_to_fit();
      return out;
   }

  private:
   /// whether `mover` has already committed on battlefield `j` in this world state
   static bool
   _committed_on(const world_state_type& wstate, ::colonel_blotto::Player mover, size_t j)
   {
      if(j < wstate.field() or wstate.terminal()) {
         return true;  // earlier fields are fully committed by both sides (and so is a resolved
                       // one)
      }
      // current field: in commit_p1 nobody has committed yet, in commit_p2 only player one has
      return wstate.phase() == Phase::commit_p2 and mover == ::colonel_blotto::Player::one;
   }

   BlottoConfig m_config{};
};

}  // namespace nor::games::colonel_blotto

namespace common {

template <>
inline std::string to_string(const nor::games::colonel_blotto::Observation& value)
{
   std::string out;
   if(value.committed_by.has_value()) {
      out += fmt::format("committed:{},", common::to_string(*value.committed_by));
   }
   if(value.field.has_value()) {
      out += fmt::format("field:{},", *value.field);
   }
   if(value.own_deploy.has_value()) {
      out += fmt::format("own_deploy:{},", value.own_deploy->troops);
   }
   if(value.revealed_allocations.has_value()) {
      out += fmt::format(
         "allocs:({},{},{}|{},{},{}),",
         value.revealed_allocations->first.at(0),
         value.revealed_allocations->first.at(1),
         value.revealed_allocations->first.at(2),
         value.revealed_allocations->second.at(0),
         value.revealed_allocations->second.at(1),
         value.revealed_allocations->second.at(2)
      );
   }
   if(value.outcomes.has_value()) {
      out += "outcomes:(";
      for(size_t j = 0; j < ::colonel_blotto::battlefield_count; ++j) {
         out += common::to_string(value.outcomes->at(j)) + ",";
      }
      out.back() = ')';
      out += ',';
   }
   if(value.terminal_cause.has_value()) {
      out += fmt::format("terminal:{},", common::to_string(*value.terminal_cause));
   }
   if(out.empty()) {
      return "-";
   }
   out.pop_back();
   return out;
}

}  // namespace common

COMMON_ENABLE_PRINT(nor::games::colonel_blotto, Observation);

namespace nor {

template <>
struct fosg_traits< games::colonel_blotto::Infostate > {
   using observation_type = nor::games::colonel_blotto::Observation;
};

template <>
struct fosg_traits< games::colonel_blotto::Environment > {
   using world_state_type = nor::games::colonel_blotto::State;
   using info_state_type = nor::games::colonel_blotto::Infostate;
   using public_state_type = nor::games::colonel_blotto::Publicstate;
   using action_type = nor::games::colonel_blotto::Deploy;
   using chance_outcome_type = std::monostate;
   using observation_type = nor::games::colonel_blotto::Observation;
};

}  // namespace nor

namespace std {

template < typename StateType >
   requires common::is_any_v<
      StateType,
      nor::games::colonel_blotto::Publicstate,
      nor::games::colonel_blotto::Infostate >
struct hash< StateType > {
   size_t operator()(const StateType& state) const noexcept { return state.hash(); }
};

}  // namespace std

#endif  // NOR_COLONEL_BLOTTO_ENVIRONMENT_HPP
