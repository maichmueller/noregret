
#ifndef NOR_ENV_KUHN_HPP
#define NOR_ENV_KUHN_HPP

#include <fmt/format.h>

#include <ranges>
#include <string>
#include <vector>

#include "common/common.hpp"
#include "kuhn_poker/kuhn_poker.hpp"
#include "nor/fosg_states.hpp"
#include "nor/fosg_traits.hpp"
#include "nor/game_defs.hpp"

namespace nor::games::kuhn {

using namespace ::kuhn;

inline auto to_kuhn_player(const nor::Player& player)
{
   return static_cast< kuhn::Player >(player);
}
inline auto to_nor_player(const kuhn::Player& player)
{
   return static_cast< nor::Player >(player);
}

/**
 * @brief compact observation record of kuhn poker.
 *
 * Replaces the previous std::string encoding ("-", "check"/"bet", the dealt
 * card's name for private deals, "<seat>:?" for public deals) with a tagged
 * 4-byte struct that is cheap to copy, compare and hash. The kinds mirror the
 * information carried by the four observation producers of the environment:
 *    none          -- no information (legacy "-")
 *    action        -- a public check/bet action
 *    private_deal  -- a card deal; carries receiver seat AND card identity
 *                     (only ever produced for the receiving player)
 *    public_deal   -- a card deal; carries only the receiver's seat
 */
struct Observation {
   enum class Kind : int8_t { none = 0, action, private_deal, public_deal };

   Kind kind = Kind::none;
   /// receiving seat of a deal (private_deal/public_deal); meaningless otherwise
   ::kuhn::Player player = ::kuhn::Player::one;
   /// dealt card identity (private_deal only); meaningless otherwise
   ::kuhn::Card card = ::kuhn::Card::two;
   /// bet/check payload (action only); meaningless otherwise
   ::kuhn::Action action = ::kuhn::Action::check;

   friend bool operator==(const Observation&, const Observation&) = default;
};

}  // namespace nor::games::kuhn

namespace std {

/// NOTE: defined BEFORE the infostate/publicstate adapter classes below so that
/// their hash members can instantiate it immediately.
template <>
struct hash< nor::games::kuhn::Observation > {
   size_t operator()(const nor::games::kuhn::Observation& obs) const noexcept
   {
      size_t seed = 0;
      common::hash_combine(seed, obs.kind, obs.player, obs.card, obs.action);
      return seed;
   }
};

}  // namespace std

namespace nor::games::kuhn {

class Publicstate: public DefaultPublicstate< Publicstate, Observation > {
   using base = DefaultPublicstate< Publicstate, Observation >;
   using base::base;
};
class Infostate: public nor::DefaultInfostate< Infostate, Observation > {
   using base = DefaultInfostate< Infostate, Observation >;
   using base::base;
};

class Environment {
  public:
   // nor fosg typedefs
   using world_state_type = State;
   using info_state_type = Infostate;
   using public_state_type = Publicstate;
   using action_type = Action;
   using chance_outcome_type = ChanceOutcome;
   using observation_type = Observation;
   using action_variant_type = action_variant_type_generator_t< action_type, chance_outcome_type >;
   // nor fosg traits. The actual player count is configured per world state (2 by default);
   // 'dynamic_extent' marks it as a runtime property like in the leduc/texas hold'em envs.
   static constexpr size_t max_player_count() { return nor::games::kuhn::State::max_player_count; }
   static constexpr size_t player_count() { return std::dynamic_extent; }
   static constexpr bool serialized() { return true; }
   static constexpr bool unrolled() { return true; }
   static constexpr Stochasticity stochasticity() { return Stochasticity::choice; }

  public:
   Environment() = default;

   std::vector< action_type > actions(Player, const world_state_type& wstate) const
   {
      return wstate.actions();
   }
   inline std::vector< chance_outcome_type > chance_actions(const world_state_type& wstate) const
   {
      return wstate.chance_actions();
   }

   [[nodiscard]] std::vector< PlayerInformedType< std::optional< action_variant_type > > >
   private_history(Player player, const world_state_type& wstate) const;

   [[nodiscard]] std::vector< PlayerInformedType< std::optional< action_variant_type > > >
   public_history(const world_state_type& wstate) const;

   [[nodiscard]] std::vector< PlayerInformedType< action_variant_type > > open_history(
      const world_state_type& wstate
   ) const;

   inline double
   chance_probability(const world_state_type& wstate, const chance_outcome_type& outcome) const
   {
      return wstate.chance_probability(outcome);
   }

   static inline std::vector< Player > players(const world_state_type& wstate)
   {
      // the full roster of initial participants: the chance player plus every seat. It is
      // intentionally fold-independent so that reach-probability maps and reward maps always
      // cover all players (folded ones keep their sunk stakes in the pot).
      std::vector< Player > roster;
      roster.reserve(wstate.player_count() + 1);
      roster.emplace_back(Player::chance);
      for(size_t seat = 0; seat < wstate.player_count(); ++seat) {
         roster.emplace_back(static_cast< Player >(seat));
      }
      return roster;
   }
   [[nodiscard]] Player active_player(const world_state_type& wstate) const;
   static bool is_terminal(const world_state_type& wstate);
   static constexpr bool is_partaking(const world_state_type&, Player) { return true; }
   static double reward(Player player, const world_state_type& wstate);

   template < typename ActionT >
      requires common::is_any_v< ActionT, action_type, chance_outcome_type >
   void transition(world_state_type& worldstate, const ActionT& action) const
   {
      worldstate.apply_action(action);
   }

   observation_type private_observation(
      Player observer,
      const world_state_type& wstate,
      const action_type& action,
      const world_state_type& next_wstate
   ) const;

   observation_type private_observation(
      Player observer,
      const world_state_type& wstate,
      const chance_outcome_type& action,
      const world_state_type& next_wstate
   ) const;

   observation_type public_observation(
      const world_state_type& wstate,
      const action_type& action,
      const world_state_type& next_wstate
   ) const;

   observation_type public_observation(
      const world_state_type& wstate,
      const chance_outcome_type& action,
      const world_state_type& next_wstate
   ) const;

   /// debug purposes
   std::string tiny_repr(const world_state_type& wstate) const;
};

/// string rendering of an observation; reproduces the legacy string encoding
/// ("-", "check"/"bet", card name, "<seat>:?" with NOR player names) so debug
/// output stays byte-identical to the pre-compaction implementation
[[nodiscard]] inline std::string to_string(const Observation& obs)
{
   switch(obs.kind) {
      case Observation::Kind::none: return "-";
      case Observation::Kind::action: return common::to_string(obs.action);
      case Observation::Kind::private_deal: return common::to_string(obs.card);
      case Observation::Kind::public_deal:
         return common::to_string(to_nor_player(obs.player)) + ":?";
   }
   return "-";
}

}  // namespace nor::games::kuhn

namespace fmt {

template <>
struct formatter< nor::games::kuhn::Observation >: formatter< std::string > {
   auto format(const nor::games::kuhn::Observation& obs, format_context& ctx) const
   {
      return formatter< std::string >::format(to_string(obs), ctx);
   }
};

}  // namespace fmt

namespace nor {

template <>
struct fosg_traits< games::kuhn::Infostate > {
   using observation_type = nor::games::kuhn::Observation;
};

template <>
struct fosg_traits< games::kuhn::Environment > {
   using world_state_type = nor::games::kuhn::State;
   using info_state_type = nor::games::kuhn::Infostate;
   using public_state_type = nor::games::kuhn::Publicstate;
   using action_type = nor::games::kuhn::Action;
   using chance_outcome_type = nor::games::kuhn::ChanceOutcome;
   using observation_type = nor::games::kuhn::Observation;
};

}  // namespace nor

namespace std {
template < typename StateType >
   requires common::
      is_any_v< StateType, nor::games::kuhn::Publicstate, nor::games::kuhn::Infostate >
   struct hash< StateType > {
   size_t operator()(const StateType& state) const noexcept { return state.hash(); }
};

}  // namespace std

#endif  // NOR_ENV_KUHN_HPP
