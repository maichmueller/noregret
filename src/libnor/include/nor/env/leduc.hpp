
#ifndef NOR_ENV_LEDUC_HPP
#define NOR_ENV_LEDUC_HPP

#include <range/v3/all.hpp>
#include <string>
#include <vector>

#include "common/common.hpp"
#include "leduc_poker/leduc_poker.hpp"
#include "nor/fosg_states.hpp"
#include "nor/fosg_traits.hpp"
#include "nor/game_defs.hpp"

namespace nor::games::leduc {

using namespace ::leduc;

inline auto to_leduc_player(const nor::Player& player)
{
   return static_cast< leduc::Player >(player);
}
inline auto to_nor_player(const leduc::Player& player)
{
   return static_cast< nor::Player >(player);
}

using Observation = std::string;

std::string
observation(const State& state, std::optional< Player > observing_player = std::nullopt);

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
   using chance_outcome_type = Card;
   using observation_type = Observation;
   // nor fosg traits
   static constexpr size_t max_player_count() { return 2; }
   static constexpr size_t player_count() { return 2; }
   static constexpr bool serialized() { return true; }
   static constexpr bool unrolled() { return true; }
   static constexpr Stochasticity stochasticity() { return Stochasticity::choice; }

   Environment() = default;

   std::vector< action_type > actions(Player, const world_state_type& wstate) const
   {
      return wstate.actions();
   }
   inline std::vector< chance_outcome_type > chance_actions(const world_state_type& wstate) const
   {
      return wstate.chance_actions();
   }

   [[nodiscard]] std::vector<
      PlayerInformedType< std::optional< std::variant< chance_outcome_type, action_type > > > >
   private_history(Player player, const world_state_type& wstate) const;

   [[nodiscard]] std::vector<
      PlayerInformedType< std::optional< std::variant< chance_outcome_type, action_type > > > >
   public_history(const world_state_type& wstate) const;

   [[nodiscard]] std::vector<
      PlayerInformedType< std::variant< chance_outcome_type, action_type > > >
   open_history(const world_state_type& wstate) const;

   inline double
   chance_probability(const world_state_type& wstate, const chance_outcome_type& outcome) const
   {
      return wstate.chance_probability(outcome);
   }

   static inline std::vector< Player > players(const world_state_type& wstate)
   {
      auto rem_players = wstate.remaining_players();
      std::vector< Player > vec_out;
      vec_out.reserve(rem_players.size());
      std::ranges::copy(
         rem_players | std::ranges::views::transform([](auto p) { return Player(p); }),
         std::back_inserter(vec_out)
      );
      return vec_out;
   }
   [[nodiscard]] Player active_player(const world_state_type& wstate) const;
   static bool is_terminal(world_state_type& wstate);
   static constexpr bool is_partaking(const world_state_type&, Player) { return true; }
   static double reward(Player player, world_state_type& wstate);

   template < typename ActionT >
   void transition(world_state_type& worldstate, const ActionT& action) const
      requires common::is_any_v< ActionT, action_type, chance_outcome_type >
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
   observation_type tiny_repr(const world_state_type& wstate) const;
};

}  // namespace nor::games::leduc

namespace nor {

template <>
struct fosg_traits< games::leduc::Infostate > {
   using observation_type = nor::games::leduc::Observation;
};

template <>
struct fosg_traits< games::leduc::Environment > {
   using world_state_type = nor::games::leduc::State;
   using info_state_type = nor::games::leduc::Infostate;
   using public_state_type = nor::games::leduc::Publicstate;
   using action_type = nor::games::leduc::Action;
   using chance_outcome_type = nor::games::leduc::Card;
   using observation_type = nor::games::leduc::Observation;
};

}  // namespace nor

namespace std {
template < typename StateType >
   requires common::
      is_any_v< StateType, nor::games::leduc::Publicstate, nor::games::leduc::Infostate >
   struct hash< StateType > {
   size_t operator()(const StateType& state) const noexcept { return state.hash(); }
};

}  // namespace std

#endif  // NOR_ENV_LEDUC_HPP
