#pragma once

#include <random>
#include <vector>

#include "aze/game/Defs.h"

namespace aze {

template < class StateType >
class Agent {
  public:
   using state_type = StateType;
   using action_type = typename state_type::action_type;

  protected:
   Team m_team;

  public:
   explicit Agent(Team team) : m_team(team) {}

   virtual ~Agent() = default;



   virtual action_type decide_action(
      const state_type &state,
      const std::vector< action_type > &poss_moves) = 0;
};

template < class StateType >
class RandomAgent: public Agent< StateType > {
   using base_type = Agent< StateType >;
   using base_type::base_type;

  public:
   explicit RandomAgent(Team team, unsigned int seed = std::random_device{}())
       : base_type(team), mt{seed}
   {
   }

   typename base_type::action_type decide_action(
      const typename base_type::state_type &state,
      const std::vector< typename base_type::action_type > &poss_moves) override
   {
      std::array< typename base_type::action_type, 1 > selected_move{
         typename base_type::action_type{{0, 0}, {0, 0}}};
      std::sample(poss_moves.begin(), poss_moves.end(), selected_move.begin(), 1, mt);

      return selected_move[0];
   }

  private:
   std::mt19937 mt;
};
}  // namespace aze