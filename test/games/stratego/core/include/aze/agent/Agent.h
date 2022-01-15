#pragma once

#include <vector>
#include <random>

#include "aze/game/Defs.h"

namespace aze {

template < class StateType >
class Agent {
  public:
   using state_type = StateType;
   using move_type = typename state_type::move_type;

  protected:
   Team m_team;

  public:
   explicit Agent(Team team) : m_team(team) {}

   virtual ~Agent() = default;

   virtual move_type decide_move(
      const state_type &state,
      const std::vector< move_type > &poss_moves) = 0;
};

template < class StateType >
class RandomAgent: public Agent< StateType > {
   using state_type = StateType;
   using base_type = Agent< state_type >;
   using base_type::base_type;
   using board_type = typename state_type::board_type;
   using move_type = typename state_type::move_type;

  public:
   explicit RandomAgent(int team, unsigned int seed = std::random_device{}())
       : base_type(team), mt{seed}
   {
   }

   move_type decide_move(const state_type &state, const std::vector< move_type > &poss_moves)
      override
   {
      std::array< move_type, 1 > selected_move{move_type{{0, 0}, {0, 0}}};
      std::sample(poss_moves.begin(), poss_moves.end(), selected_move.begin(), 1, mt);

      return selected_move[0];
   }

  private:
   std::mt19937 mt;
};
}  // namespace aze