
#pragma once

#include <algorithm>
#include <functional>
#include <iterator>
#include <random>

#include "State.h"
#include "aze/agent/Agent.h"
#include "aze/utils/logging_macros.h"
#include "aze/utils/utils.h"

namespace aze {

template < class StateType, class LogicType, size_t NPlayers >
class Game {
  public:
   using state_type = StateType;
   using logic_type = LogicType;
   using piece_type = typename state_type::piece_type;
   using token_type = typename piece_type::token_type;
   using position_type = typename state_type::position_type;
   using board_type = typename state_type::board_type;
   using agent_type = Agent< state_type >;
   constexpr static size_t n_teams = NPlayers;

  private:
   uptr< state_type > m_state;
   std::array< sptr< Agent< state_type > >, n_teams > m_agents;

  public:
   Game(uptr< state_type > stateptr, std::array< sptr< agent_type >, n_teams > agents);
   Game(state_type &&state, std::array< sptr< agent_type >, n_teams > agents);
   virtual ~Game() = default;

   virtual Status run_game(const sptr< utils::Plotter< state_type > > &plotter) = 0;
   virtual Status run_step() = 0;
   virtual void reset() = 0;

   constexpr auto nr_players() const { return n_teams; }
   auto agents() { return m_agents; }
   auto agent(Team team) { return m_agents.at(static_cast< unsigned int >(team)); }
   const auto &state() const { return m_state; }
   auto &state() { return m_state; }
};

template < class StateType, class LogicType, size_t NPlayers >
Game< StateType, LogicType, NPlayers >::Game(
   state_type &&state,
   std::array< sptr< Agent< state_type > >, n_teams > agents)
    : m_state(std::make_unique< state_type >(std::move(state))), m_agents(agents)
{
}

template < class StateType, class LogicType, size_t NPlayers >
Game< StateType, LogicType, NPlayers >::Game(
   uptr< state_type > stateptr,
   std::array< sptr< Agent< state_type > >, n_teams > agents)
    : m_state(std::move(stateptr)), m_agents(agents)
{
}

}  // namespace aze