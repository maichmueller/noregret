
#include "stratego/Game.hpp"

#include <fmt/format.h>
#include <fmt/ostream.h>
#include <fmt/ranges.h>
#include <spdlog/spdlog.h>

#include "stratego/Logic.hpp"
#include "stratego/StrategoDefs.hpp"

namespace stratego {

Status Game::run(const sptr< utils::Plotter< State > >& plotter)
{
   while(true) {
      if(plotter) {
         plotter->plot(state());
      }

      SPDLOG_DEBUG(state().to_string(Team::BLUE, false));
      SPDLOG_DEBUG(fmt::format("Status: {}", state().status()));

      if(state().status() != Status::ONGOING) {
         return state().status();
      }
      SPDLOG_DEBUG("Running transition.");
      Team active_team = state().active_team();
      auto available_actions = state().logic()->valid_actions(state(), active_team);
      auto action = agent(active_team)->decide_action(state(), available_actions);

      SPDLOG_DEBUG(
         fmt::format("Possible Moves {}", state().logic()->valid_actions(state(), active_team))
      );
      SPDLOG_DEBUG(fmt::format("Selected Action by team {}: {}", active_team, action));

      m_state->transition(action);
   }
}

void Game::reset()
{
   state().logic()->reset(state());
}

}  // namespace stratego
