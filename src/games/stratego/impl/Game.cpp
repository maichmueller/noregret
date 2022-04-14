
#include "stratego/Game.hpp"
#include "common/common.hpp"
#include "stratego/Logic.hpp"

namespace stratego {

Status Game::run(const sptr< utils::Plotter< State > >& plotter)
{
   while(true) {
      if(plotter) {
         plotter->plot(state());
      }

      Status outcome = state().status();

      LOGD(std::string("\n") + state().to_string(Team::BLUE, false));
      LOGD2("Status", static_cast< int >(outcome));

      if(state().status() != Status::ONGOING) {
         return state().status();
      }
      LOGD("Running transition.");
      Team active_team = state().active_team();
      auto action = agent(active_team)
                       ->decide_action(
                          state(), state().logic()->valid_actions(state(), active_team));
      LOGD2(
         "Possible Moves",
         aze::utils::VectorPrinter{
            state().logic()->valid_actions(state(), active_team)});
      LOGD2("Selected Action by team " + std::string(common::to_string(active_team)), action);

      m_state->transition(action);
   }
}

void Game::reset()
{
   state().logic()->reset(state());
}

}  // namespace stratego