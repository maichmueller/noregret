
#include "stratego/Game.hpp"

#include "stratego/Logic.hpp"

namespace stratego {

Status Game::run_game(const sptr< aze::utils::Plotter< state_type > >& plotter)
{
   while(true) {
      if(plotter)
         plotter->plot(*state());

      Status outcome = state()->status();

      LOGD(std::string("\n") + state()->string_representation(Team::BLUE, false));
      LOGD2("Status", static_cast< int >(outcome));

      if(state()->status() != Status::ONGOING)
         return state()->status();
      else
         run_step();
   }
}

Status Game::run_step()
{
   LOGD("Running step.")
   Team active_team = state()->active_team();
   auto action = agents()[static_cast<int>(active_team)]->decide_action(
      *state(), state()->logic()->valid_actions(*state(), active_team));
   LOGD2(
      "Possible Moves",
      aze::utils::VectorPrinter{state()->logic()->valid_actions(*state(), active_team)});
   LOGD2("Selected Action by team " + team_name(active_team), action);

   state()->apply_action(action);
   return state()->status();
}

void Game::reset()
{
   state() = std::make_shared< State >(state()->config());
}

}  // namespace stratego