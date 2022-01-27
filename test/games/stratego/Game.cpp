
#include "Game.h"

#include "Logic.h"

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
   size_t turn = state()->turn_count() % n_teams;
   auto action = agents()[turn]->decide_action(
      *state(), state()->logic()->valid_actions(*state(), Team(turn)));
   LOGD2(
      "Possible Moves",
      aze::utils::VectorPrinter{state()->logic()->valid_actions(*state(), Team(turn))});
   LOGD2("Selected Action by team " + std::to_string(turn), action);

   state()->apply_action(action);

   return state()->status();
}

void Game::reset()
{
   state() = std::make_shared< State >(state()->config());
}

}  // namespace stratego