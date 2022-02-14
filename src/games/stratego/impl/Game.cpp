
#include "stratego/Game.hpp"

#include "stratego/Logic.hpp"

namespace stratego {

Status Game::run(const sptr< aze::utils::Plotter< state_type > >& plotter)
{
   while(true) {
      if(plotter) {
         plotter->plot(*state());
      }

      Status outcome = state()->status();

      LOGD(std::string("\n") + state()->to_string(Team::BLUE, false));
      LOGD2("Status", static_cast< int >(outcome));

      if(state()->status() != Status::ONGOING) {
         return state()->status();
      }
      LOGD("Running transition.")
      Team active_team = state()->active_team();
      auto action = agent(active_team)
                       ->decide_action(
                          *state(), state()->logic()->valid_actions(*state(), active_team));
      LOGD2(
         "Possible Moves",
         aze::utils::VectorPrinter{state()->logic()->valid_actions(*state(), active_team)}); // NOLINT
      LOGD2("Selected Action by team " + utils::enum_name(active_team), action);

      transition(action);
   }
}

Status Game::transition(const Action& action)
{
   state()->transition(action);
   return state()->status();
}

void Game::reset()
{
   if(state()->config().fixed_starting_team) {
      state() = std::make_unique< State >(state()->config());
   } else {
      Config cfg_copy = state()->config();
      cfg_copy.starting_team = aze::utils::random::choose(
         std::array{Team::BLUE, Team::RED}, state()->rng());
   }
}

}  // namespace stratego