
#include "rock_paper_scissors/state.hpp"
#include "common/common.hpp"

namespace rps {

void State::apply_action(Action action)
{
   if(m_picks[0].has_value()) {
      if(m_picks[1].has_value()) {
         throw std::logic_error("Each player already chose an action.");
      }
   }
   m_active_team = Team::two;
   m_picks[static_cast<uint8_t>(action.team)] = action;
}

double State::payoff(Team team) const
{
   constexpr common::CEMap< std::array< Hand, 2 >, double, 9 > winner_map{
      std::pair{std::array{Hand::paper, Hand::paper}, 0.},
      std::pair{std::array{Hand::paper, Hand::rock}, 1.},
      std::pair{std::array{Hand::paper, Hand::scissors}, -1.},
      std::pair{std::array{Hand::scissors, Hand::paper}, 1.},
      std::pair{std::array{Hand::scissors, Hand::rock}, -1.},
      std::pair{std::array{Hand::scissors, Hand::scissors}, 0.},
      std::pair{std::array{Hand::rock, Hand::paper}, -1.},
      std::pair{std::array{Hand::rock, Hand::rock}, 0.},
      std::pair{std::array{Hand::rock, Hand::scissors}, 1.},
   };
   if(team == Team::one) {
      return winner_map.at({m_picks[0].value().hand, m_picks[1].value().hand});
   } else {
      return winner_map.at({m_picks[1].value().hand, m_picks[0].value().hand});
   }
}

bool State::terminal() const
{
   return m_picks[0].has_value() and m_picks[1].has_value();
}

}  // namespace rps