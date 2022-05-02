#ifndef NOR_RPS_STATE_HPP
#define NOR_RPS_STATE_HPP

#include <array>
#include <optional>
#include <unordered_map>

namespace rps {

enum class Team { one = 0, two };

enum class Hand { rock = 0, paper, scissors };

struct Action {
   Team team;
   Hand hand;
};

inline bool operator==(const Action& action1, const Action& action2)
{
   return action1.team == action2.team and action1.hand == action2.hand;
}

class State {
  public:
   State() = default;

   void apply_action(Action action);

   double payoff(Team team) const;
   bool terminal() const;

   [[nodiscard]] auto active_team() const { return m_active_team; }
   [[nodiscard]] auto& picks() const { return m_picks; }

  private:
   Team m_active_team = Team::one;
   std::array< std::optional< Action >, 2 > m_picks = {};
};

}  // namespace rps

#endif  // NOR_RPS_STATE_HPP
