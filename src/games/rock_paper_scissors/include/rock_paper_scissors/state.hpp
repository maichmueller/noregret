#ifndef NOR_RPS_STATE_HPP
#define NOR_RPS_STATE_HPP

#include <array>
#include <optional>
#include <unordered_map>

namespace rps {

enum class Team { one = 0, two };

enum class Hand { rock = 0, paper, scissors };

class State {
  public:
   State() = default;

   void apply_action(Hand action);

   double payoff(Team team) const;
   bool terminal() const;

   [[nodiscard]] auto active_team() const { return m_active_team; }
   [[nodiscard]] auto& picks() const { return m_picks; }

  private:
   Team m_active_team = Team::one;
   std::array< std::optional< Hand >, 2 > m_picks = {};
};

}  // namespace rps

#endif  // NOR_RPS_STATE_HPP
