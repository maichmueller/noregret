
#ifndef NOR_STATE_HPP
#define NOR_STATE_HPP

#include <array>
#include <optional>
#include <range/v3/all.hpp>
#include <sstream>
#include <vector>

namespace kuhn {

enum class Player { chance = -1, one = 0, two = 1 };

enum class Card { jack = 0, queen = 1, king = 2 };

using ChanceAction = Card;

enum class Action { check = 0, bet };

/**
 * @brief stores the currently commited action sequence
 *
 * It's a simple wrapper for the action sequence to allow for std::hash specialization and
 * operator== overloading.
 */
struct History {
   std::vector< Action > sequence{};
};

bool operator==(const History& left, const History& right)
{
   if(left.sequence.size() != right.sequence.size()) {
      return false;
   }
   return ranges::all_of(
      ranges::views::zip(left.sequence, right.sequence), [](const auto& paired_actions) {
         return std::get< 0 >(paired_actions) == std::get< 1 >(paired_actions);
      });
}

class State {
  public:
   State() = default;

   void apply_action(Action action);
   void apply_action(ChanceAction action);
   [[nodiscard]] bool is_valid(Action action) const;
   [[nodiscard]] bool is_valid(ChanceAction action) const;
   [[nodiscard]] bool is_terminal() const;
   [[nodiscard]] std::vector< Action > actions() const;
   [[nodiscard]] std::vector< ChanceAction > chance_actions() const;
   [[nodiscard]] int16_t payoff(Player player) const;

   [[nodiscard]] auto active_player() const { return m_active_player; }
   [[nodiscard]] auto card(Player player) const
   {
      return m_player_cards[static_cast< uint8_t >(player)];
   }
   [[nodiscard]] auto& history() const { return m_history; }

  private:
   Player m_active_player = Player::chance;
   std::array< std::optional< Card >, 2 > m_player_cards = {std::nullopt, std::nullopt};
   History m_history{};
   bool m_terminal = false;
   bool m_terminal_checked = false;

   static const std::vector< History >& _all_terminal_histories();
   bool _all_cards_engaged() const;
};

}  // namespace kuhn

namespace std {

template <>
struct hash< kuhn::History > {
   size_t operator()(const kuhn::History& history) const
   {
      std::stringstream ss;
      for(auto action : history.sequence) {
         ss << std::to_string(static_cast< int >(action));
      }
      return std::hash< std::string >{}(ss.str());
   }
};
}  // namespace std

#endif  // NOR_STATE_HPP
