
#ifndef NOR_STATE_HPP
#define NOR_STATE_HPP

#include <array>
#include <optional>
#include <range/v3/all.hpp>
#include <sstream>
#include <vector>

namespace kuhn {

enum class Player { chance = -1, one = 0, two = 1 };

enum class Card {
   two = 2,
   three = 3,
   four = 4,
   five = 5,
   six = 6,
   seven = 7,
   eight = 8,
   nine = 9,
   ten = 10,
   jack = 11,
   queen = 12,
   king = 13,
   ace = 14
};

enum class Action { check = 0, bet };

struct ChanceOutcome {
   Player player;
   Card card;
};

inline bool operator==(const ChanceOutcome& outcome1, const ChanceOutcome& outcome2)
{
   return outcome1.player == outcome2.player and outcome1.card == outcome2.card;
}

/**
 * @brief stores the currently commited action sequence
 *
 * It's a simple wrapper for the action sequence to allow for std::hash specialization and
 * operator== overloading.
 */
struct History {
   std::vector< Action > sequence{};
};

inline bool operator==(const History& left, const History& right)
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
   State(std::vector< Card > card_pool = {Card::jack, Card::queen, Card::king})
       : m_card_pool(std::move(card_pool))
   {
   }

   void apply_action(Action action);
   void apply_action(ChanceOutcome action);
   [[nodiscard]] bool is_valid(Action action) const;
   [[nodiscard]] bool is_valid(ChanceOutcome outcome) const;
   [[nodiscard]] bool is_terminal() const;
   [[nodiscard]] std::vector< Action > actions() const;
   [[nodiscard]] std::vector< ChanceOutcome > chance_actions() const;
   [[nodiscard]] double chance_probability(ChanceOutcome action) const;
   [[nodiscard]] int16_t payoff(Player player) const;

   [[nodiscard]] auto active_player() const { return m_active_player; }
   [[nodiscard]] auto card(Player player) const
   {
      return m_player_cards[static_cast< uint8_t >(player)];
   }
   [[nodiscard]] auto& history() const { return m_history; }
   [[nodiscard]] auto& cards() const { return m_player_cards; }

  private:
   Player m_active_player = Player::chance;
   std::array< std::optional< Card >, 2 > m_player_cards = {std::nullopt, std::nullopt};
   History m_history{};
   std::vector< Card > m_card_pool;

   static const std::vector< History >& _all_terminal_histories();
   [[nodiscard]] bool _all_cards_engaged() const;
   bool _has_higher_card(Player player) const;
};

}  // namespace kuhn

#endif  // NOR_STATE_HPP
