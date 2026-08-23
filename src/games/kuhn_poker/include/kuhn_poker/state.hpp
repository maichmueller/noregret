
#ifndef NOR_KUHN_POKER_STATE_HPP
#define NOR_KUHN_POKER_STATE_HPP

#include <array>
#include <cstddef>
#include <optional>
#include <ranges>
#include <sstream>
#include <vector>

namespace kuhn {

enum class Player { chance = -1, one = 0, two = 1, three = 2 };

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
using History = std::vector< Action >;

inline bool operator==(const History& left, const History& right)
{
   if(left.size() != right.size()) {
      return false;
   }
   return std::ranges::all_of(std::views::zip(left, right), [](const auto& paired_actions) {
      return std::get< 0 >(paired_actions) == std::get< 1 >(paired_actions);
   });
}

/**
 * @brief Kuhn poker generalized to an arbitrary number of players.
 *
 * Rules:
 * - each player antes 1 chip and receives one private card from the given deck
 *   (deck size must be >= player count; decks are expected to hold unique cards, but true
 *   ties at showdown are still handled defensively via an even pot split).
 * - betting proceeds in seat order P1 -> P2 -> ... -> PN cyclically until closure.
 * - there is at most a single bet of size 1 per deal (no raises). 'Action::bet' either opens
 *   the betting (if no bet is outstanding) or calls/matches an outstanding bet.
 *   'Action::check' passes if no bet is outstanding and folds otherwise. This contextual
 *   reading reproduces the classic 2-player encoding exactly.
 * - the hand ends ('closure') once every active player has matched the outstanding wager (or
 *   passed when none exists), or early once all but one player have folded (the remaining
 *   player takes the whole pot immediately).
 * - at showdown the highest card among non-folded players wins the whole pot. True ties (only
 *   possible with duplicated deck entries) split the pot evenly, with any remainder chips
 *   going to the tied players in seat order.
 */
class State {
  public:
   /// upper bound on seats: bounded by the number of distinct card ranks available
   static constexpr size_t max_player_count = 13;

   State(
      std::vector< Card > card_pool = {Card::jack, Card::queen, Card::king},
      size_t player_count = 2
   );

   void apply_action(Action action);
   void apply_action(ChanceOutcome action);
   [[nodiscard]] bool is_valid(Action action) const;
   [[nodiscard]] bool is_valid(ChanceOutcome outcome) const;
   [[nodiscard]] bool is_terminal() const;
   [[nodiscard]] std::vector< Action > actions() const;
   [[nodiscard]] std::vector< ChanceOutcome > chance_actions() const;
   [[nodiscard]] double chance_probability(ChanceOutcome action) const;
   [[nodiscard]] int payoff(Player player) const;

   [[nodiscard]] auto active_player() const { return m_active_player; }
   [[nodiscard]] auto card(Player player) const
   {
      return m_player_cards.at(static_cast< uint8_t >(player));
   }
   [[nodiscard]] auto& history() const { return m_history; }
   /// the acting seat for each entry of history() (parallel to history())
   [[nodiscard]] auto& history_actors() const { return m_actors; }
   [[nodiscard]] auto& cards() const { return m_player_cards; }
   [[nodiscard]] size_t player_count() const { return m_player_count; }
   [[nodiscard]] bool folded(Player player) const
   {
      return m_folded.at(static_cast< uint8_t >(player)) != 0;
   }

  private:
   size_t m_player_count;
   std::vector< std::optional< Card > > m_player_cards;
   History m_history{};
   std::vector< Player > m_actors{};
   std::vector< char > m_folded;
   /// number of active players that still owe an action before the betting can close
   int m_open_responses = 0;
   bool m_bet_outstanding = false;
   Player m_active_player = Player::chance;
   std::vector< Card > m_card_pool;

   [[nodiscard]] bool _all_cards_engaged() const;
   [[nodiscard]] size_t _dealt_count() const;
   [[nodiscard]] int _active_count() const;
   [[nodiscard]] Player _next_active_seat(Player after) const;
   [[nodiscard]] int _contribution(Player player) const;
};

}  // namespace kuhn

#endif  // NOR_KUHN_POKER_STATE_HPP
