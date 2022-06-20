
#ifndef NOR_TEXAS_HOLDEM_POKER_STATE_HPP
#define NOR_TEXAS_HOLDEM_POKER_STATE_HPP

#include <array>
#include <optional>
#include <range/v3/all.hpp>
#include <sstream>
#include <vector>

#include "common/common.hpp"

namespace texasholdem {

enum class Player {
   chance = -1,
   one = 0,
   two = 1,
   three = 2,
   four = 3,
   five = 4,
   six = 5,
   seven = 6,
   eight = 7,
   nine = 8,
   ten = 9
};

enum class Rank {
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

enum class Suit { diamonds = 0, clubs = 1, hearts = 2, spades = 3 };

struct Card {
   Rank rank;
   Suit suit;
};

inline bool operator==(const Card& card1, const Card& card2)
{
   return card1.rank == card2.rank and card1.suit == card2.suit;
}

enum class ActionType { check = 0, call = 1, bet = 2, raise = 3 };

struct Action {
   ActionType action_type;
   float bet;
};

inline bool operator==(const Action& action1, const Action& action2)
{
   // we are never expecting large floating point values here so the absolute comparison should
   // suffice for the use case at hand
   return action1.action_type == action2.action_type
          and std::fabs(action1.bet - action2.bet) <= std::numeric_limits< float >::epsilon();
}

struct ChanceOutcome {
   Player player;
   Card card;
};

inline bool operator==(const ChanceOutcome& outcome1, const ChanceOutcome& outcome2)
{
   return outcome1.player == outcome2.player and outcome1.card == outcome2.card;
}

struct PokerConfig {
   size_t n_players;
   size_t n_raises_allowed = 2;
   std::vector< float > bet_sizes_round_one = {2};
   std::vector< float > bet_sizes_round_two = {4};
   std::vector< Card > available_cards = {
      {Rank::jack, Suit::clubs},
      {Rank::jack, Suit::hearts},
      {Rank::queen, Suit::clubs},
      {Rank::queen, Suit::hearts},
      {Rank::king, Suit::clubs},
      {Rank::king, Suit::hearts}};
};


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
         const auto& [left_action, right_action] = paired_actions;
         return left_action == right_action;
      });
}

class State {
  public:
   State(sptr< PokerConfig > config);

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
   std::vector< Player > m_remaining_players;
   std::vector< std::optional< Card > > m_player_cards;
   std::optional< Card > m_public_card = std::nullopt;
   History m_history{};
   sptr< PokerConfig > m_config;

   static const std::vector< History >& _all_terminal_histories();
   [[nodiscard]] bool _all_cards_engaged() const;
   [[nodiscard]] bool _has_higher_card(Player player) const;
   void _flip_active_player();
};

}  // namespace leduc

#endif  // NOR_STATE_HPP
