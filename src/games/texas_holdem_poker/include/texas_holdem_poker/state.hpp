
#ifndef NOR_TEXAS_HOLDEM_POKER_STATE_HPP
#define NOR_TEXAS_HOLDEM_POKER_STATE_HPP

#include <array>
#include <optional>
#include <range/v3/all.hpp>
#include <sstream>
#include <variant>
#include <vector>

#include "common/common.hpp"

namespace texholdem {

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

enum class BetLimit { limit = 1, no_limit = 0, pot_limit = 2 };

enum class PokerToken { dealer = 0, small_blind = 1, big_blind = 2 };

template < typename T, std::integral To = size_t >
inline auto as_int(T p)
{
   // we let things silently fail in the call site if an incorrect value is passed in here
   return static_cast< To >(p);
};

struct PokerConfig {
   size_t n_players;
   // how many rounds should the game last? 3 rounds is standard (flop-turn-river)
   size_t n_rounds = 3;
   // how many boardcards are going to be drawn on each round (index i is #cards of round i)
   std::vector< size_t > boardcards_per_round = {3, 1, 1};
   // the starting amount the big blind will need to pay.
   float small_blind = 0.f;
   // the starting amount the small blind will need to pay.
   float big_blind = 0.f;
   // what holdem variant is to be played in each round? Limit/No-Limit/Pot-Limit
   std::vector< std::variant< BetLimit, float > > bet_size_limits;
   // how often can players raise the bet in each round
   std::vector< size_t > bet_nr_limits = {
      std::dynamic_extent,
      std::dynamic_extent,
      std::dynamic_extent};
   // the starting deck to play with
   std::vector< Card > deck = {
      {Rank::two, Suit::diamonds},   {Rank::two, Suit::clubs},
      {Rank::two, Suit::hearts},     {Rank::two, Suit::spades},

      {Rank::three, Suit::diamonds}, {Rank::three, Suit::clubs},
      {Rank::three, Suit::hearts},   {Rank::three, Suit::spades},

      {Rank::four, Suit::diamonds},  {Rank::four, Suit::clubs},
      {Rank::four, Suit::hearts},    {Rank::four, Suit::spades},

      {Rank::five, Suit::diamonds},  {Rank::five, Suit::clubs},
      {Rank::five, Suit::hearts},    {Rank::five, Suit::spades},

      {Rank::six, Suit::diamonds},   {Rank::six, Suit::clubs},
      {Rank::six, Suit::hearts},     {Rank::six, Suit::spades},

      {Rank::seven, Suit::diamonds}, {Rank::seven, Suit::clubs},
      {Rank::seven, Suit::hearts},   {Rank::seven, Suit::spades},

      {Rank::eight, Suit::diamonds}, {Rank::eight, Suit::clubs},
      {Rank::eight, Suit::hearts},   {Rank::eight, Suit::spades},

      {Rank::nine, Suit::diamonds},  {Rank::nine, Suit::clubs},
      {Rank::nine, Suit::hearts},    {Rank::nine, Suit::spades},

      {Rank::ten, Suit::diamonds},   {Rank::ten, Suit::clubs},
      {Rank::ten, Suit::hearts},     {Rank::ten, Suit::spades},

      {Rank::jack, Suit::diamonds},  {Rank::jack, Suit::clubs},
      {Rank::jack, Suit::hearts},    {Rank::jack, Suit::spades},

      {Rank::queen, Suit::diamonds}, {Rank::queen, Suit::clubs},
      {Rank::queen, Suit::hearts},   {Rank::queen, Suit::spades},

      {Rank::king, Suit::diamonds},  {Rank::king, Suit::clubs},
      {Rank::king, Suit::hearts},    {Rank::king, Suit::spades},

      {Rank::ace, Suit::diamonds},   {Rank::ace, Suit::clubs},
      {Rank::ace, Suit::hearts},     {Rank::ace, Suit::spades},
   };
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
      ranges::views::zip(left.sequence, right.sequence),
      [](const auto& paired_actions) {
         const auto& [left_action, right_action] = paired_actions;
         return left_action == right_action;
      }
   );
}

/**
 * @brief stores the currently commited action sequence
 */
class HistorySinceBet {
  public:
   HistorySinceBet(std::vector< std::optional< Action > > cont) : m_container(std::move(cont)) {}

   auto& operator[](Player player) { return m_container[as_int(player)]; }

   auto& operator[](Player player) const { return m_container[as_int(player)]; }

   void reset()
   {
      ranges::for_each(m_container, [](auto& action_opt) { action_opt.reset(); });
   }

   bool all_acted(const std::vector< Player >& remaining_players)
   {
      return ranges::all_of(remaining_players, [&](Player player) {
         return m_container[as_int(player)].has_value();
      });
   }

   auto& container() { return m_container; }
   [[nodiscard]] auto& container() const { return m_container; }

  private:
   std::vector< std::optional< Action > > m_container{};
};

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
   size_t m_round = 0;
   std::vector< Player > m_remaining_players;
   std::vector< std::optional< Card > > m_player_cards;
   std::vector< std::optional< Card > > m_public_cards;
   History m_action_history{};
   History m_history_since_last_bet{};
   std::vector< double > m_stakes;
   unsigned int m_bets_this_round = 0;
   bool m_is_terminal = false;
   bool m_terminal_checked = false;
   sptr< PokerConfig > m_config;

   static const std::vector< History >& _all_terminal_histories();
   [[nodiscard]] bool _all_cards_engaged() const;
   [[nodiscard]] bool _has_higher_card(Player player) const;
   void _flip_active_player();
};

}  // namespace texholdem

#endif  // NOR_STATE_HPP
