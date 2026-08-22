
#ifndef NOR_TEXAS_HOLDEM_POKER_UTILS_HPP
#define NOR_TEXAS_HOLDEM_POKER_UTILS_HPP

#include <functional>
#include <string>

#include "common/common.hpp"
#include "state.hpp"

namespace texholdem {

constexpr common::CEBijection< Rank, std::string_view, 13 > rank_name_bij = {
   std::pair{Rank::two, "two"},
   std::pair{Rank::three, "three"},
   std::pair{Rank::four, "four"},
   std::pair{Rank::five, "five"},
   std::pair{Rank::six, "six"},
   std::pair{Rank::seven, "seven"},
   std::pair{Rank::eight, "eight"},
   std::pair{Rank::nine, "nine"},
   std::pair{Rank::ten, "ten"},
   std::pair{Rank::jack, "jack"},
   std::pair{Rank::queen, "queen"},
   std::pair{Rank::king, "king"},
   std::pair{Rank::ace, "ace"}};

constexpr common::CEBijection< Suit, std::string_view, 4 > suit_name_bij = {
   std::pair{Suit::clubs, "clubs"},
   std::pair{Suit::diamonds, "diamonds"},
   std::pair{Suit::hearts, "hearts"},
   std::pair{Suit::spades, "spades"}};

constexpr common::CEBijection< ActionType, std::string_view, 6 > actiontype_name_bij = {
   std::pair{ActionType::fold, "fold"},
   std::pair{ActionType::check, "check"},
   std::pair{ActionType::call, "call"},
   std::pair{ActionType::bet, "bet"},
   std::pair{ActionType::raise, "raise"},
   std::pair{ActionType::all_in, "all-in"}};

constexpr common::CEBijection< Street, std::string_view, 4 > street_name_bij = {
   std::pair{Street::preflop, "preflop"},
   std::pair{Street::flop, "flop"},
   std::pair{Street::turn, "turn"},
   std::pair{Street::river, "river"}};

constexpr common::CEBijection< Player, std::string_view, 7 > player_name_bij = {
   std::pair{Player::chance, "chance"},
   std::pair{Player::one, "one"},
   std::pair{Player::two, "two"},
   std::pair{Player::three, "three"},
   std::pair{Player::four, "four"},
   std::pair{Player::five, "five"},
   std::pair{Player::six, "six"}};

}  // namespace texholdem

namespace common {

template <>
inline std::string to_string(const texholdem::Rank& value)
{
   return std::string(texholdem::rank_name_bij.at(value));
}

template <>
inline std::string to_string(const texholdem::Suit& value)
{
   return std::string(texholdem::suit_name_bij.at(value));
}

template <>
inline std::string to_string(const texholdem::ActionType& value)
{
   return std::string(texholdem::actiontype_name_bij.at(value));
}

template <>
inline std::string to_string(const texholdem::Street& value)
{
   return std::string(texholdem::street_name_bij.at(value));
}

template <>
inline std::string to_string(const texholdem::Player& value)
{
   return std::string(texholdem::player_name_bij.at(value));
}

template <>
inline std::string to_string(const texholdem::Card& value)
{
   return to_string(value.rank) + "-" + *to_string(value.suit).begin();
}

template <>
inline std::string to_string(const texholdem::Action& value)
{
   switch(value.kind) {
      case texholdem::ActionType::bet:
      case texholdem::ActionType::raise: {
         return fmt::format("{}-->{:.2f}", to_string(value.kind), value.amount);
      }
      default: return to_string(value.kind);
   }
}

}  // namespace common

COMMON_ENABLE_PRINT(texholdem, Rank);
COMMON_ENABLE_PRINT(texholdem, Suit);
COMMON_ENABLE_PRINT(texholdem, Action);
COMMON_ENABLE_PRINT(texholdem, ActionType);
COMMON_ENABLE_PRINT(texholdem, Street);
COMMON_ENABLE_PRINT(texholdem, Player);
COMMON_ENABLE_PRINT(texholdem, Card);

namespace std {

template <>
struct hash< texholdem::Card > {
   size_t operator()(const texholdem::Card& card) const noexcept
   {
      size_t seed = 0;
      common::hash_combine(seed, std::hash< unsigned >{}(unsigned(card.rank)));
      common::hash_combine(seed, std::hash< unsigned >{}(unsigned(card.suit)));
      return seed;
   }
};

template <>
struct hash< texholdem::Action > {
   size_t operator()(const texholdem::Action& action) const noexcept
   {
      size_t seed = 0;
      common::hash_combine(seed, std::hash< unsigned >{}(unsigned(action.kind)));
      common::hash_combine(seed, std::hash< double >{}(action.amount));
      return seed;
   }
};

}  // namespace std

#endif  // NOR_TEXAS_HOLDEM_POKER_UTILS_HPP
