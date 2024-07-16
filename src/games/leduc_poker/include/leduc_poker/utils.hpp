
#ifndef NOR_LEDUC_POKER_UTILS_HPP
#define NOR_LEDUC_POKER_UTILS_HPP

#include <string>

#include "common/common.hpp"
#include "state.hpp"

namespace leduc {

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
   std::pair{Suit::diamonds, "diamonds"},
   std::pair{Suit::clubs, "clubs"},
   std::pair{Suit::hearts, "hearts"},
   std::pair{Suit::spades, "spades"}};

constexpr common::CEBijection< ActionType, std::string_view, 3 > actiontype_name_bij = {
   std::pair{ActionType::check, "check"},
   std::pair{ActionType::fold, "fold"},
   std::pair{ActionType::bet, "bet"}};

constexpr common::CEBijection< Player, std::string_view, 11 > player_name_bij = {
   std::pair{Player::chance, "chance"},
   std::pair{Player::one, "one"},
   std::pair{Player::two, "two"},
   std::pair{Player::three, "three"},
   std::pair{Player::four, "four"},
   std::pair{Player::five, "five"},
   std::pair{Player::six, "six"},
   std::pair{Player::seven, "seven"},
   std::pair{Player::eight, "eight"},
   std::pair{Player::nine, "nine"},
   std::pair{Player::ten, "ten"}};

}  // namespace leduc

namespace common {

template <>
inline std::string to_string(const leduc::Rank &value)
{
   return std::string(leduc::rank_name_bij.at(value));
}

template <>
inline std::string to_string(const leduc::Suit &value)
{
   return std::string(leduc::suit_name_bij.at(value));
}

template <>
inline std::string to_string(const leduc::ActionType &value)
{
   return std::string(leduc::actiontype_name_bij.at(value));
}

template <>
inline std::string to_string(const leduc::Player &value)
{
   return std::string(leduc::player_name_bij.at(value));
}

template <>
inline std::string to_string(const leduc::Card &value)
{
   return to_string(value.rank) + "-" + *to_string(value.suit).begin();
}

template <>
inline std::string to_string(const leduc::Action &value)
{
   return fmt::format(
      "{}{}",
      common::to_string(value.action_type),
      value.action_type == leduc::ActionType::bet ? fmt::format("-->{:.2f}", value.bet) : ""
   );
}

}  // namespace common

COMMON_ENABLE_PRINT(leduc, Rank);
COMMON_ENABLE_PRINT(leduc, Suit);
COMMON_ENABLE_PRINT(leduc, Action);
COMMON_ENABLE_PRINT(leduc, ActionType);
COMMON_ENABLE_PRINT(leduc, Player);
COMMON_ENABLE_PRINT(leduc, Card);

namespace std {

template <>
struct hash< leduc::Action > {
   size_t operator()(const leduc::Action &action) const
   {
      std::stringstream ss;
      ss << action.action_type << "_" << action.bet;
      return std::hash< std::string >{}(ss.str());
   }
};

template <>
struct hash< leduc::Card > {
   size_t operator()(const leduc::Card &card) const noexcept
   {
      size_t seed{0};
      common::hash_combine(seed, std::hash< leduc::Rank >{}(card.rank));
      common::hash_combine(seed, std::hash< leduc::Suit >{}(card.suit));
      return seed;
   }
};

template <>
struct hash< leduc::HistorySinceBet > {
   size_t operator()(const leduc::HistorySinceBet &history) const noexcept
   {
      auto to_string = [](const auto &action) {
         return (action.has_value() ? common::to_string(action.value()) : std::string("?"));
      };
      std::string str = to_string(history[leduc::Player::one]);
      for(auto action_iter = history.container().begin() + 1;
          action_iter != history.container().end() - 1;
          action_iter++) {
         str.append(to_string(*action_iter) + "-");
      }
      str.append(to_string(history.container().back()));
      return std::hash< std::string >{}(str);
   }
};

}  // namespace std

#endif  // NOR_LEDUC_POKER_UTILS_HPP
