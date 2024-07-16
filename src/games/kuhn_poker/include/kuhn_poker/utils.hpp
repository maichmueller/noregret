
#ifndef NOR_KUHN_POKER_UTILS_HPP
#define NOR_KUHN_POKER_UTILS_HPP

#include <string>

#include "common/common.hpp"
#include "state.hpp"
// #include ""

namespace kuhn {

constexpr common::CEBijection< Card, std::string_view, 13 > card_name_bij = {
   std::pair{Card::two, "two"},
   std::pair{Card::three, "three"},
   std::pair{Card::four, "four"},
   std::pair{Card::five, "five"},
   std::pair{Card::six, "six"},
   std::pair{Card::seven, "seven"},
   std::pair{Card::eight, "eight"},
   std::pair{Card::nine, "nine"},
   std::pair{Card::ten, "ten"},
   std::pair{Card::jack, "jack"},
   std::pair{Card::queen, "queen"},
   std::pair{Card::king, "king"},
   std::pair{Card::ace, "ace"}};

constexpr common::CEBijection< Action, std::string_view, 2 > action_name_bij = {
   std::pair{Action::check, "check"},
   std::pair{Action::bet, "bet"}};

constexpr common::CEBijection< Player, std::string_view, 2 > player_name_bij = {
   std::pair{Player::one, "one"},
   std::pair{Player::two, "two"}};

}  // namespace kuhn

namespace common {

template <>
inline std::string to_string(const kuhn::Card &value)
{
   return std::string(kuhn::card_name_bij.at(value));
}

template <>
inline std::string to_string(const kuhn::Action &value)
{
   return std::string(kuhn::action_name_bij.at(value));
}

template <>
inline std::string to_string(const kuhn::Player &value)
{
   return std::string(kuhn::player_name_bij.at(value));
}

template <>
inline std::string to_string(const kuhn::ChanceOutcome &value)
{
   return std::string(kuhn::card_name_bij.at(value.card));
}

}  // namespace common

COMMON_ENABLE_PRINT(kuhn, Card);
COMMON_ENABLE_PRINT(kuhn, Action);
COMMON_ENABLE_PRINT(kuhn, Player);
COMMON_ENABLE_PRINT(kuhn, ChanceOutcome);

// // these operator<< definitions are specifically made for gtest which cannot handle the lookup in
// // global namespace without throwing multiple template matching errors.
// namespace kuhn {
//
// inline auto &operator<<(std::ostream &os, Player e)
// {
//    return os << common::to_string(e);
// }
// inline auto &operator<<(std::ostream &os, Card e)
// {
//    return os << common::to_string(e);
// }
// inline auto &operator<<(std::ostream &os, ChanceOutcome e)
// {
//    return os << common::to_string(e.card);
// }
// inline auto &operator<<(std::ostream &os, Action e)
// {
//    return os << common::to_string(e);
// }
//
// }  // namespace kuhn

namespace std {

template <>
struct hash< kuhn::History > {
   size_t operator()(const kuhn::History &history) const
   {
      std::stringstream ss;
      for(auto action : history) {
         ss << std::to_string(static_cast< int >(action));
      }
      return std::hash< std::string >{}(ss.str());
   }
};

template <>
struct hash< kuhn::ChanceOutcome > {
   size_t operator()(const kuhn::ChanceOutcome &chance_outcome) const
   {
      std::stringstream ss;
      ss << chance_outcome.player << chance_outcome.card;
      return std::hash< std::string >{}(ss.str());
   }
};
}  // namespace std

#endif  // NOR_KUHN_POKER_UTILS_HPP
