
#ifndef NOR_TEXAS_HOLDEM_POKER_UTILS_HPP
#define NOR_TEXAS_HOLDEM_POKER_UTILS_HPP

#include <string>

#include "common/common.hpp"
#include "state.hpp"

namespace kuhn {

constexpr common::CEBijection< Rank, std::string_view, 3 > card_name_bij = {
   std::pair{Rank::jack, "jack"},
   std::pair{Rank::queen, "queen"},
   std::pair{Rank::king, "king"}};

constexpr common::CEBijection< Action, std::string_view, 2 > action_name_bij = {
   std::pair{Action::check, "check"},
   std::pair{Action::bet, "bet"}};

constexpr common::CEBijection< Player, std::string_view, 2 > player_name_bij = {
   std::pair{Player::one, "one"},
   std::pair{Player::two, "two"}};

}  // namespace kuhn

namespace common {

template <>
inline std::string to_string(const kuhn::Rank &value)
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

// these operator<< definitions are specifically made for gtest which cannot handle the lookup in
// global namespace without throwing multiple template matching errors.
namespace kuhn {

inline auto &operator<<(std::ostream &os, Player e)
{
   return os << common::to_string(e);
}
inline auto &operator<<(std::ostream &os, Rank e)
{
   return os << common::to_string(e);
}
inline auto &operator<<(std::ostream &os, ChanceOutcome e)
{
   return os << common::to_string(e.card);
}
inline auto &operator<<(std::ostream &os, Action e)
{
   return os << common::to_string(e);
}

}  // namespace kuhn

namespace std {

template <>
struct hash< kuhn::History > {
   size_t operator()(const kuhn::History &history) const
   {
      std::stringstream ss;
      for(auto action : history.sequence) {
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

template <>
struct hash< texholdem::Card > {
   size_t operator()(const texholdem::Card &card) const
   {
      size_t hash = std::hash< texholdem::Rank >{}(card.rank);
      common::hash_combine(hash, std::hash< texholdem::Suit >{}(card.suit));
      return hash;
   }
};


}  // namespace std

#endif  // NOR_TEXAS_HOLDEM_POKER_UTILS_HPP
