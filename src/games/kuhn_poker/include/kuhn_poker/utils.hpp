
#ifndef NOR_KUHN_POKER_UTILS_HPP
#define NOR_KUHN_POKER_UTILS_HPP

#include "state.hpp"
#include "common/common.hpp"
#include <string>

namespace kuhn {

constexpr common::CEBijection< Card, std::string_view, 3 >
   card_name_bij = {
      std::pair{Card::jack, "jack"},
      std::pair{Card::queen, "queen"},
      std::pair{Card::king, "king"}
};

constexpr common::CEBijection< Action, std::string_view, 2 >
   action_name_bij = {
      std::pair{Action::check, "check"},
      std::pair{Action::bet, "bet"}
};

}

namespace common {

template <>
inline std::string_view enum_name(kuhn::Card card) {
   return kuhn::card_name_bij.at(card);
}

template <>
inline std::string_view enum_name(kuhn::Action action) {
   return kuhn::action_name_bij.at(action);
}

}

#endif  // NOR_KUHN_POKER_UTILS_HPP
