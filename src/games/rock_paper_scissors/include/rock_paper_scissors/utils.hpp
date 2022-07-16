
#ifndef NOR_RPS__UTILS_HPP
#define NOR_RPS__UTILS_HPP

#include <string>

#include "common/common.hpp"
#include "state.hpp"

namespace rps {

constexpr common::CEBijection< Hand, std::string_view, 3 > hand_name_bij = {
   std::pair{Hand::rock, "rock"},
   std::pair{Hand::paper, "paper"},
   std::pair{Hand::scissors, "scissors"}};

constexpr common::CEBijection< Team, std::string_view, 2 > team_name_bij = {
   std::pair{Team::one, "one"},
   std::pair{Team::two, "two"}};

}  // namespace rps

namespace common {

template <>
inline std::string to_string(const rps::Hand& value)
{
   return std::string(rps::hand_name_bij.at(value));
}

template <>
inline std::string to_string(const rps::Team& value)
{
   return std::string(rps::team_name_bij.at(value));
}

template <>
inline std::string to_string(const rps::Action& value)
{
   return std::string(rps::team_name_bij.at(value.team)) + "-"
          + std::string(rps::hand_name_bij.at((value.hand)));
}


template <>
struct printable<rps::Hand> : std::true_type {};
template <>
struct printable<rps::Team> : std::true_type {};
template <>
struct printable<rps::Action> : std::true_type {};

}  // namespace common

namespace std {
template < typename ActionType >
requires common::is_any_v< ActionType, rps::Action, const rps::Action > struct hash< ActionType > {
   size_t operator()(const ActionType& action) const noexcept
   {
      return hash< std::string >{}(
         common::to_string(action.team) + "-" + common::to_string(action.hand));
   }
};

inline auto& operator<<(std::ostream& os, const rps::Action& action)
{
   os << common::to_string(action);
   return os;
}

}  // namespace std

#endif  // NOR_RPS_UTILS_HPP
