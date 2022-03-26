#pragma once

#include <string>
#include <xtensor/xtensor.hpp>

#include "StrategoDefs.hpp"
#include "aze/aze.h"
#include "common/common.hpp"

namespace stratego::utils {

template < typename StateType >
class Plotter {
  public:
   virtual ~Plotter() = default;

   virtual void plot(const StateType &state) = 0;
};

constexpr inline Team opponent(Team t)
{
   if(t == Team::BLUE) {
      return Team::RED;
   }
   if(t == Team::RED) {
      return Team::BLUE;
   }
   return Team::NEUTRAL;
}

template < typename T, std::integral IntType >
std::vector< T > flatten_counter(const std::map< T, IntType > &counter)
{
   std::vector< T > flattened_vec;
   for(const auto &[val, count] : counter) {
      auto to_emplace = std::vector< T >(count, val);
      std::copy(to_emplace.begin(), to_emplace.end(), std::back_inserter(flattened_vec));
   }
   return flattened_vec;
}

std::string print_board(
   const Board &board,
   std::optional< aze::Team > team = std::nullopt,
   bool hide_unknowns = false);

}  // namespace stratego::utils

namespace common {

template <>
std::string_view enum_name(stratego::Status e);
template <>
std::string_view enum_name(stratego::Team e);
template <>
std::string_view enum_name(stratego::FightOutcome e);
template <>
std::string_view enum_name(stratego::Token e);
template <>
std::string_view enum_name(stratego::DefinedBoardSizes e);

template <>
auto from_string(std::string_view str) -> stratego::Status;
template <>
auto from_string(std::string_view str) -> stratego::Team;
template <>
auto from_string(std::string_view str) -> stratego::FightOutcome;
template <>
auto from_string(std::string_view str) -> stratego::Token;
template <>
auto from_string(std::string_view str) -> stratego::DefinedBoardSizes;

}  // namespace common

template < aze::utils::is_enum Enum >
requires aze::utils::
   is_any_v< Enum, stratego::Token, stratego::Status, stratego::Team, stratego::FightOutcome >
inline auto &operator<<(std::ostream &os, Enum e)
{
   os << common::enum_name(e);
   return os;
}

template < aze::utils::is_enum Enum >
requires aze::utils::
   is_any_v< Enum, stratego::Token, stratego::Status, stratego::Team, stratego::FightOutcome >
inline auto &operator<<(std::stringstream &ss, Enum e)
{
   ss << common::enum_name(e);
   return ss;
}

// these operator<< definitions are specifically made for gtest which cannot handle the lookup in
// global namespace without throwing multiple template matching errors. As such, the printing
// overload needs to be specialized for every enum INSIDE their original namespace ('stratego' in
// this case for Token and FightOutcome and 'aze' for Team and Status). Why isn't clear. The compile
// error does not occur under the latest GCC trunk, but will occur under gcc 11.1 and 11.2. However,
// even without compilation error, the output still fails
namespace stratego {

inline auto &operator<<(std::ostream &os, Token e)
{
   os << common::enum_name(e);
   return os;
}
inline auto &operator<<(std::ostream &os, FightOutcome e)
{
   os << common::enum_name(e);
   return os;
}
}  // namespace stratego

namespace aze {
inline auto &operator<<(std::ostream &os, Status e)
{
   os << common::enum_name(e);
   return os;
}
inline auto &operator<<(std::ostream &os, Team e)
{
   os << common::enum_name(e);
   return os;
}
}  // namespace aze
