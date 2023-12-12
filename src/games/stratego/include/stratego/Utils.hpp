#pragma once

#include <fmt/ranges.h>

#include <concepts>
#include <optional>
#include <ostream>
#include <ranges>
#include <string>
#include <string_view>
#include <vector>

#include "StrategoDefs.hpp"
#include "common/string_utils.hpp"

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
   flattened_vec.reserve(counter.size());
   for(const auto &[val, count] : counter) {
      for([[maybe_unused]] auto ___ : std::views::iota(static_cast< IntType >(0), count)) {
         flattened_vec.emplace_back(val);
      }
   }
   return flattened_vec;
}

std::string print_board(
   const Board &board,
   std::optional< stratego::Team > team = std::nullopt,
   bool hide_unknowns = false
);

}  // namespace stratego::utils

namespace stratego {
// forward-declare the action class here
class Action;
}  // namespace stratego

namespace common {

template <>
std::string to_string(const stratego::Status &e);
template <>
std::string to_string(const stratego::Team &e);
template <>
std::string to_string(const stratego::FightOutcome &e);
template <>
std::string to_string(const stratego::Token &e);
template <>
std::string to_string(const stratego::DefinedBoardSizes &e);
template <>
std::string to_string(const stratego::Action &e);

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

COMMON_ENABLE_PRINT(stratego, Status);
COMMON_ENABLE_PRINT(stratego, Team);
COMMON_ENABLE_PRINT(stratego, FightOutcome);
COMMON_ENABLE_PRINT(stratego, Token);
COMMON_ENABLE_PRINT(stratego, DefinedBoardSizes);
COMMON_ENABLE_PRINT(stratego, Action);
COMMON_NO_RANGE_FORMATTER(stratego::Action);

// these operator<< definitions are specifically made for gtest which cannot handle the lookup in
// global namespace without throwing multiple template matching errors. As such, the printing
// overload needs to be specialized for every enum INSIDE their original namespace ('stratego' in
// this case for Token and FightOutcome and 'aze' for Team and Status). Why isn't clear. The compile
// error does not occur under the latest GCC trunk, but will occur under gcc 11.1 and 11.2. However,
// even without compilation error, the output still fails
// namespace stratego {
//
// inline auto &operator<<(std::ostream &os, Token e)
// {
//    return os << common::to_string(e);
// }
// inline auto &operator<<(std::ostream &os, FightOutcome e)
// {
//    return os << common::to_string(e);
// }
// }  // namespace stratego
