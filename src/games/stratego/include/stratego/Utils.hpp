#pragma once

#include <string>
#include <xtensor/xtensor.hpp>

#include "StrategoDefs.hpp"
#include "aze/aze.h"

namespace stratego::utils {

template < typename Enum >
requires std::is_enum_v< Enum > std::string enum_name(Enum e);

template <>
std::string enum_name(Status e);
template <>
std::string enum_name(Team e);
template <>
std::string enum_name(FightOutcome e);
template <>
std::string enum_name(Token e);
template <>
std::string enum_name(DefinedBoardSizes e);

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

std::string print_board(const Board &board, std::optional<aze::Team> team = std::nullopt, bool hide_unknowns = false);

}  // namespace stratego::utils

template < aze::utils::is_enum Enum >
requires aze::utils::
   is_any_v< Enum, stratego::Token, stratego::Status, stratego::Team, stratego::FightOutcome >
inline auto &operator<<(std::ostream &os, Enum e)
{
   os << stratego::utils::enum_name(e);
   return os;
}