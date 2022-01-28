#pragma once

#include "aze/aze.h"

#include <string>
#include <xtensor/xtensor.hpp>

#include "StrategoDefs.hpp"

namespace stratego::utils {

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

std::string print_board(const Board &board, aze::Team team, bool hide_unknowns);

}  // namespace stratego::utils