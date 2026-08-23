
#ifndef NOR_BATTLESHIP_UTILS_HPP
#define NOR_BATTLESHIP_UTILS_HPP

#include <functional>
#include <string>

#include "common/common.hpp"
#include "state.hpp"

namespace common {

template <>
inline std::string to_string(const battleship::Cell& cell)
{
   return fmt::format("({},{})", int(cell.row), int(cell.col));
}

template <>
inline std::string to_string(const battleship::Place& place)
{
   return fmt::format("place{}-{}", common::to_string(place.a), common::to_string(place.b));
}

template <>
inline std::string to_string(const battleship::Fire& fire)
{
   return fmt::format("fire{}", common::to_string(fire.target));
}

template <>
inline std::string to_string(const battleship::Action& action)
{
   if(std::holds_alternative< battleship::Place >(action)) {
      return common::to_string(std::get< battleship::Place >(action));
   }
   return common::to_string(std::get< battleship::Fire >(action));
}

template <>
inline std::string to_string(const battleship::Phase& phase)
{
   switch(phase) {
      case battleship::Phase::one_placement: return "one_placement";
      case battleship::Phase::two_placement: return "two_placement";
      case battleship::Phase::one_fire: return "one_fire";
      case battleship::Phase::two_fire: return "two_fire";
      default: return "over";
   }
}

template <>
inline std::string to_string(const battleship::Player& player)
{
   return player == battleship::Player::one ? "one" : "two";
}

template <>
inline std::string to_string(const battleship::Config& config)
{
   return fmt::format(
      "battleship({}x{}, {} ships/fleet, R={}, value={})",
      config.rows,
      config.cols,
      config.ships_per_fleet,
      config.max_shots,
      config.ship_value
   );
}

}  // namespace common

COMMON_ENABLE_PRINT(battleship, Cell);
COMMON_ENABLE_PRINT(battleship, Place);
COMMON_ENABLE_PRINT(battleship, Fire);
COMMON_ENABLE_PRINT(battleship, Action);
COMMON_ENABLE_PRINT(battleship, Phase);
COMMON_ENABLE_PRINT(battleship, Player);

namespace std {

template <>
struct hash< battleship::Cell > {
   size_t operator()(const battleship::Cell& cell) const noexcept
   {
      size_t seed = 0;
      common::hash_combine(seed, std::hash< int >{}(int(cell.row)));
      common::hash_combine(seed, std::hash< int >{}(int(cell.col)));
      return seed;
   }
};

template <>
struct hash< battleship::Place > {
   size_t operator()(const battleship::Place& place) const noexcept
   {
      size_t seed = 0;
      common::hash_combine(seed, std::hash< battleship::Cell >{}(place.a));
      common::hash_combine(seed, std::hash< battleship::Cell >{}(place.b));
      return seed;
   }
};

template <>
struct hash< battleship::Fire > {
   size_t operator()(const battleship::Fire& fire) const noexcept
   {
      size_t seed = 0;
      common::hash_combine(seed, std::hash< battleship::Cell >{}(fire.target));
      return seed;
   }
};

template <>
struct hash< battleship::Action > {
   size_t operator()(const battleship::Action& action) const noexcept
   {
      size_t seed = 0;
      if(std::holds_alternative< battleship::Place >(action)) {
         common::hash_combine(
            seed, std::hash< battleship::Place >{}(std::get< battleship::Place >(action))
         );
      } else {
         common::hash_combine(
            seed, std::hash< battleship::Fire >{}(std::get< battleship::Fire >(action))
         );
      }
      return seed;
   }
};

}  // namespace std

#endif  // NOR_BATTLESHIP_UTILS_HPP
