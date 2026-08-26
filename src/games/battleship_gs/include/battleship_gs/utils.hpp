
#ifndef NOR_BATTLESHIP_GS_UTILS_HPP
#define NOR_BATTLESHIP_GS_UTILS_HPP

#include <functional>
#include <string>
#include <variant>

#include "battleship_gs/state.hpp"
#include "common/common.hpp"

namespace common {

template <>
inline std::string to_string(const battleship_gs::Cell& cell)
{
   return fmt::format("({},{})", int(cell.row), int(cell.col));
}

template <>
inline std::string to_string(const battleship_gs::ShipSpec& ship)
{
   return fmt::format("(len {}, value {})", unsigned(ship.length), ship.value);
}

template <>
inline std::string to_string(const battleship_gs::Place& place)
{
   return fmt::format(
      "place{}+({},{})", common::to_string(place.start), int(place.drow), int(place.dcol)
   );
}

template <>
inline std::string to_string(const battleship_gs::Fire& fire)
{
   return fmt::format("fire{}", common::to_string(fire.target));
}

template <>
inline std::string to_string(const battleship_gs::Action& action)
{
   if(std::holds_alternative< battleship_gs::Place >(action)) {
      return common::to_string(std::get< battleship_gs::Place >(action));
   }
   return common::to_string(std::get< battleship_gs::Fire >(action));
}

template <>
inline std::string to_string(const battleship_gs::Phase& phase)
{
   switch(phase) {
      case battleship_gs::Phase::one_placement: return "one_placement";
      case battleship_gs::Phase::two_placement: return "two_placement";
      case battleship_gs::Phase::one_fire: return "one_fire";
      case battleship_gs::Phase::two_fire: return "two_fire";
      default: return "over";
   }
}

template <>
inline std::string to_string(const battleship_gs::Player& player)
{
   return player == battleship_gs::Player::one ? "one" : "two";
}

template <>
inline std::string to_string(const battleship_gs::Config& config)
{
   std::string out = fmt::format(
      "battleship_gs({}x{}, r={}, gamma={}, S=[",
      config.rows,
      config.cols,
      config.max_shots,
      config.loss_multiplier
   );
   for(const auto& [length, value] : config.fleet) {
      out += fmt::format("({},{})", unsigned(length), value);
   }
   out += "])";
   return out;
}

}  // namespace common

COMMON_ENABLE_PRINT(battleship_gs, Cell);
COMMON_ENABLE_PRINT(battleship_gs, ShipSpec);
COMMON_ENABLE_PRINT(battleship_gs, Place);
COMMON_ENABLE_PRINT(battleship_gs, Fire);
COMMON_ENABLE_PRINT(battleship_gs, Action);
COMMON_ENABLE_PRINT(battleship_gs, Phase);
COMMON_ENABLE_PRINT(battleship_gs, Player);

namespace std {

template <>
struct hash< battleship_gs::Cell > {
   size_t operator()(const battleship_gs::Cell& cell) const noexcept
   {
      size_t seed = 0;
      common::hash_combine(seed, std::hash< int >{}(int(cell.row)));
      common::hash_combine(seed, std::hash< int >{}(int(cell.col)));
      return seed;
   }
};

template <>
struct hash< battleship_gs::Place > {
   size_t operator()(const battleship_gs::Place& place) const noexcept
   {
      size_t seed = 0;
      common::hash_combine(seed, std::hash< battleship_gs::Cell >{}(place.start));
      common::hash_combine(seed, std::hash< int >{}(int(place.drow)));
      common::hash_combine(seed, std::hash< int >{}(int(place.dcol)));
      return seed;
   }
};

template <>
struct hash< battleship_gs::Fire > {
   size_t operator()(const battleship_gs::Fire& fire) const noexcept
   {
      size_t seed = 0;
      common::hash_combine(seed, std::hash< battleship_gs::Cell >{}(fire.target));
      return seed;
   }
};

template <>
struct hash< battleship_gs::Action > {
   size_t operator()(const battleship_gs::Action& action) const noexcept
   {
      size_t seed = 0;
      if(std::holds_alternative< battleship_gs::Place >(action)) {
         common::hash_combine(
            seed, std::hash< battleship_gs::Place >{}(std::get< battleship_gs::Place >(action))
         );
      } else {
         common::hash_combine(
            seed, std::hash< battleship_gs::Fire >{}(std::get< battleship_gs::Fire >(action))
         );
      }
      return seed;
   }
};

}  // namespace std

#endif  // NOR_BATTLESHIP_GS_UTILS_HPP
