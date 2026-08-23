
#ifndef NOR_DARK_HEX_UTILS_HPP
#define NOR_DARK_HEX_UTILS_HPP

#include <functional>
#include <string>

#include "common/common.hpp"
#include "state.hpp"

namespace common {

template <>
inline std::string to_string(const dark_hex::Player& player)
{
   switch(player) {
      case dark_hex::Player::one: return "one";
      case dark_hex::Player::two: return "two";
      default: return "none";
   }
}

template <>
inline std::string to_string(const dark_hex::RulesMode& mode)
{
   return mode == dark_hex::RulesMode::cdh ? "cdh" : "adh";
}

template <>
inline std::string to_string(const dark_hex::Move& move)
{
   return fmt::format("move{}", unsigned(move.cell_index));
}

template <>
inline std::string to_string(const dark_hex::Config& config)
{
   return fmt::format(
      "dark_hex({}x{}, {})",
      config.board_size,
      config.board_size,
      common::to_string(config.rules_mode)
   );
}

template <>
inline std::string to_string(const dark_hex::LogEntry& entry)
{
   return fmt::format("{}:{}", common::to_string(entry.actor), unsigned(entry.cell_index));
}

}  // namespace common

COMMON_ENABLE_PRINT(dark_hex, Player);
COMMON_ENABLE_PRINT(dark_hex, RulesMode);
COMMON_ENABLE_PRINT(dark_hex, Move);
COMMON_ENABLE_PRINT(dark_hex, Config);
COMMON_ENABLE_PRINT(dark_hex, LogEntry);

namespace std {

template <>
struct hash< dark_hex::Move > {
   size_t operator()(const dark_hex::Move& move) const noexcept
   {
      return std::hash< unsigned >{}(unsigned(move.cell_index));
   }
};

}  // namespace std

#endif  // NOR_DARK_HEX_UTILS_HPP
