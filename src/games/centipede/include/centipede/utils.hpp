
#ifndef NOR_CENTIPEDE_UTILS_HPP
#define NOR_CENTIPEDE_UTILS_HPP

#include <functional>
#include <string>

#include "centipede/state.hpp"
#include "common/common.hpp"

namespace common {

template <>
inline std::string to_string(const centipede::Player& player)
{
   switch(player) {
      case centipede::Player::one: return "one";
      case centipede::Player::two: return "two";
      default: return "none";
   }
}

template <>
inline std::string to_string(const centipede::TerminalCause& cause)
{
   switch(cause) {
      case centipede::TerminalCause::taken: return "taken";
      case centipede::TerminalCause::exhausted: return "exhausted";
      default: return "none";
   }
}

template <>
inline std::string to_string(const centipede::Move& move)
{
   return move.take ? "take" : "push";
}

template <>
inline std::string to_string(const centipede::Config& config)
{
   return fmt::format(
      "centipede(N={}, m0={}, m1={})", config.rounds, config.pile_big, config.pile_small
   );
}

}  // namespace common

COMMON_ENABLE_PRINT(centipede, Player);
COMMON_ENABLE_PRINT(centipede, TerminalCause);
COMMON_ENABLE_PRINT(centipede, Move);
COMMON_ENABLE_PRINT(centipede, Config);

namespace std {

// NOTE: these hashes must be visible before the FOSG adapter classes (ordering pitfall)

template <>
struct hash< centipede::Move > {
   size_t operator()(const centipede::Move& move) const noexcept
   {
      return std::hash< bool >{}(move.take);
   }
};

}  // namespace std

#endif  // NOR_CENTIPEDE_UTILS_HPP
