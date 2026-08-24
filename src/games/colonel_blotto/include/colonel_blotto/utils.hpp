
#ifndef NOR_COLONEL_BLOTTO_UTILS_HPP
#define NOR_COLONEL_BLOTTO_UTILS_HPP

#include <functional>
#include <string>

#include "colonel_blotto/state.hpp"
#include "common/common.hpp"

namespace common {

template <>
inline std::string to_string(const colonel_blotto::Player& player)
{
   switch(player) {
      case colonel_blotto::Player::one: return "one";
      case colonel_blotto::Player::two: return "two";
      default: return "none";
   }
}

template <>
inline std::string to_string(const colonel_blotto::Phase& phase)
{
   return phase == colonel_blotto::Phase::commit_p1 ? "commit_p1" : "commit_p2";
}

template <>
inline std::string to_string(const colonel_blotto::FieldOutcome& outcome)
{
   switch(outcome) {
      case colonel_blotto::FieldOutcome::one_wins: return "one_wins";
      case colonel_blotto::FieldOutcome::split: return "split";
      case colonel_blotto::FieldOutcome::two_wins: return "two_wins";
      default: return "none";
   }
}

template <>
inline std::string to_string(const colonel_blotto::TerminalCause& cause)
{
   return cause == colonel_blotto::TerminalCause::resolved ? "resolved" : "none";
}

template <>
inline std::string to_string(const colonel_blotto::Deploy& deploy)
{
   return fmt::format("deploy:{}", deploy.troops);
}

template <>
inline std::string to_string(const colonel_blotto::BlottoConfig& config)
{
   return fmt::format("colonel_blotto(B={})", config.budget);
}

}  // namespace common

COMMON_ENABLE_PRINT(colonel_blotto, Player);
COMMON_ENABLE_PRINT(colonel_blotto, Phase);
COMMON_ENABLE_PRINT(colonel_blotto, FieldOutcome);
COMMON_ENABLE_PRINT(colonel_blotto, TerminalCause);
COMMON_ENABLE_PRINT(colonel_blotto, Deploy);
COMMON_ENABLE_PRINT(colonel_blotto, BlottoConfig);

namespace std {

// NOTE: these hashes must be visible before the FOSG adapter classes (ordering pitfall)

template <>
struct hash< colonel_blotto::Deploy > {
   size_t operator()(const colonel_blotto::Deploy& deploy) const noexcept
   {
      return std::hash< unsigned >{}(deploy.troops);
   }
};

}  // namespace std

#endif  // NOR_COLONEL_BLOTTO_UTILS_HPP
