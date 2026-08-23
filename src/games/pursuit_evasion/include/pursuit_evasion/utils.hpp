
#ifndef NOR_PURSUIT_EVASION_UTILS_HPP
#define NOR_PURSUIT_EVASION_UTILS_HPP

#include <functional>
#include <string>
#include <variant>

#include "common/common.hpp"
#include "state.hpp"

namespace common {

template <>
inline std::string to_string(const pursuit_evasion::Player& player)
{
   switch(player) {
      case pursuit_evasion::Player::one: return "attacker";
      case pursuit_evasion::Player::two: return "defender";
      default: return "none";
   }
}

template <>
inline std::string to_string(const pursuit_evasion::Phase& phase)
{
   return phase == pursuit_evasion::Phase::commit_attacker ? "commit_attacker" : "commit_defender";
}

template <>
inline std::string to_string(const pursuit_evasion::TerminalCause& cause)
{
   using Cause = pursuit_evasion::TerminalCause;
   switch(cause) {
      case Cause::capture: return "capture";
      case Cause::escape: return "escape";
      case Cause::timeout: return "timeout";
      default: return "none";
   }
}

template <>
inline std::string to_string(const pursuit_evasion::AttMove& move)
{
   if(move.is_wait()) {
      return "wait";
   }
   const auto& edge = pursuit_evasion::k_attacker_edges[move.edge_id];
   return fmt::format("{}->{}", unsigned(edge.from), unsigned(edge.to));
}

template <>
inline std::string to_string(const pursuit_evasion::DefMove& move)
{
   return fmt::format("patrols({},{})", unsigned(move.p1), unsigned(move.p2));
}

template <>
inline std::string to_string(const pursuit_evasion::Action& action)
{
   if(std::holds_alternative< pursuit_evasion::AttMove >(action)) {
      return common::to_string(std::get< pursuit_evasion::AttMove >(action));
   }
   return common::to_string(std::get< pursuit_evasion::DefMove >(action));
}

template <>
inline std::string to_string(const pursuit_evasion::Config& config)
{
   return fmt::format("pursuit_evasion(m={})", config.rounds);
}

template <>
inline std::string to_string(const pursuit_evasion::RoundRecord& record)
{
   return fmt::format(
      "round(att={}, p1={}, p2={}, sight1={}, sight2={}, cause={})",
      record.att_edge,
      unsigned(record.p1_to),
      unsigned(record.p2_to),
      record.sight1,
      record.sight2,
      common::to_string(record.cause)
   );
}

}  // namespace common

COMMON_ENABLE_PRINT(pursuit_evasion, Player);
COMMON_ENABLE_PRINT(pursuit_evasion, Phase);
COMMON_ENABLE_PRINT(pursuit_evasion, TerminalCause);
COMMON_ENABLE_PRINT(pursuit_evasion, AttMove);
COMMON_ENABLE_PRINT(pursuit_evasion, DefMove);
COMMON_ENABLE_PRINT(pursuit_evasion, Action);
COMMON_ENABLE_PRINT(pursuit_evasion, Config);
COMMON_ENABLE_PRINT(pursuit_evasion, RoundRecord);

namespace std {

// NOTE: these hashes must be visible before the FOSG adapter classes (ordering pitfall)

template <>
struct hash< pursuit_evasion::AttMove > {
   size_t operator()(const pursuit_evasion::AttMove& move) const noexcept
   {
      return std::hash< unsigned >{}(unsigned(move.edge_id));
   }
};

template <>
struct hash< pursuit_evasion::DefMove > {
   size_t operator()(const pursuit_evasion::DefMove& move) const noexcept
   {
      size_t seed = 0;
      common::hash_combine(seed, std::hash< unsigned >{}(unsigned(move.p1)));
      common::hash_combine(seed, std::hash< unsigned >{}(17u + unsigned(move.p2)));
      return seed;
   }
};

template <>
struct hash< pursuit_evasion::Action > {
   size_t operator()(const pursuit_evasion::Action& action) const noexcept
   {
      size_t seed = 0;
      if(std::holds_alternative< pursuit_evasion::AttMove >(action)) {
         common::hash_combine(
            seed,
            std::hash< pursuit_evasion::AttMove >{}(std::get< pursuit_evasion::AttMove >(action))
         );
      } else {
         common::hash_combine(seed, 7331u);
         common::hash_combine(
            seed,
            std::hash< pursuit_evasion::DefMove >{}(std::get< pursuit_evasion::DefMove >(action))
         );
      }
      return seed;
   }
};

}  // namespace std

#endif  // NOR_PURSUIT_EVASION_UTILS_HPP
