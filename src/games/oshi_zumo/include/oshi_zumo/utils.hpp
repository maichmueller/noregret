
#ifndef NOR_OSHI_ZUMO_UTILS_HPP
#define NOR_OSHI_ZUMO_UTILS_HPP

#include <functional>
#include <string>

#include "common/common.hpp"
#include "oshi_zumo/state.hpp"

namespace common {

template <>
inline std::string to_string(const oshi_zumo::Player& player)
{
   switch(player) {
      case oshi_zumo::Player::one: return "one";
      case oshi_zumo::Player::two: return "two";
      default: return "none";
   }
}

template <>
inline std::string to_string(const oshi_zumo::Phase& phase)
{
   return phase == oshi_zumo::Phase::commit_p1 ? "commit_p1" : "commit_p2";
}

template <>
inline std::string to_string(const oshi_zumo::TerminalCause& cause)
{
   using Cause = oshi_zumo::TerminalCause;
   switch(cause) {
      case Cause::edge_arrival: return "edge_arrival";
      case Cause::horizon: return "horizon";
      case Cause::both_broke: return "both_broke";
      default: return "none";
   }
}

template <>
inline std::string to_string(const oshi_zumo::Bid& bid)
{
   return fmt::format("bid:{}", bid.amount);
}

template <>
inline std::string to_string(const oshi_zumo::RoundRecord& record)
{
   return fmt::format(
      "round(one={}, two={}, pos={}, paid=({},{}), cause={})",
      record.bid_one,
      record.bid_two,
      record.wrestler_pos_after,
      record.one_paid,
      record.two_paid,
      common::to_string(record.cause)
   );
}

template <>
inline std::string to_string(const oshi_zumo::Config& config)
{
   return fmt::format(
      "oshi_zumo(size={}, coins={}, min_bid={}, horizon={})",
      config.size,
      config.coins,
      config.min_bid,
      config.horizon
   );
}

}  // namespace common

COMMON_ENABLE_PRINT(oshi_zumo, Player);
COMMON_ENABLE_PRINT(oshi_zumo, Phase);
COMMON_ENABLE_PRINT(oshi_zumo, TerminalCause);
COMMON_ENABLE_PRINT(oshi_zumo, Bid);
COMMON_ENABLE_PRINT(oshi_zumo, RoundRecord);
COMMON_ENABLE_PRINT(oshi_zumo, Config);

namespace std {

// NOTE: these hashes must be visible before the FOSG adapter classes (ordering pitfall)

template <>
struct hash< oshi_zumo::Bid > {
   size_t operator()(const oshi_zumo::Bid& bid) const noexcept
   {
      return std::hash< unsigned >{}(bid.amount);
   }
};

}  // namespace std

#endif  // NOR_OSHI_ZUMO_UTILS_HPP
