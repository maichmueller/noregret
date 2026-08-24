
#ifndef NOR_SHAPLEY_UTILS_HPP
#define NOR_SHAPLEY_UTILS_HPP

#include <functional>
#include <string>

#include "common/common.hpp"
#include "shapley/state.hpp"

namespace common {

template <>
inline std::string to_string(const shapley::Player& player)
{
   switch(player) {
      case shapley::Player::one: return "one";
      case shapley::Player::two: return "two";
      default: return "none";
   }
}

template <>
inline std::string to_string(const shapley::Phase& phase)
{
   return phase == shapley::Phase::commit_p1 ? "commit_p1" : "commit_p2";
}

template <>
inline std::string to_string(const shapley::Play& play)
{
   static constexpr std::array< const char*, 3 > k_names{{"top", "middle", "bottom"}};
   return fmt::format("play:{}", k_names.at(play.strategy));
}

}  // namespace common

COMMON_ENABLE_PRINT(shapley, Player);
COMMON_ENABLE_PRINT(shapley, Phase);
COMMON_ENABLE_PRINT(shapley, Play);

namespace std {

// NOTE: these hashes must be visible before the FOSG adapter classes (ordering pitfall)

template <>
struct hash< shapley::Play > {
   size_t operator()(const shapley::Play& play) const noexcept
   {
      return std::hash< unsigned >{}(play.strategy);
   }
};

}  // namespace std

#endif  // NOR_SHAPLEY_UTILS_HPP
