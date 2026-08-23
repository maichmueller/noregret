
#ifndef NOR_LIARS_DICE_UTILS_HPP
#define NOR_LIARS_DICE_UTILS_HPP

#include <functional>
#include <string>

#include "common/common.hpp"
#include "state.hpp"

namespace liars_dice {

constexpr common::CEBijection< Player, std::string_view, 3 > player_name_bij = {
   std::pair{Player::chance, "chance"},
   std::pair{Player::one, "one"},
   std::pair{Player::two, "two"}};

}  // namespace liars_dice

namespace common {

template <>
inline std::string to_string(const liars_dice::Player& value)
{
   return std::string(liars_dice::player_name_bij.at(value));
}

template <>
inline std::string to_string(const liars_dice::Roll& value)
{
   return fmt::format("{}:{}", common::to_string(value.player), int(value.face));
}

template <>
inline std::string to_string(const liars_dice::Bid& value)
{
   return fmt::format("{}x{}", int(value.count), int(value.face));
}

template <>
inline std::string to_string(const liars_dice::Action& value)
{
   if(value.kind == liars_dice::ActionType::challenge) {
      return "challenge";
   }
   return fmt::format("bid {}", common::to_string(value.bid));
}

}  // namespace common

COMMON_ENABLE_PRINT(liars_dice, Player);
COMMON_ENABLE_PRINT(liars_dice, Roll);
COMMON_ENABLE_PRINT(liars_dice, Bid);
COMMON_ENABLE_PRINT(liars_dice, Action);

namespace std {

template <>
struct hash< liars_dice::Roll > {
   size_t operator()(const liars_dice::Roll& roll) const noexcept
   {
      size_t seed = 0;
      common::hash_combine(seed, std::hash< int >{}(int(roll.player)));
      common::hash_combine(seed, std::hash< int >{}(int(roll.face)));
      return seed;
   }
};

template <>
struct hash< liars_dice::Bid > {
   size_t operator()(const liars_dice::Bid& bid) const noexcept
   {
      size_t seed = 0;
      common::hash_combine(seed, std::hash< int >{}(int(bid.count)));
      common::hash_combine(seed, std::hash< int >{}(int(bid.face)));
      return seed;
   }
};

template <>
struct hash< liars_dice::Action > {
   size_t operator()(const liars_dice::Action& action) const noexcept
   {
      size_t seed = 0;
      common::hash_combine(seed, std::hash< unsigned >{}(unsigned(action.kind)));
      if(action.kind == liars_dice::ActionType::bid) {
         common::hash_combine(seed, std::hash< liars_dice::Bid >{}(action.bid));
      }
      return seed;
   }
};

}  // namespace std

#endif  // NOR_LIARS_DICE_UTILS_HPP
