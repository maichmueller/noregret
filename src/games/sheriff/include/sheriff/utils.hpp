
#ifndef NOR_SHERIFF_UTILS_HPP
#define NOR_SHERIFF_UTILS_HPP

#include <string>
#include <variant>

#include "common/common.hpp"
#include "sheriff/state.hpp"

namespace common {

template <>
inline std::string to_string(const sheriff::Load& load)
{
   return fmt::format("load{}", load.items);
}

template <>
inline std::string to_string(const sheriff::Offer& offer)
{
   return fmt::format("offer{}", offer.bribe);
}

template <>
inline std::string to_string(const sheriff::Respond& respond)
{
   return respond.accept ? "accept" : "reject";
}

template <>
inline std::string to_string(const sheriff::Action& action)
{
   if(std::holds_alternative< sheriff::Load >(action)) {
      return common::to_string(std::get< sheriff::Load >(action));
   }
   if(std::holds_alternative< sheriff::Offer >(action)) {
      return common::to_string(std::get< sheriff::Offer >(action));
   }
   return common::to_string(std::get< sheriff::Respond >(action));
}

template <>
inline std::string to_string(const sheriff::Phase& phase)
{
   switch(phase) {
      case sheriff::Phase::load: return "load";
      case sheriff::Phase::offer: return "offer";
      case sheriff::Phase::respond: return "respond";
      default: return "over";
   }
}

template <>
inline std::string to_string(const sheriff::Player& player)
{
   switch(player) {
      case sheriff::Player::one: return "smuggler";
      case sheriff::Player::two: return "sheriff";
      default: return "none";
   }
}

template <>
inline std::string to_string(const sheriff::Config& config)
{
   return fmt::format(
      "sheriff(v={},p={},s={},n_max={},b_max={},r={})",
      config.v,
      config.p,
      config.s,
      config.n_max,
      config.b_max,
      config.rounds
   );
}

}  // namespace common

COMMON_ENABLE_PRINT(sheriff, Load);
COMMON_ENABLE_PRINT(sheriff, Offer);
COMMON_ENABLE_PRINT(sheriff, Respond);
COMMON_ENABLE_PRINT(sheriff, Action);
COMMON_ENABLE_PRINT(sheriff, Phase);
COMMON_ENABLE_PRINT(sheriff, Player);

namespace std {

template <>
struct hash< sheriff::Load > {
   size_t operator()(const sheriff::Load& load) const noexcept
   {
      size_t seed = 0;
      common::hash_combine(seed, std::hash< unsigned >{}(31u + load.items));
      return seed;
   }
};

template <>
struct hash< sheriff::Offer > {
   size_t operator()(const sheriff::Offer& offer) const noexcept
   {
      size_t seed = 0;
      common::hash_combine(seed, std::hash< unsigned >{}(67u + offer.bribe));
      return seed;
   }
};

template <>
struct hash< sheriff::Respond > {
   size_t operator()(const sheriff::Respond& respond) const noexcept
   {
      size_t seed = 0;
      common::hash_combine(seed, std::hash< bool >{}(respond.accept));
      return seed;
   }
};

template <>
struct hash< sheriff::Action > {
   size_t operator()(const sheriff::Action& action) const noexcept
   {
      size_t seed = 0;
      if(std::holds_alternative< sheriff::Load >(action)) {
         common::hash_combine(
            seed, std::hash< sheriff::Load >{}(std::get< sheriff::Load >(action))
         );
      } else if(std::holds_alternative< sheriff::Offer >(action)) {
         common::hash_combine(
            seed, std::hash< sheriff::Offer >{}(std::get< sheriff::Offer >(action))
         );
      } else {
         common::hash_combine(
            seed, std::hash< sheriff::Respond >{}(std::get< sheriff::Respond >(action))
         );
      }
      return seed;
   }
};

}  // namespace std

#endif  // NOR_SHERIFF_UTILS_HPP
