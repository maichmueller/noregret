
#ifndef NOR_THREE_PLAYER_GOOFSPIEL_UTILS_HPP
#define NOR_THREE_PLAYER_GOOFSPIEL_UTILS_HPP

#include <cstdint>
#include <functional>
#include <string>

#include "common/common.hpp"
#include "state.hpp"

namespace three_player_goofspiel {

constexpr common::CEBijection< Player, std::string_view, 4 > player_name_bij = {
   std::pair{Player::chance, "chance"},
   std::pair{Player::alex, "alex"},
   std::pair{Player::bob, "bob"},
   std::pair{Player::cedric, "cedric"}};

constexpr common::CEBijection< Phase, std::string_view, 6 > phase_name_bij = {
   std::pair{Phase::deal, "deal"},
   std::pair{Phase::prize_reveal, "prize_reveal"},
   std::pair{Phase::commit_alex, "commit_alex"},
   std::pair{Phase::commit_bob, "commit_bob"},
   std::pair{Phase::commit_cedric, "commit_cedric"},
   std::pair{Phase::resolve, "resolve"}};

constexpr common::CEBijection< RoundWinner, std::string_view, 4 > outcome_name_bij = {
   std::pair{RoundWinner::alex, "alex"},
   std::pair{RoundWinner::bob, "bob"},
   std::pair{RoundWinner::cedric, "cedric"},
   std::pair{RoundWinner::tie, "tie"}};

}  // namespace three_player_goofspiel

namespace common {

template <>
inline std::string to_string(const three_player_goofspiel::Player& value)
{
   return std::string(three_player_goofspiel::player_name_bij.at(value));
}

template <>
inline std::string to_string(const three_player_goofspiel::Phase& value)
{
   return std::string(three_player_goofspiel::phase_name_bij.at(value));
}

template <>
inline std::string to_string(const three_player_goofspiel::RoundWinner& value)
{
   return std::string(three_player_goofspiel::outcome_name_bij.at(value));
}

template <>
inline std::string to_string(const three_player_goofspiel::Bid& value)
{
   return fmt::format("bid:{}", unsigned(value.card));
}

template <>
inline std::string to_string(const three_player_goofspiel::ChanceOutcome& value)
{
   using Kind = three_player_goofspiel::ChanceOutcome::Kind;
   switch(value.kind) {
      case Kind::deal: return fmt::format("deal:{:x}", unsigned(value.mask_a));
      case Kind::prize: return fmt::format("prize:{}", unsigned(value.value));
      case Kind::confirm: return std::string("resolve");
   }
   return std::string("?");
}

}  // namespace common

COMMON_ENABLE_PRINT(three_player_goofspiel, Player);
COMMON_ENABLE_PRINT(three_player_goofspiel, Phase);
COMMON_ENABLE_PRINT(three_player_goofspiel, RoundWinner);
COMMON_ENABLE_PRINT(three_player_goofspiel, Bid);
COMMON_ENABLE_PRINT(three_player_goofspiel, ChanceOutcome);

namespace std {

template <>
struct hash< three_player_goofspiel::Bid > {
   size_t operator()(const three_player_goofspiel::Bid& bid) const noexcept
   {
      size_t seed = 0;
      common::hash_combine(seed, std::hash< unsigned >{}(unsigned(bid.card)));
      return seed;
   }
};

template <>
struct hash< three_player_goofspiel::ChanceOutcome > {
   size_t operator()(const three_player_goofspiel::ChanceOutcome& outcome) const noexcept
   {
      size_t seed = 0;
      common::hash_combine(seed, std::hash< unsigned >{}(unsigned(outcome.kind)));
      common::hash_combine(seed, std::hash< unsigned >{}(unsigned(outcome.value)));
      common::hash_combine(seed, std::hash< unsigned >{}(unsigned(outcome.mask_a)));
      return seed;
   }
};

}  // namespace std

#endif  // NOR_THREE_PLAYER_GOOFSPIEL_UTILS_HPP
