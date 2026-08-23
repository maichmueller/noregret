
#ifndef NOR_GOOFSPIEL_UTILS_HPP
#define NOR_GOOFSPIEL_UTILS_HPP

#include <cstdint>
#include <functional>
#include <string>

#include "common/common.hpp"
#include "state.hpp"

namespace goofspiel {

constexpr common::CEBijection< Player, std::string_view, 3 > player_name_bij = {
   std::pair{Player::chance, "chance"},
   std::pair{Player::one, "one"},
   std::pair{Player::two, "two"}};

constexpr common::CEBijection< Phase, std::string_view, 4 > phase_name_bij = {
   std::pair{Phase::prize_reveal, "prize_reveal"},
   std::pair{Phase::commit_p1, "commit_p1"},
   std::pair{Phase::commit_p2, "commit_p2"},
   std::pair{Phase::resolve, "resolve"}};

constexpr common::CEBijection< RoundOutcome, std::string_view, 3 > outcome_name_bij = {
   std::pair{RoundOutcome::p1_wins, "p1"},
   std::pair{RoundOutcome::tie, "tie"},
   std::pair{RoundOutcome::p2_wins, "p2"}};

}  // namespace goofspiel

namespace common {

template <>
inline std::string to_string(const goofspiel::Player& value)
{
   return std::string(goofspiel::player_name_bij.at(value));
}

template <>
inline std::string to_string(const goofspiel::Phase& value)
{
   return std::string(goofspiel::phase_name_bij.at(value));
}

template <>
inline std::string to_string(const goofspiel::RoundOutcome& value)
{
   return std::string(goofspiel::outcome_name_bij.at(value));
}

template <>
inline std::string to_string(const goofspiel::Bid& value)
{
   return fmt::format("bid:{}", unsigned(value.card));
}

template <>
inline std::string to_string(const goofspiel::PrizeCard& value)
{
   if(value.value == 0) {
      return std::string("resolve");
   }
   return fmt::format("prize:{}", unsigned(value.value));
}

}  // namespace common

COMMON_ENABLE_PRINT(goofspiel, Player);
COMMON_ENABLE_PRINT(goofspiel, Phase);
COMMON_ENABLE_PRINT(goofspiel, RoundOutcome);
COMMON_ENABLE_PRINT(goofspiel, Bid);
COMMON_ENABLE_PRINT(goofspiel, PrizeCard);

namespace std {

template <>
struct hash< goofspiel::Bid > {
   size_t operator()(const goofspiel::Bid& bid) const noexcept
   {
      size_t seed = 0;
      common::hash_combine(seed, std::hash< unsigned >{}(unsigned(bid.card)));
      return seed;
   }
};

template <>
struct hash< goofspiel::PrizeCard > {
   size_t operator()(const goofspiel::PrizeCard& outcome) const noexcept
   {
      size_t seed = 0;
      common::hash_combine(seed, std::hash< unsigned >{}(unsigned(outcome.value)));
      return seed;
   }
};

}  // namespace std

#endif  // NOR_GOOFSPIEL_UTILS_HPP
