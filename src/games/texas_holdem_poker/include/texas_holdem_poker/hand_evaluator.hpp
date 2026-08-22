
#ifndef NOR_TEXAS_HOLDEM_POKER_HAND_EVALUATOR_HPP
#define NOR_TEXAS_HOLDEM_POKER_HAND_EVALUATOR_HPP

#include <algorithm>
#include <array>
#include <cassert>
#include <cstddef>
#include <cstdint>

#include "texas_holdem_poker/state.hpp"

namespace texholdem::eval {

/// hand categories ordered by strength
enum class HandCategory : uint32_t {
   high_card = 0,
   one_pair = 1,
   two_pair = 2,
   three_of_a_kind = 3,
   straight = 4,
   flush = 5,
   full_house = 6,
   four_of_a_kind = 7,
   straight_flush = 8
};

/**
 * @brief Evaluates the best 5-card poker hand out of the given (5 to 8) cards.
 *
 * Returns a single comparable score: higher scores denote strictly better hands. The score
 * encodes the hand category together with its tie-breaking kickers (base-15 packing over
 * exactly five digit slots), so equal scores denote an exact tie (relevant for split pots).
 */
inline uint32_t evaluate(const Card* cards, size_t count)
{
   constexpr size_t min_cards = 5;
   constexpr size_t max_cards = 8;
   assert(count >= min_cards && count <= max_cards);

   std::array< uint32_t, 15 > rank_count{};
   std::array< uint32_t, 4 > suit_count{};
   std::array< uint32_t, 4 > suit_mask{};
   uint32_t full_mask = 0;
   for(size_t i = 0; i < count; ++i) {
      uint32_t rank = as_int(cards[i].rank);
      uint32_t suit = as_int(cards[i].suit);
      ++rank_count[rank];
      ++suit_count[suit];
      suit_mask[suit] |= (1u << rank);
      full_mask |= (1u << rank);
   }

   auto pack = [](HandCategory category, const std::array< uint32_t, 5 >& digits) {
      uint32_t score = static_cast< uint32_t >(category);
      for(uint32_t digit : digits) {
         score = score * 15u + digit;
      }
      return score;
   };
   /// top of the best straight within the given rank mask (0 if none); the ace plays low for the
   /// wheel A-2-3-4-5 which yields a '5-high' straight
   auto straight_top = [](uint32_t rank_mask) -> uint32_t {
      uint32_t mask = rank_mask | (((rank_mask >> 14u) & 1u) << 1u);
      for(int top = 14; top >= 6; --top) {
         uint32_t window = ((1u << 5) - 1u) << static_cast< uint32_t >(top - 4);
         if((mask & window) == window) {
            return static_cast< uint32_t >(top);
         }
      }
      constexpr uint32_t wheel = 0b111110u;  // ranks 1..5
      if((mask & wheel) == wheel) {
         return 5;
      }
      return 0;
   };

   // 1) (royal) straight flush -- note that a plain flush is handled below since a full house
   //    beats it and both may be present among <= 7 cards
   for(size_t suit = 0; suit < 4; ++suit) {
      if(suit_count[suit] >= min_cards) {
         if(uint32_t top = straight_top(suit_mask[suit]); top != 0) {
            return pack(HandCategory::straight_flush, {top, 0, 0, 0, 0});
         }
      }
   }

   // group the ranks by multiplicity in descending order (count desc, then rank desc)
   std::array< std::pair< uint32_t, uint32_t >, max_cards > groups{};  //< (count, rank)
   size_t n_groups = 0;
   for(int rank = 14; rank >= 2; --rank) {
      if(rank_count[rank] > 0) {
         groups[n_groups++] = {rank_count[rank], static_cast< uint32_t >(rank)};
      }
   }
   std::stable_sort(
      groups.begin(),
      std::next(groups.begin(), std::ptrdiff_t(n_groups)),
      [](const auto& left, const auto& right) { return left.first > right.first; }
   );
   const auto count_of = [&](size_t i) { return i < n_groups ? groups[i].first : 0u; };
   const auto rank_of = [&](size_t i) { return i < n_groups ? groups[i].second : 0u; };

   // 2) four of a kind
   if(count_of(0) == 4) {
      return pack(HandCategory::four_of_a_kind, {rank_of(0), rank_of(1), 0, 0, 0});
   }
   // 3) full house
   if(count_of(0) == 3 and count_of(1) >= 2) {
      return pack(HandCategory::full_house, {rank_of(0), rank_of(1), 0, 0, 0});
   }
   // 4) flush
   for(size_t suit = 0; suit < 4; ++suit) {
      if(suit_count[suit] >= min_cards) {
         std::array< uint32_t, 5 > digits{0, 0, 0, 0, 0};
         size_t used = 0;
         for(int rank = 14; rank >= 2 && used < digits.size(); --rank) {
            if(suit_mask[suit] & (1u << rank)) {
               digits[used++] = static_cast< uint32_t >(rank);
            }
         }
         return pack(HandCategory::flush, digits);
      }
   }
   // 5) straight
   if(uint32_t top = straight_top(full_mask); top != 0) {
      return pack(HandCategory::straight, {top, 0, 0, 0, 0});
   }
   // 6) three of a kind
   if(count_of(0) == 3) {
      return pack(HandCategory::three_of_a_kind, {rank_of(0), rank_of(1), rank_of(2), 0, 0});
   }
   // 7) two pair
   if(count_of(0) == 2 and count_of(1) == 2) {
      return pack(HandCategory::two_pair, {rank_of(0), rank_of(1), rank_of(2), 0, 0});
   }
   // 8) one pair
   if(count_of(0) == 2) {
      return pack(HandCategory::one_pair, {rank_of(0), rank_of(1), rank_of(2), rank_of(3), 0});
   }
   // 9) high card
   return pack(
      HandCategory::high_card, {rank_of(0), rank_of(1), rank_of(2), rank_of(3), rank_of(4)}
   );
}

}  // namespace texholdem::eval

#endif  // NOR_TEXAS_HOLDEM_POKER_HAND_EVALUATOR_HPP
