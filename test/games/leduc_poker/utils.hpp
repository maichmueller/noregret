

#ifndef NOR_UTILS_HPP
#define NOR_UTILS_HPP

#include <range/v3/all.hpp>

template < typename Range1, typename Range2 >
bool cmp_equal_rngs(Range1 rng1, Range2 rng2)
{
   return ranges::all_of(ranges::views::zip(rng1, rng2), [](const auto& v_w) {
      const auto& [v, w] = v_w;
      return v == w;
   });
}

template < typename Range1, typename Range2 >
bool cmp_equal_rngs_unsorted(Range1 rng1, Range2 rng2)
{
   return ranges::all_of(rng1, [&](const auto& v) { return ranges::contains(rng2, v); })
          and ranges::all_of(rng2, [&](const auto& v) { return ranges::contains(rng1, v); });
}

template < typename Range1, typename Range2, typename Sorter1, typename Sorter2 >
bool cmp_equal_rngs(Range1 rng1, Range2 rng2, Sorter1 sorter1, Sorter2 sorter2)
{
   std::sort(rng1.begin(), rng1.end(), sorter1);
   std::sort(rng2.begin(), rng2.end(), sorter2);
   return ranges::all_of(ranges::views::zip(rng1, rng2), [](const auto& v_w) {
      const auto& [v, w] = v_w;
      return v == w;
   });
}

#endif  // NOR_UTILS_HPP
