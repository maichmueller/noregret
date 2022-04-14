#pragma once

#include <execution>
#include <range/v3/all.hpp>
#include <stratego/stratego.hpp>


template < typename Range1, typename Range2 >
bool cmp_equal_rngs(Range1&& rng1, Range2&& rng2)
{
   return ranges::all_of(ranges::views::zip(rng1, rng2), [](const auto& v_w) {
      const auto& [v, w] = v_w;
      return v == w;
   });
}

template < typename Range1, typename Range2, typename Sorter1, typename Sorter2 >
bool cmp_equal_rngs(Range1 rng1, Range2 rng2, Sorter1 sorter1, Sorter2 sorter2)
{
   std::sort(std::execution::par, rng1.begin(), rng1.end(), sorter1);
   std::sort(std::execution::par, rng2.begin(), rng2.end(), sorter2);
   return ranges::all_of(ranges::views::zip(rng1, rng2), [](const auto& v_w) {
      const auto& [v, w] = v_w;
      return v == w;
   });
}

template < typename Range1, typename Range2 >
bool cmp_equal_rngs_sorted(Range1&& rng1, Range2&& rng2)
{
   return ranges::all_of(ranges::views::zip(rng1, rng2), [](const auto& v_w) {
      const auto& [v, w] = v_w;
      return v == w;
   });
}

struct flattable_sorter {
   template < typename T, typename U >
   requires requires(T t) { {t.flatten()}; } && requires(U u)
   {
      {u.flatten()};
   }
   auto operator()(T flattable1, U flattable2)
   {
      std::array< int, 2 > reduces{0, 0};
      std::array both{flattable1.flatten(), flattable2.flatten()};
      for(unsigned int i = 0; i <= 2; ++i) {
         int factor = 1;
         for(auto val = both[i].rbegin(); val != both[i].rend(); val++) {
            reduces[i] += *val * factor;
            factor *= 10;
         }
      }
      return reduces[0] <= reduces[1];
   }
};

template < typename T, typename SortF = flattable_sorter >
struct sorted {
   T value;
   explicit sorted(T val, SortF sort = SortF()) : value(std::move(val))
   {
      std::sort(value.begin(), value.end(), sort);
   }
   auto begin() const { return value.begin(); }
   auto end() const { return value.end(); }
};
template < typename T, typename SortF = flattable_sorter >
struct eq_rng {
   sorted< T > sorted_rng;
   explicit eq_rng(T val, SortF sort = SortF()) : sorted_rng(val, sort) {}
   bool operator==(const eq_rng& other) const { return cmp_equal_rngs(value(), other.value()); };

   auto begin() const { return sorted_rng.begin(); }
   auto end() const { return sorted_rng.end(); }

   const auto& value() const { return sorted_rng.value; }
   friend auto& operator<<(std::ostream& os, const eq_rng& rng)
   {
      os << aze::utils::SpanPrinter{std::span{rng.value()}};
      return os;
   }
};

