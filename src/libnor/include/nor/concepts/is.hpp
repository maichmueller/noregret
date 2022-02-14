

#ifndef NOR_IS_HPP
#define NOR_IS_HPP

#include <concepts>
#include <cstddef>

namespace nor::concepts::is {

template < typename T >
concept sized = requires(T const t)
{
   {
      t.size()
      } -> std::same_as< size_t >;
};

template < typename Iter >
concept const_iter = std::is_const_v<
   std::remove_reference_t< typename std::iterator_traits< Iter >::reference > >;


template <typename T, typename Trait>
concept trait = requires(T t) {
   T::Trait;
};



}  // namespace nor::concepts::is

#endif  // NOR_IS_HPP
