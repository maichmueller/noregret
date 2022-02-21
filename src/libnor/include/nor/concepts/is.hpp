

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

template < typename T >
concept hashable = requires(T a)
{
   {
      std::hash< T >{}(a)
      } -> std::convertible_to< std::size_t >;
};

template < typename T >
concept enum_ = std::is_enum_v< T >;

template < class T, class... Ts >
concept any_of = ::std::disjunction< ::std::is_same< T, Ts >... >::value;

template < class T, class... Ts >
concept same_as_all = ::std::conjunction< ::std::is_same< T, Ts >... >::value;

template < typename T >
concept iterator = requires(T t)
{
   std::iterator_traits< T >::difference_type;
   std::iterator_traits< T >::pointer;
   std::iterator_traits< T >::reference;
   std::iterator_traits< T >::value_type;
   std::iterator_traits< T >::iterator_category;
};
}  // namespace nor::concepts::is

#endif  // NOR_IS_HPP
