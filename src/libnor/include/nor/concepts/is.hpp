

#ifndef NOR_IS_HPP
#define NOR_IS_HPP

#include <concepts>
#include <cstddef>
#include <variant>

namespace nor::concepts::is {

template < typename T >
concept sized = requires(T const t) {
   {
      t.size()
   } -> std::same_as< size_t >;
};

template < typename Iter >
concept const_iter = std::is_const_v<
   std::remove_reference_t< typename std::iterator_traits< Iter >::reference > >;

template < typename T >
concept hashable = requires(T t) {
   {
      std::hash< T >{}(t)
   } -> std::convertible_to< std::size_t >;
};

template < typename T >
constexpr bool variant(const T&)
{
   return false;
}

template < typename... Types >
   requires(sizeof...(Types) > 1)
constexpr bool variant(const std::variant< Types... >&)
{
   return true;
}

template < typename T >
concept enum_ = std::is_enum_v< T >;

template < class T, class... Ts >
concept any_of = ::std::disjunction_v< ::std::is_same< T, Ts >... >;

template < class T, class... Ts >
concept same_as_all = ::std::conjunction_v< ::std::is_same< T, Ts >... >;

template < typename T >
concept dereferencable = requires(T t) { *t; };

template < typename T >
concept iterator = requires(T t) {
   std::iterator_traits< T >::difference_type;
   std::iterator_traits< T >::pointer;
   std::iterator_traits< T >::reference;
   std::iterator_traits< T >::value_type;
   std::iterator_traits< T >::iterator_category;
};

template < typename T >
concept empty = std::is_empty_v< T >;

template < typename T >
concept not_empty = not std::is_empty_v< T >;

template < typename T, typename Output >
concept dynamic_pointer_castable_to = requires(T t) { std::dynamic_pointer_cast< Output >(t); };

template < typename T >
concept smart_pointer_like = requires(T t) {
   // this would be true if T was a raw pointer (hence we cannot allow
   // this to be true)
   requires not std::is_pointer_v< T >;
   // needs to be convertible to bool (nullptr or not nullptr)
   bool(t);
   // needs to have the underlying type stored in it as typedef
   typename T::element_type;
   // needs to be dereferencable
   {
      t.operator*()
   } -> std::same_as< typename T::element_type& >;
   // needs to allow for arrow operator calls
   t.operator->();
};

template < typename T >
concept pointer = std::is_pointer_v< T >;

template < typename T, template < class... > class Template >
concept specialization = common::is_specialization_v< T, Template >;

template < typename T >
// clang-format off
concept copyable_someway =
   (
      std::is_pointer_v< T >
      and concepts::has::method::clone< std::remove_pointer_t<T> >
   )
   or concepts::has::method::clone< T >
   or concepts::has::method::copy< T >
   or std::is_copy_constructible_v< T >;
// clang-format on

template < typename Env >
concept serialized = requires(Env e) { requires Env::serialized(); };

template < typename Env >
concept unrolled = requires(Env e) { requires Env::unrolled(); };

template < typename Env >
concept samples_chance = requires(Env e) {
   requires Env::stochasticity() == Stochasticity::sample;
};

template < typename Env >
concept enumerates_chance = requires(Env e) {
   requires Env::stochasticity() == Stochasticity::choice;
};

template < typename Env >
concept deterministic = requires(Env e) {
   requires Env::stochasticity() == Stochasticity::deterministic;
};

}  // namespace nor::concepts::is

#endif  // NOR_IS_HPP
