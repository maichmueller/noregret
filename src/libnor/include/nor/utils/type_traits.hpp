
#ifndef NOR_TYPE_TRAITS_HPP
#define NOR_TYPE_TRAITS_HPP

namespace nor {

template < typename T, typename U >
struct const_as {
   using type = std::conditional_t< std::is_const_v< T >, std::add_const_t< U >, U >;
};

template < typename T, typename U >
using const_as_t = typename const_as<T, U>::type;

template < typename T, typename U >
struct ref_as {
   using type = std::conditional_t<
      std::is_lvalue_reference_v< T >,
      std::add_lvalue_reference_t< U >,
      std::conditional_t<
         std::is_rvalue_reference_v< T >,
         std::add_rvalue_reference_t< U >,
         U > >;
};

template < typename T, typename U >
using ref_as_t = typename ref_as<T, U>::type;

}  // namespace nor
#endif  // NOR_TYPE_TRAITS_HPP
