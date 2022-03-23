
#ifndef NOR_TYPE_TRAITS_HPP
#define NOR_TYPE_TRAITS_HPP

#include <concepts>

namespace nor {
//
// template < typename T, typename U >
// struct add_const_if {
//   using type = U;
//};
// template < typename T, typename U >
// requires(not std::is_pointer_v< T >) struct add_const_if< T *const, U > {
//   using type = U;
//};
// template < typename T, typename U >
// requires(std::is_same_v< T, std::remove_cvref_t< T > >) struct add_const_if< T const, U > {
//   using type = const U;
//};
// template < typename T, typename U >
// struct add_const_if< T const *const, U > {
//   using type = const U;
//};
// template < typename T, typename U >
// struct add_const_if< T const *, U > {
//   using type = U const;
//};
// template < typename T, typename U >
// struct add_const_if< T const *, U & > {
//   using type = U const &;
//};
// template < typename T, typename U >
// struct add_const_if< T const *, U && > {
//   using type = U const &&;
//};
// template < typename T, typename U >
// struct add_const_if< T const *, U * > {
//   using type = U const *;
//};
// template < typename T, typename U >
// struct add_const_if< T const &, U > {
//   using type = U const;
//};
// template < typename T, typename U >
// struct add_const_if< T const &, U & > {
//   using type = U const &;
//};
// template < typename T, typename U >
// struct add_const_if< T const &, U && > {
//   using type = U const &&;
//};
// template < typename T, typename U >
// struct add_const_if< T const &, U * > {
//   using type = U const *;
//};
// template < typename T, typename U >
// struct add_const_if< T const &&, U > {
//   using type = U const;
//};
// template < typename T, typename U >
// struct add_const_if< T const &&, U & > {
//   using type = U const &;
//};
// template < typename T, typename U >
// struct add_const_if< T const &&, U && > {
//   using type = U const &&;
//};
// template < typename T, typename U >
// struct add_const_if< T const &&, U * > {
//   using type = U const *;
//};
// template < typename T, typename U >
// struct add_const_if< T const[], U > {
//   using type = U const;
//};
// template < typename T, typename U >
// struct add_const_if< T const[], U & > {
//   using type = U const &;
//};
// template < typename T, typename U >
// struct add_const_if< T const[], U && > {
//   using type = U const &&;
//};
// template < typename T, typename U >
// struct add_const_if< T const[], U * > {
//   using type = U const *;
//};
// template < typename T, typename U >
// using add_const_if_t = typename add_const_if< T, U >::type;
//
// template < typename T, typename U >
// struct add_reference_if {
//   using type = std::conditional_t<
//      std::is_lvalue_reference_v< std::remove_const_t< T > >,
//      std::add_lvalue_reference_t< U >,
//      std::conditional_t<
//         std::is_rvalue_reference_v< std::remove_const_t< T > >,
//         std::add_rvalue_reference_t< U >,
//         U > >;
//};
//
// template < typename T, typename U >
// using add_reference_if_t = typename add_reference_if< T, U >::type;
//
// template < typename T, typename U >
// struct remove_const_if {
//   using type = std::conditional_t<
//      not std::is_const_v< std::remove_reference_t< T > >,
//      std::remove_const_t< U >,
//      U >;
//};
//
// template < typename T, typename U >
// using remove_const_if_t = typename remove_const_if< T, U >::type;
//
// template < typename T, typename U >
// struct remove_reference_if {
//   using type = std::conditional_t<
//      not std::is_reference_v< std::remove_const_t< T > >,
//      std::remove_reference_t< U >,
//      U >;
//};
//
// template < typename T, typename U >
// using remove_reference_if_t = typename remove_reference_if< T, U >::type;

template < typename T, typename U >
struct const_as {
   using type = std::conditional_t< std::is_const_v< T >, std::add_const_t< U >, U >;
};

template < typename T, typename U >
using const_as_t = typename const_as< T, U >::type;

template < typename T, typename U >
struct reference_as {
   using type = std::conditional_t<
      std::is_lvalue_reference_v< T >,
      std::add_lvalue_reference_t< U >,
      std::conditional_t< std::is_rvalue_reference_v< T >, std::add_rvalue_reference_t< U >, U > >;
};

template < typename T, typename U >
using reference_as_t = typename reference_as< T, U >::type;

/**
 * @brief type_trait to check if functior F can be invoked with each arg
 * @tparam F
 * @tparam Ret
 * @tparam HeadArg
 * @tparam TailArg
 */
template < typename F, typename Ret, typename HeadArg, typename... TailArg >
struct invocable_with_each {
   inline constexpr static bool value = invocable_with_each< F, Ret, HeadArg >::value
                                        && invocable_with_each< F, Ret, TailArg... >::value;
};

template < typename F, typename Ret, typename HeadArg, typename TailArg >
struct invocable_with_each< F, Ret, HeadArg, TailArg > {
   inline constexpr static bool value = invocable_with_each< F, Ret, HeadArg >::value
                                        && invocable_with_each< F, Ret, TailArg >::value;
};

template < typename F, typename Ret, typename Arg >
struct invocable_with_each< F, Ret, Arg > {
  private:
   inline constexpr static bool is_invocable = std::invocable< F, Arg >;

   constexpr static bool matches_return_type()
   {
      if constexpr(is_invocable) {
         return std::is_same_v< std::invoke_result_t< F, Arg >, Ret >;
      } else {
         return false;
      }
   }

  public:
   inline constexpr static bool value = is_invocable && matches_return_type();
};

template < typename F, typename Ret, typename... Args >
inline constexpr bool invocable_with_each_v = invocable_with_each< F, Ret, Args... >::value;

template < typename T, typename HeadType, typename... TailTypes >
struct first_convertible_to {
   inline static constexpr bool value = std::convertible_to< T, HeadType >;
   using type = std::
      conditional_t< value, HeadType, typename first_convertible_to< T, TailTypes... >::type >;
};

template < typename T, typename HeadType >
struct first_convertible_to< T, HeadType > {
   inline static constexpr bool value = std::convertible_to< T, HeadType >;
   using type = std::conditional_t< value, HeadType, void >;
};

template < typename T, typename... Ts >
using first_convertible_to_t = typename first_convertible_to< T, Ts... >::type;


template < template < typename > class condition, typename First, typename... Rest >
struct all_predicate {
  private:
   static constexpr bool eval()
   {
      if constexpr(sizeof...(Rest) == 0) {
         return condition< First >::value;
      } else {
         return condition< First >::value && all_predicate< condition, Rest... >::value;
      }
   }
  public:
   static constexpr bool value = eval();
};
template < template < typename > class condition, typename... Types >
inline constexpr bool all_predicate_v = all_predicate< condition, Types... >::value;

template < template < typename > class condition, typename First, typename... Rest >
struct any_predicate {
  private:
   static constexpr bool eval()
   {
      if(sizeof...(Rest) > 0) {
         if constexpr(condition< First >::value) {
            return true;
         }
         return all_predicate< condition, Rest... >::value;
      } else {
         return condition< First >::value;
      }
   }
  public:
   static constexpr bool value = eval();
};
template < template < typename > class condition, typename... Types >
inline constexpr bool any_predicate_v = any_predicate< condition, Types... >::value;


}  // namespace nor
#endif  // NOR_TYPE_TRAITS_HPP
