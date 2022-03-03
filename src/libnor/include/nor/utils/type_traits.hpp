
#ifndef NOR_TYPE_TRAITS_HPP
#define NOR_TYPE_TRAITS_HPP

#include <concepts>

namespace nor {

template < typename T, typename U >
struct const_as {
   using type = std::conditional_t< std::is_const_v< T >, std::add_const_t< U >, U >;
};

template < typename T, typename U >
using const_as_t = typename const_as< T, U >::type;

template < typename T, typename U >
struct ref_as {
   using type = std::conditional_t<
      std::is_lvalue_reference_v< T >,
      std::add_lvalue_reference_t< U >,
      std::conditional_t< std::is_rvalue_reference_v< T >, std::add_rvalue_reference_t< U >, U > >;
};

template < typename T, typename U >
using ref_as_t = typename ref_as< T, U >::type;

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

   constexpr static bool correct_return_result()
   {
      if constexpr(is_invocable) {
         return std::is_same_v< std::invoke_result_t< F, Arg >, Ret >;
      } else {
         return false;
      }
   }

  public:
   inline constexpr static bool value = is_invocable && correct_return_result();
};

template < typename F, typename Ret, typename... Args >
inline constexpr bool invocable_with_each_v = invocable_with_each< F, Ret, Args... >::value;


template <typename T, typename HeadType, typename...TailTypes>
struct first_convertible_to {
   inline static constexpr bool value = std::convertible_to<T, HeadType>;
   using type = std::conditional_t<value, HeadType, typename first_convertible_to<T, TailTypes...>::type>;
};

template <typename T, typename HeadType>
struct first_convertible_to<T, HeadType> {
   inline static constexpr bool value = std::convertible_to<T, HeadType>;
   using type = std::conditional_t<value, HeadType, void>;
};

template <typename T, typename...Ts>
using first_convertible_to_t = typename first_convertible_to<T, Ts...>::type;


}  // namespace nor
#endif  // NOR_TYPE_TRAITS_HPP
