
#ifndef NOR_COMMON_TYPES_HPP
#define NOR_COMMON_TYPES_HPP

#include <iostream>
#include <range/v3/all.hpp>
#include <variant>
#include <vector>

#include "common/misc.hpp"

namespace common {

/// is_specialization checks whether T is a specialized template class of 'Template'
/// This has the limitation of
/// usage:
///     constexpr bool is_vector = is_specialization< std::vector< int >, std::vector>;
///
/// Note that this trait has 2 limitations:
///  1) Does not work with non-type parameters.
///     (i.e. templates such as std::array will not be compatible with this type trait)
///  2) Generally, templated typedefs do not get captured as the underlying template but as the
///     typedef template. As such the following scenarios will return:
///          specialization<uptr<int>, uptr> << std::endl;            -> false
///          specialization<std::unique_ptr<int>, uptr>;              -> false
///          specialization<std::unique_ptr<int>, std::unique_ptr>;   -> true
///          specialization<uptr<int>, std::unique_ptr>;              -> true
///     for a typedef template template <typename T> using uptr = std::unique_ptr< T >;
template < class T, template < class... > class Template >
struct is_specialization: std::false_type {};

template < template < class... > class Template, class... Args >
struct is_specialization< Template< Args... >, Template >: std::true_type {};

template < class T, template < class... > class Template >
constexpr bool is_specialization_v = is_specialization< T, Template >::value;

template < typename... Ts >
using tuple_of_const_ref = std::tuple< const std::remove_cvref_t< Ts >&... >;

template < bool condition, typename T >
struct const_if {
   using type = T;
};

template < typename T >
struct const_if< true, T > {
   using type = const T;
};

template < bool condition, typename T >
using const_if_t = typename const_if< condition, T >::type;

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
 * @brief type_trait to check if functor F can be invoked with each arg
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

template < bool condition, typename T >
struct const_ref_if;

template < typename T >
struct const_ref_if< true, T > {
   using type = const std::remove_cvref_t< T >&;
};

template < typename T >
struct const_ref_if< false, T > {
   using type = std::remove_cvref_t< T >&;
};

template < bool condition, typename T >
using const_ref_if_t = typename const_ref_if< condition, T >::type;

inline void hash_combine([[maybe_unused]] std::size_t& /*seed*/) {}

template < typename T, typename... Rest >
inline void hash_combine(std::size_t& seed, const T& v, Rest... rest)
{
   std::hash< T > hasher;
   seed ^= hasher(v) + 0x9e3779b9 + (seed << 6) + (seed >> 2);
   hash_combine(seed, rest...);
}

template < typename T, typename Hasher = std::hash< std::remove_cvref_t< T > > >
struct ref_wrapper_hasher {
   // allow for heterogenous lookup
   using is_transparent = std::true_type;

   auto operator()(const std::reference_wrapper< T >& value) const { return Hasher{}(value.get()); }
   auto operator()(const T& value) const { return Hasher{}(value); }
};

template < typename T >
   requires std::equality_comparable< T >
struct ref_wrapper_comparator {
   // allow for heterogenous lookup
   using is_transparent = std::true_type;
   auto operator()(const std::reference_wrapper< T >& ref1, const std::reference_wrapper< T >& ref2)
      const
   {
      return ref1.get() == ref2.get();
   }
   auto operator()(const std::reference_wrapper< T >& ref, const T& t) const
   {
      return ref.get() == t;
   }
   auto operator()(const T& t, const std::reference_wrapper< T >& ref) const
   {
      return t == ref.get();
   }
   auto operator()(const T& t1, const T& t2) const { return t1 == t2; }
};

namespace detail {

template < typename T, typename U >
concept pointer_like_to = requires(T t) {
   *t;
   requires std::same_as< U, std::remove_cvref_t< decltype(*t) > >;
};

}  // namespace detail

template < size_t N >
struct nth_proj {
   template < typename T >
   constexpr decltype(auto) operator()(const T & t) const noexcept
   {
      return std::get< N >(t);
   }
};

template < size_t N >
struct nth_proj_hash {
   template < typename T >
   constexpr decltype(auto) operator()(const T & t) const noexcept
   {
      return std::hash< std::tuple_element_t< N, T > >{}(std::get< N >(t));
   }
};

template < size_t N, typename FixedType = void >
struct nth_proj_comparator {
   template < typename T, typename U >
   static constexpr bool convertible_to = std::convertible_to< T, U >;

   template < typename T, typename U = T >
      requires std::conditional_t<
         std::is_void_v< FixedType >,
         std::integral_constant<
            bool,
            convertible_to< std::tuple_element_t< N, U >, std::tuple_element_t< N, T > >
               or convertible_to< std::tuple_element_t< N, T >, std::tuple_element_t< N, U > > >,
         std::conditional_t<
            convertible_to< FixedType, std::tuple_element_t< N, T > >
               and convertible_to< FixedType, std::tuple_element_t< N, U > >,
            std::true_type,
            std::false_type > >::value
   constexpr decltype(auto) operator()(const T & t, const U & u) const noexcept
   {
      if constexpr(std::is_void_v< FixedType >) {
         return std::equal_to<  //clang-format off
            std::conditional_t<
               convertible_to< std::tuple_element_t< N, T >, std::tuple_element_t< N, U > >,
               std::tuple_element_t< N, T >,
               std::tuple_element_t< N, U > >  //clang-format on
            >{}(std::get< N >(t), std::get< N >(u));
      } else {
         return std::equal_to< FixedType >{}(std::get< N >(t), std::get< N >(u));
      }
   }

   template < typename T >
      requires std::conditional_t<
         std::is_void_v< FixedType >,
         std::true_type,
         std::conditional_t<
            convertible_to< FixedType, std::tuple_element_t< N, T > >,
            std::true_type,
            std::false_type > >::value
   constexpr decltype(auto) operator()(const T & t, const std::tuple_element_t< N, T >& u)
      const noexcept
   {
      if constexpr(std::is_void_v< FixedType >) {
         return std::equal_to< std::tuple_element_t< N, T > >{}(std::get< N >(t), u);
      } else {
         return std::equal_to< FixedType >{}(std::get< N >(t), u);
      }
   }
   template < typename T >
   constexpr decltype(auto) operator()(const std::tuple_element_t< N, T >& u, const T & t)
      const noexcept
   {
      return operator()(t, u);
   }
};

template < typename T, typename Hasher = std::hash< T > >
struct value_hasher {
   // allow for heterogenous lookup
   using is_transparent = std::true_type;

   auto operator()(std::reference_wrapper< T > ref) const noexcept { return Hasher{}(ref); }
   auto operator()(std::reference_wrapper< const T > ref) const noexcept { return Hasher{}(ref); }
   auto operator()(const detail::pointer_like_to< T > auto& ptr_like) const noexcept
   {
      return Hasher{}(*ptr_like);
   }
   template < typename U >
   auto operator()(const U& u) const noexcept
   {
      return Hasher{}(u);
   }
};

/// @brief class which imports the operator() overload of each given type to call their std hash
template < typename T, typename... Ts >
struct std_hasher: public std_hasher< Ts... >, public std_hasher< T > {
   using std_hasher< T >::operator();
   using std_hasher< Ts... >::operator();
};

template < typename T >
struct std_hasher< T >: std::hash< T > {
   using std::hash< T >::operator();
   //   size_t operator()(const T& t) const noexcept { return std::hash< T >{}(t); }
};

template < typename T >
struct variant_hasher;

/// @brief A helper class to enable heterogenous hashing of variant types
///
/// Each individual type T is hashable by their own std::hash< T > class without prior conversion to
/// the variant type. However, this does necessiate a disambiguation for types that are implicitly
/// convertible to a T of the variant type. E.g. the call
///     variant_hasher< std::variant< int, std::string >>{}("Char String")
/// would be ambiguous now and would have to be disambiguated by wrapping the character string in
/// std::string("Char String") or variant_type("Char String")
///
/// \tparam Ts, the variant types
template < typename... Ts >
// requires is_specialization_v<T, std::variant>
struct variant_hasher< std::variant< Ts... > >: public std_hasher< Ts... > {
   // allow for heterogenous lookup
   using is_transparent = std::true_type;

   // hashing of the actual variant type
   auto operator()(const std::variant< Ts... >& variant) const noexcept
   {
      return std::visit(
         []< typename VarType >(const VarType& var_element) noexcept {
            return std::hash< VarType >{}(var_element);
         },
         variant
      );
   }
   // hashing of the individual variant types by their own std hash functions
   using std_hasher< Ts... >::operator();
   //   using std::hash< Ts >::operator()...;
};

template < typename T, typename comparator = std::equal_to< T > >
   requires std::equality_comparable< T >
struct value_comparator {
   // allow for heterogenous lookup
   using is_transparent = std::true_type;

   auto operator()(
      const detail::pointer_like_to< T > auto& ptr1,
      const detail::pointer_like_to< T > auto& ptr2
   ) const noexcept
   {
      return comparator{}(*ptr1, *ptr2);
   }
   auto operator()(std::reference_wrapper< T > ref1, std::reference_wrapper< T >& ref2)
      const noexcept
   {
      return comparator{}(ref1.get(), ref2.get());
   }
   auto operator()(const T& t1, const T& t2) const { return t1 == t2; }

   auto operator()(const detail::pointer_like_to< T > auto& ptr1, std::reference_wrapper< T > ref2)
      const noexcept
   {
      return comparator{}(*ptr1, ref2);
   }
   auto operator()(std::reference_wrapper< T > t1, const detail::pointer_like_to< T > auto& ptr2)
      const noexcept
   {
      return comparator{}(t1.get(), *ptr2);
   }

   auto operator()(const detail::pointer_like_to< T > auto& ptr1, const T& t2) const noexcept
   {
      return comparator{}(*ptr1, t2);
   }
   auto operator()(const T& t1, const detail::pointer_like_to< T > auto& ptr2) const noexcept
   {
      return comparator{}(t1, *ptr2);
   }

   auto operator()(std::reference_wrapper< T > ref1, const T& t2) const noexcept
   {
      return comparator{}(ref1.get(), t2);
   }
   auto operator()(const T& t1, std::reference_wrapper< T > ref2) const noexcept
   {
      return comparator{}(t1, ref2.get());
   }
   /// forwarding template operator in case the comparator allows more inputs
   template < typename V, typename U >
   auto operator()(const V& v, const U& u) const noexcept
   {
      return comparator{}(v, u);
   }
};

/**
 * Literal class type that wraps a constant expression string.
 * Can be used as template parameter to differentiate via 'strings'
 */
template < size_t N >
struct StringLiteral {
   constexpr StringLiteral(const char (&str)[N]) { std::copy_n(str, N, value); }

   char value[N];
};

template < typename... Ts >
struct Overload: Ts... {
   using Ts::operator()...;
};
template < typename... Ts >
Overload(Ts...) -> Overload< Ts... >;

template < typename... Ts >
struct TransparentOverload: Ts... {
   using is_transparent = std::true_type;
   using Ts::operator()...;
};
template < typename... Ts >
TransparentOverload(Ts...) -> TransparentOverload< Ts... >;

template < typename ReturnType >
struct monostate_error_visitor {
   ReturnType operator()(std::monostate)
   {
      // this should never be visited, but if so --> error
      throw std::logic_error("We entered a std::monostate visit branch.");
      if constexpr(std::is_void_v< ReturnType >) {
         return;
      } else {
         return ReturnType{};
      }
   }
};

template < typename TargetVariant, typename... Ts >
auto make_overload_with_monostate(Ts... ts)
{
   return Overload{
      monostate_error_visitor< std::invoke_result_t<
         decltype([] { return std::visit< Overload< Ts... >, TargetVariant >; }),
         Overload< Ts... >,
         TargetVariant > >{},
      ts...
   };
}

/// logical XOR of the conditions (using fold expressions and bitwise xor)
template < typename... Conditions >
struct logical_xor: std::integral_constant< bool, (Conditions::value ^ ...) > {};
/// helper variable to get the contained value of the trait
template < typename... Conditions >
constexpr bool logical_xor_v = logical_xor< Conditions... >::value;

/// logical AND of the conditions (merely aliased)
template < typename... Conditions >
using logical_and = std::conjunction< Conditions... >;
/// helper variable to get the contained value of the trait
template < typename... Conditions >
constexpr bool logical_and_v = logical_and< Conditions... >::value;

/// logical OR of the conditions (merely aliased)
template < typename... Conditions >
using logical_or = std::disjunction< Conditions... >;
/// helper variable to get the contained value of the trait
template < typename... Conditions >
constexpr bool logical_or_v = logical_or< Conditions... >::value;
/// check if type T matches any of the given types in Ts...

/// logical NEGATION of the conditions (specialized for booleans)
template < bool... conditions >
constexpr bool none_of = logical_and_v< std::integral_constant< bool, not conditions >... >;

/// logical ANY of the conditions (specialized for booleans)
template < bool... conditions >
constexpr bool any_of = logical_or_v< std::integral_constant< bool, conditions >... >;

/// logical AND of the conditions (specialized for booleans)
template < bool... conditions >
constexpr bool all_of = logical_and_v< std::integral_constant< bool, conditions >... >;

template < class T, class... Ts >
struct is_any: ::std::disjunction< ::std::is_same< T, Ts >... > {};
template < class T, class... Ts >
inline constexpr bool is_any_v = is_any< T, Ts... >::value;

template < class T, class... Ts >
struct is_none: ::std::negation< ::std::disjunction< ::std::is_same< T, Ts >... > > {};
template < class T, class... Ts >
inline constexpr bool is_none_v = is_none< T, Ts... >::value;

template < class T, class... Ts >
struct all_same: ::std::conjunction< ::std::is_same< T, Ts >... > {};
template < class T, class... Ts >
inline constexpr bool all_same_v = all_same< T, Ts... >::value;

// Define a case for the switch trait
// Cond is the condition for the case to take effect, type is the effect (the type this case uses)
template < bool Cond, typename T >
struct case_ {
   static constexpr bool condition = Cond;
   using type = T;
};

// Switch type trait
template < typename... Cases >
struct switch_;

template < typename... Cases >
using switch_t = typename switch_< Cases... >::type;

template <>
struct switch_<> {
   using type = void;
};

template < typename Case, typename... Cases >
struct switch_< Case, Cases... > {
   using type = std::
      conditional_t< Case::condition, typename Case::type, typename switch_< Cases... >::type >;
};

template < typename Key, typename Value, std::size_t Size >
struct CEMap {
   std::array< std::pair< Key, Value >, Size > data;

   [[nodiscard]] constexpr Value at(const Key& key) const
   {
      const auto itr = std::find_if(begin(data), end(data), [&key](const auto& v) {
         return v.first == key;
      });
      if(itr != end(data)) {
         return itr->second;
      } else {
         throw std::range_error("Not Found");
      }
   }
};

template < typename Key, typename Value, std::size_t Size >
struct CEBijection {
   std::array< std::pair< Key, Value >, Size > data;

   template < typename T >
      requires is_any_v< T, Key, Value >
   [[nodiscard]] constexpr auto at(const T& elem) const
   {
      const auto itr = std::find_if(begin(data), end(data), [&elem](const auto& v) {
         if constexpr(std::is_same_v< T, Key >) {
            return v.first == elem;
         } else {
            return v.second == elem;
         }
      });
      if(itr != end(data)) {
         if constexpr(std::is_same_v< T, Key >) {
            return itr->second;
         } else {
            return itr->first;
         }
      } else {
         throw std::range_error("Not Found");
      }
   }
};

template < typename Predicate >
class not_pred {
  public:
   constexpr explicit not_pred(const Predicate& pred) : m_pred(pred) {}

   constexpr bool operator()(const auto& x) const { return not m_pred(x); }

  protected:
   Predicate m_pred;
};

template < typename T >
struct inferred_iter_value_type {
   using type = std::remove_cvref_t< decltype(*(ranges::begin(std::declval< T& >()))) >;
};

template < typename T >
using inferred_iter_value_type_t = typename inferred_iter_value_type< T >::type;

template < typename T >
struct inferred_iter_first_value_type {
   using type = std::remove_cvref_t< decltype(ranges::begin(std::declval< T& >())->first) >;
};

template < typename T >
using inferred_iter_first_value_type_t = typename inferred_iter_first_value_type< T >::type;

template < typename T >
struct inferred_iter_second_value_type {
   using type = std::remove_cvref_t< decltype(ranges::begin(std::declval< T& >())->second) >;
};

template < typename T >
using inferred_iter_second_value_type_t = typename inferred_iter_second_value_type< T >::type;

template < class... >
inline constexpr bool always_false_v = false;

template < typename T >
decltype(auto) deref(T&& t)
{
   return std::forward< T >(t);
}

template < typename T >
   requires std::is_pointer_v< std::remove_cvref_t< T > >
            or is_specialization_v< std::remove_cvref_t< T >, std::reference_wrapper >
decltype(auto) deref(T&& t)
{
   if constexpr(is_specialization_v< std::remove_cvref_t< T >, std::reference_wrapper >) {
      return std::forward< T >(t).get();
   } else {
      return *std::forward< T >(t);
   }
}

template < typename T >
// clang-format off
   requires(
      is_specialization_v< std::remove_cvref_t< T >, std::shared_ptr >
      or is_specialization_v< std::remove_cvref_t< T >, std::unique_ptr >
   )
decltype(auto)  // clang-format on
deref(T&& t)
{
   return *std::forward< T >(t);
}

template < ranges::range Range >
class deref_view: public ranges::view_base {
  public:
   struct iterator;
   deref_view() = default;
   deref_view(ranges::range auto&& base) : m_base(base) {}

   iterator begin() { return ranges::begin(m_base); }
   iterator end() { return ranges::end(m_base); }

  private:
   Range m_base;
};

template < ranges::range Range >
struct deref_view< Range >::iterator: ranges::iterator_t< Range > {
   using base = ranges::iterator_t< Range >;
   using value_type = std::remove_cvref_t< decltype(deref(*(std::declval< Range >().begin()))) >;
   using difference_type = ranges::range_difference_t< Range >;

   iterator() = default;

   iterator(const base& b) : base{b} {}

   iterator operator++(int) { return static_cast< base& >(*this)++; }

   iterator& operator++()
   {
      ++static_cast< base& >(*this);
      return (*this);
   }

   decltype(auto) operator*() const { return deref(*static_cast< base >(*this)); }
};

template < ranges::range Range >
deref_view(Range&&) -> deref_view< ranges::cpp20::views::all_t< Range > >;

struct deref_fn {
   template < typename Rng >
   auto operator()(Rng&& rng) const
   {
      return deref_view{ranges::views::all(std::forward< Rng >(rng))};
   }

   template < typename Rng >
   friend auto operator|(Rng&& rng, deref_fn const&)
   {
      return deref_view{ranges::views::all(std::forward< Rng >(rng))};
   }
};

}  // namespace common

namespace ranges::views {

constexpr ::common::deref_fn deref{};

}  // namespace ranges::views

#endif  // NOR_COMMON_TYPES_HPP
