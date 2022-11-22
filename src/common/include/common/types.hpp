
#ifndef NOR_COMMON_TYPES_HPP
#define NOR_COMMON_TYPES_HPP

#include <iostream>
#include <range/v3/all.hpp>
#include <variant>
#include <vector>

#include "common/misc.hpp"

namespace common {

/// is_specialization checks whether T is a specialized template class of 'Template'
/// This has the limitation of not working with non-type parameters, i.e. templates such as
/// std::array will not be compatible with this type trait
/// usage:
///     constexpr bool is_vector = is_specialization< std::vector< int >, std::vector>;
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
concept pointer_like = requires(T t) {
                          *t;
                          requires std::same_as< U, std::remove_cvref_t< decltype(*t) > >;
                       };

}

template < typename T, typename Hasher = std::hash< std::remove_cvref_t< T > > >
struct value_hasher {
   // allow for heterogenous lookup
   using is_transparent = std::true_type;

   auto operator()(const detail::pointer_like< T > auto& ptr) const { return Hasher{}(*ptr); }
   auto operator()(std::reference_wrapper< T > ref) const { return Hasher{}(ref.get()); }
   auto operator()(const T& t) const { return Hasher{}(t); }
};

template < typename T, typename... Ts >
struct std_hasher: public std_hasher< Ts... >, public std_hasher< T > {
   using std_hasher< T >::operator();
   using std_hasher< Ts... >::operator();
};

template < typename T >
struct std_hasher< T > {
   size_t operator()(const T& t) const { return std::hash< T >{}(t); }
};

template < typename T >
struct variant_hasher;

/// @brief A helper class to enable heterogenous hashing of variant types
///
/// Each individual type T is hashable by their own std::hash< T > class without prior conversion to
/// the variant type. However, this does necessiate a disambiguation for types that are implicitly
/// convertible to a T of the variant type. E.g. the call variant_hasher< std::variant< int,
/// std::string >>{}("Char String") would be ambiguous now and would have to be disambiguated by
/// wrapping the character stiring in std::string("Char String") or variant_type("Char String")
///
/// \tparam Ts, the variant types
template < typename... Ts >
// requires is_specialization_v<T, std::variant>
struct variant_hasher< std::variant< Ts... > >: public std_hasher< Ts... > {
   // allow for heterogenous lookup
   using is_transparent = std::true_type;

   // hashing of the actual variant type
   auto operator()(const std::variant< Ts... >& variant) const
   {
      return std::visit(
         []< typename VarType >(const VarType& var_element) {
            return std::hash< VarType >{}(var_element);
         },
         variant
      );
   }
   // hashing of the individual variant types by their own std hash functions
   using std_hasher< Ts... >::operator();
};

template < typename T >
   requires std::equality_comparable< T >
struct value_comparator {
   // allow for heterogenous lookup
   using is_transparent = std::true_type;

   auto operator()(
      const detail::pointer_like< T > auto& ptr1,
      const detail::pointer_like< T > auto& ptr2
   ) const
   {
      return *ptr1 == *ptr2;
   }
   auto operator()(std::reference_wrapper< T > ref1, std::reference_wrapper< T >& ref2) const
   {
      return ref1.get() == ref2.get();
   }
   auto operator()(const T& t1, const T& t2) const { return t1 == t2; }

   auto operator()(const detail::pointer_like< T > auto& ptr1, std::reference_wrapper< T > ref2)
      const
   {
      return *ptr1 == ref2;
   }
   auto operator()(std::reference_wrapper< T > t1, const detail::pointer_like< T > auto& ptr2) const
   {
      return t1.get() == *ptr2;
   }

   auto operator()(const detail::pointer_like< T > auto& ptr1, const T& t2) const
   {
      return *ptr1 == t2;
   }
   auto operator()(const T& t1, const detail::pointer_like< T > auto& ptr2) const
   {
      return t1 == *ptr2;
   }

   auto operator()(std::reference_wrapper< T > ref1, const T& t2) const { return ref1.get() == t2; }
   auto operator()(const T& t1, std::reference_wrapper< T > ref2) const { return t1 == ref2.get(); }
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

template < typename T >
   requires requires(T t) { std::cout << t; }
struct VectorPrinter {
   const std::vector< T >& value;
   std::string_view delimiter;

   VectorPrinter(const std::vector< T >& vec, const std::string& delim = std::string(", "))
       : value(vec), delimiter(delim)
   {
   }

   friend auto& operator<<(std::ostream& os, const VectorPrinter& printer)
   {
      os << "[";
      for(unsigned int i = 0; i < printer.value.size() - 1; ++i) {
         os << printer.value[i] << printer.delimiter;
      }
      os << printer.value.back() << "]";
      return os;
   }
};

template < typename T >
   requires requires(T t) { std::cout << t; }
struct SpanPrinter {
   ranges::span< T > value;
   std::string_view delimiter;

   SpanPrinter(ranges::span< T > vec, const std::string& delim = std::string(", "))
       : value(vec), delimiter(delim)
   {
   }

   friend auto& operator<<(std::ostream& os, const SpanPrinter& printer)
   {
      os << "[";
      for(unsigned int i = 0; i < printer.value.size() - 1; ++i) {
         os << printer.value[i] << printer.delimiter;
      }
      os << printer.value.back() << "]";
      return os;
   }
};

template < typename... Ts >
struct Overload: Ts... {
   using Ts::operator()...;
};
template < typename... Ts >
Overload(Ts...) -> Overload< Ts... >;

template < typename ReturnType >
struct monostate_error_visitor {
   ReturnType operator()(std::monostate)
   {
      // this should never be visited, but if so --> error
      throw std::logic_error("We entered a std::monostate visit branch.");
      if constexpr(std::is_void_v< ReturnType >) {
         return;
      } else {
         return {};
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
      ts...};
}

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

}  // namespace common

#endif  // NOR_COMMON_TYPES_HPP
