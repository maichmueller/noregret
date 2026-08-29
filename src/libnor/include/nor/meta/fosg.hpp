#ifndef NOR_META_FOSG_HPP
#define NOR_META_FOSG_HPP

#include <cstddef>
#include <string_view>
#include <type_traits>

#include "nor/meta/features.hpp"

namespace nor::meta {

/**
 * @brief Structural string constant usable as a non-type template parameter.
 * (std::string_view is not structural on GCC16.)
 */
template < std::size_t N >
struct fixed_str {
   char data[N + 1]{};

   constexpr fixed_str(const char (&str)[N + 1])
   {
      for(std::size_t i = 0; i <= N; ++i) {
         data[i] = str[i];
      }
   }

   constexpr bool operator==(const fixed_str& other) const = default;

   [[nodiscard]] constexpr std::string_view view() const { return {data, N}; }
};

template < std::size_t N >
fixed_str(const char (&)[N]) -> fixed_str< N - 1 >;

/**
 * @brief Searches T and its base classes for a TYPE member with the given
 * identifier and returns its reflection. Returns a null reflection if T does
 * not declare or inherit such a member.
 *
 * Note: the transient range returned by members_of is intentionally consumed
 * inside this consteval function.
 */
consteval std::meta::info member_type_or_void(std::meta::info T, std::string_view name)
{
   for(auto m : std::meta::members_of(T, std::meta::access_context::unchecked())) {
      if(std::meta::is_type(m) and std::meta::has_identifier(m)
         and std::meta::identifier_of(m) == name) {
         return m;
      }
   }
   for(auto base : std::meta::bases_of(T, std::meta::access_context::unchecked())) {
      if(auto member = member_type_or_void(std::meta::type_of(base), name);
         member != std::meta::info{}) {
         return member;
      }
   }
   return std::meta::info{};
}

/**
 * @brief Compile-time check whether T declares or inherits a TYPE member with
 * the given name.
 */
template < typename T, fixed_str Name >
consteval bool has_member_type()
{
   // note: GCC16's std::meta::info is a scalar without an is_null() query --
   // compare against a default-constructed (null) reflection instead
   return not (member_type_or_void(^^T, Name.view()) == std::meta::info{});
}

/**
 * @brief Splice-based lookup of the nested type 'Name.value' of T.
 * Substitution failure if T does not provide such a type member.
 */
template < typename T, fixed_str Name >
   requires(has_member_type< T, Name >())
using fosg_type = [:member_type_or_void(^^T, Name.view()):];

namespace detail {

template < typename T, bool HasMember, fixed_str Name >
struct fosg_type_or_void_impl {
   using type = void;
};

template < typename T, fixed_str Name >
struct fosg_type_or_void_impl< T, true, Name > {
   using type = fosg_type< T, Name >;
};

}  // namespace detail

/**
 * @brief Like fosg_type but resolves to void instead of failing substitution
 * when T does not provide the requested type member.
 */
template < typename T, fixed_str Name >
using fosg_type_or_void = typename detail::
   fosg_type_or_void_impl< T, has_member_type< T, Name >(), Name >::type;

}  // namespace nor::meta

#endif  // NOR_META_FOSG_HPP
