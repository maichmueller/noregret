#ifndef NOR_META_ENUM_NAMES_HPP
#define NOR_META_ENUM_NAMES_HPP

#include <array>
#include <cstddef>
#include <optional>
#include <string_view>
#include <type_traits>

#include "nor/meta/features.hpp"

namespace nor::meta {

/**
 * @brief A single (value, name) pair of one enumerator of an enum type.
 */
template < typename E >
   requires std::is_enum_v< E >
struct enum_entry {
   E value;
   std::string_view name;
};

namespace detail {

/**
 * @brief Compile-time table of all enumerator (value, name) pairs of E built
 * via P2996 reflection. The transient range returned by enumerators_of is
 * wrapped in define_static_array so that it can be consumed by the expansion
 * statement.
 */
template < typename E >
   requires std::is_enum_v< E >
consteval auto make_enum_entries()
{
   using Enum = std::remove_cv_t< E >;
   static constexpr auto raw =  //
      std::define_static_array(std::meta::enumerators_of(^^Enum));
   std::array< enum_entry< Enum >, raw.size() > entries{};
   std::size_t i = 0;
   template for(constexpr auto m : raw)
   {
      entries[i] = enum_entry< Enum >{std::meta::extract< Enum >(m), std::meta::identifier_of(m)};
      ++i;
   }
   return entries;
}

}  // namespace detail

/**
 * @brief Returns the name of the given enumerator as a string view.
 *
 * Returns an empty view if the value does not correspond to any enumerator of
 * E.
 */
template < typename E >
   requires std::is_enum_v< E >
[[nodiscard]] constexpr std::string_view enum_name(E value)
{
   static constexpr auto entries = detail::make_enum_entries< E >();
   for(const auto& entry : entries) {
      if(entry.value == value) {
         return entry.name;
      }
   }
   return {};
}

/**
 * @brief Returns the enumerator of E with the given name or nullopt if no such
 * enumerator exists.
 */
template < typename E >
   requires std::is_enum_v< E >
[[nodiscard]] constexpr std::optional< E > enum_from_name(std::string_view name)
{
   static constexpr auto entries = detail::make_enum_entries< E >();
   for(const auto& entry : entries) {
      if(entry.name == name) {
         return entry.value;
      }
   }
   return std::nullopt;
}

}  // namespace nor::meta

#endif  // NOR_META_ENUM_NAMES_HPP
