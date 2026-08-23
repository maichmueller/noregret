#ifndef NOR_META_ENUM_NAMES_HPP
#define NOR_META_ENUM_NAMES_HPP

#include <array>
#include <optional>
#include <string_view>
#include <type_traits>

#include "nor/meta/features.hpp"

namespace nor::meta {

#if defined(NOR_REFLECTION)

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

#else  // fallback: no reflection available

   // Generic fallback based on the compiler's pretty-function spelling of an
   // enumerator. This keeps the current behavior for valid enumerators on GCC
   // without requiring any hand-maintained tables.

   #include <limits>

namespace detail {

template < typename E >
constexpr std::string_view builtin_enumerator_name(E value)
{
   std::string_view view = __PRETTY_FUNCTION__;
   constexpr std::string_view marker = "value = ";
   const auto pos = view.find(marker);
   if(pos == std::string_view::npos) {
      return {};
   }
   const auto start = pos + marker.size();
   auto end = view.size();
   for(auto i = start; i < view.size(); ++i) {
      const char c = view[i];
      if(c == ']' or c == ';' or c == ',') {
         end = i;
         break;
      }
   }
   std::string_view name = view.substr(start, end - start);
   // invalid values are spelled like 'Enum(42)' -- treat them as unnamed
   if(name.find('(') != std::string_view::npos) {
      return {};
   }
   // scoped enums carry their qualification which is not part of the
   // enumerator name
   const auto sep = name.rfind("::");
   if(sep != std::string_view::npos) {
      name.remove_prefix(sep + 2);
   }
   return name;
}

}  // namespace detail

template < typename E >
   requires std::is_enum_v< E >
[[nodiscard]] constexpr std::string_view enum_name(E value)
{
   return detail::builtin_enumerator_name(value);
}

/**
 * @brief Fallback name lookup via a bounded scan over the underlying integer
 * representation.
 *
 * Note: unlike the reflected path, this fallback can only find enumerators
 * whose value lies in [-8, 256).
 */
template < typename E >
   requires std::is_enum_v< E >
[[nodiscard]] constexpr std::optional< E > enum_from_name(std::string_view name)
{
   using Base = std::underlying_type_t< E >;
   constexpr Base lower = -8;
   constexpr Base upper = 256;
   for(Base base = lower;; ++base) {
      const E candidate = static_cast< E >(base);
      if(detail::builtin_enumerator_name(candidate) == name) {
         return candidate;
      }
      if(base == upper - 1) {
         break;
      }
   }
   return std::nullopt;
}

#endif  // NOR_REFLECTION

}  // namespace nor::meta

#endif  // NOR_META_ENUM_NAMES_HPP
