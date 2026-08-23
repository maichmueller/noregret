#ifndef NOR_META_ENUM_NAMES_HPP
#define NOR_META_ENUM_NAMES_HPP

#include <array>
#include <optional>
#include <string_view>
#include <type_traits>

#include "nor/meta/features.hpp"

namespace nor::meta {

using namespace std::literals::string_view_literals;

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

   // Without reflection there is no portable way to derive enumerator names, so
   // the fallback keeps the current behavior via explicit name tables for the
   // enums that need it (nor::Player and nor::Stochasticity -- mirroring the
   // tables in nor/utils/utils.hpp) and reports unknown names for all other enum
   // types.

   #include "nor/game_defs.hpp"

namespace detail {

template < typename E >
struct fallback_name_table {
   [[nodiscard]] static constexpr std::string_view name(E) { return {}; }
   [[nodiscard]] static constexpr std::optional< E > parse(std::string_view)
   {
      return std::nullopt;
   }
};

template < std::size_t N, typename E >
constexpr std::string_view table_lookup(
   const std::array< E, N >& values,
   const std::array< std::string_view, N >& names,
   E value
)
{
   for(std::size_t i = 0; i < N; ++i) {
      if(values[i] == value) {
         return names[i];
      }
   }
   return {};
}

template < std::size_t N, typename E >
constexpr std::optional< E > table_parse(
   const std::array< E, N >& values,
   const std::array< std::string_view, N >& names,
   std::string_view name
)
{
   for(std::size_t i = 0; i < N; ++i) {
      if(names[i] == name) {
         return values[i];
      }
   }
   return std::nullopt;
}

constexpr std::array player_values{
   Player::unknown, Player::chance,  Player::alex,     Player::bob,      Player::cedric,
   Player::dexter,  Player::emily,   Player::florence, Player::gustavo,  Player::henrick,
   Player::ian,     Player::julia,   Player::kelvin,   Player::lea,      Player::michael,
   Player::norbert, Player::oscar,   Player::pedro,    Player::quentin,  Player::rosie,
   Player::sophia,  Player::tristan, Player::ulysses,  Player::victoria, Player::william,
   Player::xavier,  Player::yusuf,   Player::zoey};
constexpr std::array player_names{
   "unknown"sv,  "chance"sv,  "alex"sv,     "bob"sv,     "cedric"sv,  "dexter"sv, "emily"sv,
   "florence"sv, "gustavo"sv, "henrick"sv,  "ian"sv,     "julia"sv,   "kelvin"sv, "lea"sv,
   "michael"sv,  "norbert"sv, "oscar"sv,    "pedro"sv,   "quentin"sv, "rosie"sv,  "sophia"sv,
   "tristan"sv,  "ulysses"sv, "victoria"sv, "william"sv, "xavier"sv,  "yusuf"sv,  "zoey"sv};

template <>
struct fallback_name_table< Player > {
   [[nodiscard]] static constexpr std::string_view name(Player value)
   {
      return table_lookup(player_values, player_names, value);
   }
   [[nodiscard]] static constexpr std::optional< Player > parse(std::string_view str)
   {
      return table_parse(player_values, player_names, str);
   }
};

constexpr std::array stochasticity_values{
   Stochasticity::deterministic,
   Stochasticity::sample,
   Stochasticity::choice};
constexpr std::array stochasticity_names{"deterministic"sv, "sample"sv, "choice"sv};

template <>
struct fallback_name_table< Stochasticity > {
   [[nodiscard]] static constexpr std::string_view name(Stochasticity value)
   {
      return table_lookup(stochasticity_values, stochasticity_names, value);
   }
   [[nodiscard]] static constexpr std::optional< Stochasticity > parse(std::string_view str)
   {
      return table_parse(stochasticity_values, stochasticity_names, str);
   }
};

}  // namespace detail

template < typename E >
   requires std::is_enum_v< E >
[[nodiscard]] constexpr std::string_view enum_name(E value)
{
   return detail::fallback_name_table< E >::name(value);
}

/**
 * @brief Fallback lookup. Note: only nor::Player and nor::Stochasticity are
 * supported without reflection; all other enums resolve to nullopt.
 */
template < typename E >
   requires std::is_enum_v< E >
[[nodiscard]] constexpr std::optional< E > enum_from_name(std::string_view name)
{
   return detail::fallback_name_table< E >::parse(name);
}

#endif  // NOR_REFLECTION

}  // namespace nor::meta

#endif  // NOR_META_ENUM_NAMES_HPP
