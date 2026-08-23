#ifndef NOR_META_DIAGNOSTICS_HPP
#define NOR_META_DIAGNOSTICS_HPP

#include <array>
#include <cstddef>
#include <string_view>

#include "nor/meta/features.hpp"
#include "nor/meta/fosg.hpp"

#if defined(NOR_REFLECTION)

namespace nor::meta {

/**
 * @brief The TYPE members a FOSG environment must provide so that the classic
 * nor::auto_*_type trait chain resolves them. This list mirrors the types
 * consumed by concepts::fosg in nor/concepts/concrete.hpp.
 */
inline constexpr std::array< std::string_view, 6 > fosg_member_type_names{
   "action_type",
   "observation_type",
   "info_state_type",
   "public_state_type",
   "world_state_type",
   "chance_outcome_type"};

/**
 * @brief static_assert-friendly check whether Env declares every required FOSG
 * member type.
 */
template < typename Env >
consteval bool has_all_fosg_members()
{
   for(const auto& name : fosg_member_type_names) {
      if(not has_member_type< Env >(name)) {
         return false;
      }
   }
   return true;
}

/**
 * @brief Compile-time check whether T declares a TYPE member with the given
 * plain identifier (non-template convenience overload of
 * has_member_type<T, fixed_str>).
 */
template < typename T >
consteval bool has_member_type(std::string_view name)
{
   return not (member_type_or_void(^^T, name) == std::meta::info{});
}

struct fosg_member_report {
   std::array< std::string_view, fosg_member_type_names.size() > names{};
   std::size_t count = 0;
};

/**
 * @brief Lists the FOSG member types Env does not declare.
 */
template < typename Env >
consteval fosg_member_report missing_fosg_members()
{
   fosg_member_report report;
   for(const auto& name : fosg_member_type_names) {
      if(not has_member_type< Env >(name)) {
         report.names[report.count++] = name;
      }
   }
   return report;
}

/**
 * @brief A fixed capacity string built at compile time.
 */
template < std::size_t Capacity >
struct constexpr_message {
   char data[Capacity]{};
   std::size_t size = 0;

   [[nodiscard]] constexpr std::string_view view() const { return {data, size}; }
};

namespace detail {

constexpr void append(std::string_view str, char* buffer, std::size_t capacity, std::size_t& size)
{
   for(const char c : str) {
      if(size + 1 >= capacity) {
         break;
      }
      buffer[size++] = c;
   }
}

}  // namespace detail

/**
 * @brief Builds a human-readable compile-time message naming the FOSG member
 * types Env fails to declare. Intended to be embedded in static_asserts or
 * checked in unit tests.
 */
template < typename Env, std::size_t Capacity = 256 >
consteval constexpr_message< Capacity > missing_fosg_members_message()
{
   constexpr_message< Capacity > msg;
   const auto report = missing_fosg_members< Env >();
   if(report.count == 0) {
      detail::append("all FOSG member types present", msg.data, Capacity, msg.size);
      return msg;
   }
   detail::append("missing FOSG member type(s): ", msg.data, Capacity, msg.size);
   for(std::size_t i = 0; i < report.count; ++i) {
      if(i > 0) {
         detail::append(", ", msg.data, Capacity, msg.size);
      }
      detail::append(report.names[i], msg.data, Capacity, msg.size);
   }
   return msg;
}

}  // namespace nor::meta

#endif  // NOR_REFLECTION

#endif  // NOR_META_DIAGNOSTICS_HPP
