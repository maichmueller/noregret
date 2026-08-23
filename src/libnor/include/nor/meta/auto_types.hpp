#ifndef NOR_META_AUTO_TYPES_HPP
#define NOR_META_AUTO_TYPES_HPP

#include <type_traits>

#include "nor/meta/features.hpp"
#include "nor/meta/fosg.hpp"

#if defined(NOR_REFLECTION)

namespace nor::meta {

/**
 * @brief Reflection-based equivalents of the classic nor::auto_*_type traits
 * (see nor/fosg_traits.hpp). Instead of walking fosg_traits<T> specializations
 * and nested typedefs via SFINAE chains, these look up the corresponding TYPE
 * member of T directly through P2996 reflection. Missing members resolve to
 * void, exactly like the classic path.
 *
 * NOTE: these are intentionally separate names in nor::meta. Replacing the
 * canonical nor::auto_*_type aliases globally is phase 3 and must not happen
 * here.
 */

template < typename T >
using auto_action_type = fosg_type_or_void< std::remove_cvref_t< T >, "action_type" >;

template < typename T >
using auto_chance_outcome_type = fosg_type_or_void<
   std::remove_cvref_t< T >,
   "chance_outcome_type" >;

template < typename T >
using auto_action_policy_type = fosg_type_or_void< std::remove_cvref_t< T >, "action_policy_type" >;

template < typename T >
using auto_chance_distribution_type = fosg_type_or_void<
   std::remove_cvref_t< T >,
   "chance_distribution_type" >;

template < typename T >
using auto_observation_type = fosg_type_or_void< std::remove_cvref_t< T >, "observation_type" >;

template < typename T >
using auto_info_state_type = fosg_type_or_void< std::remove_cvref_t< T >, "info_state_type" >;

template < typename T >
using auto_public_state_type = fosg_type_or_void< std::remove_cvref_t< T >, "public_state_type" >;

template < typename T >
using auto_world_state_type = fosg_type_or_void< std::remove_cvref_t< T >, "world_state_type" >;

}  // namespace nor::meta

#endif  // NOR_REFLECTION

#endif  // NOR_META_AUTO_TYPES_HPP
