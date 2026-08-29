#ifndef NOR_META_AUTO_TYPES_HPP
#define NOR_META_AUTO_TYPES_HPP

#include <type_traits>

#include "nor/meta/features.hpp"
#include "nor/meta/fosg.hpp"

namespace nor::meta {

/**
 * @brief Reflected FOSG associated types.
 *
 * These aliases look up the corresponding TYPE member through P2996
 * reflection. Missing members resolve to void, which preserves the probing
 * semantics of the former trait chain.
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

// Keep the established nor::auto_* names as aliases to the reflected source of
// truth. This makes reflection canonical without forcing callers to migrate
// their spelling.
namespace nor {

template < typename T >
using auto_action_type = meta::auto_action_type< T >;

template < typename T >
using auto_chance_outcome_type = meta::auto_chance_outcome_type< T >;

template < typename T >
using auto_action_policy_type = meta::auto_action_policy_type< T >;

template < typename T >
using auto_chance_distribution_type = meta::auto_chance_distribution_type< T >;

template < typename T >
using auto_observation_type = meta::auto_observation_type< T >;

template < typename T >
using auto_info_state_type = meta::auto_info_state_type< T >;

template < typename T >
using auto_public_state_type = meta::auto_public_state_type< T >;

template < typename T >
using auto_world_state_type = meta::auto_world_state_type< T >;

}  // namespace nor

#endif  // NOR_META_AUTO_TYPES_HPP
