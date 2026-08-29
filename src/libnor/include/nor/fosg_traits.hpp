
#ifndef NOR_FOSG_TRAITS_HPP
#define NOR_FOSG_TRAITS_HPP

#include <type_traits>
#include <variant>

#include "common/common.hpp"
#include "fwd.hpp"
#include "nor/meta/auto_types.hpp"

namespace nor {

/**
 * @brief Compatibility view of the reflected FOSG associated types.
 *
 * New code should use the nor::auto_* aliases. The primary template keeps
 * direct fosg_traits<T>::member_type users source-compatible while allowing
 * existing explicit specializations to remain valid extension points.
 */
template < typename T >
struct fosg_traits {
   using action_type = auto_action_type< T >;
   using chance_outcome_type = auto_chance_outcome_type< T >;
   using action_policy_type = auto_action_policy_type< T >;
   using chance_distribution_type = auto_chance_distribution_type< T >;
   using observation_type = auto_observation_type< T >;
   using info_state_type = auto_info_state_type< T >;
   using public_state_type = auto_public_state_type< T >;
   using world_state_type = auto_world_state_type< T >;
};

template < typename... Ts >
struct action_variant_type_generator {
   using type = void;
};

template < typename Action, typename ChanceOutcome >
   requires(not std::is_void_v< Action > or not std::is_void_v< ChanceOutcome >)
struct action_variant_type_generator< Action, ChanceOutcome > {
   using type = std::variant<
      std::
         conditional_t< common::is_any_v< Action, void, std::monostate >, std::monostate, Action >,
      std::conditional_t<
         common::is_any_v< ChanceOutcome, void, std::monostate >,
         std::monostate,
         ChanceOutcome > >;
};

template < typename Action, typename ChanceOutcome >
using action_variant_type_generator_t =  //
   typename action_variant_type_generator< Action, ChanceOutcome >::type;

////

template < typename T >
using auto_action_variant_type = action_variant_type_generator_t<
   auto_action_type< T >,
   auto_chance_outcome_type< T > >;

// Compatibility facade for callers that used the implementation detail
// before the reflected aliases became canonical.
template < typename T >
struct fosg_auto_traits {
   using action_type = auto_action_type< T >;
   using chance_outcome_type = auto_chance_outcome_type< T >;
   using action_policy_type = auto_action_policy_type< T >;
   using chance_distribution_type = auto_chance_distribution_type< T >;
   using observation_type = auto_observation_type< T >;
   using info_state_type = auto_info_state_type< T >;
   using public_state_type = auto_public_state_type< T >;
   using world_state_type = auto_world_state_type< T >;
   using action_variant_type = auto_action_variant_type< T >;
};

template < typename SubsetType, typename SupersetType >
struct fosg_traits_partial_match {
  private:
   template < typename A >
   constexpr static bool action_type_is_void()
   {
      return std::is_same_v< auto_action_type< A >, void >;
   }
   template < typename A >
   constexpr static bool observation_type_is_void()
   {
      return std::is_same_v< auto_observation_type< A >, void >;
   }
   template < typename A >
   constexpr static bool info_state_type_is_void()
   {
      return std::is_same_v< auto_info_state_type< A >, void >;
   }
   template < typename A >
   constexpr static bool public_state_type_is_void()
   {
      return std::is_same_v< auto_public_state_type< A >, void >;
   }
   template < typename A >
   constexpr static bool world_state_type_is_void()
   {
      return std::is_same_v< auto_world_state_type< A >, void >;
   }
   template < typename A >
   constexpr static bool all_void()
   {
      return action_type_is_void< A >() && observation_type_is_void< A >()
             && info_state_type_is_void< A >() && public_state_type_is_void< A >()
             && world_state_type_is_void< A >();
   }

   template < typename... >
   constexpr static bool always_false()
   {
      return false;
   }

   constexpr static bool eval()
   {
      if constexpr(all_void< SubsetType >()) {
         return true;
      } else {
         if constexpr(not action_type_is_void< SubsetType >()) {
            if constexpr(not std::is_same_v<
                            auto_action_type< SubsetType >,
                            auto_action_type< SupersetType > >) {
               common::debug< auto_action_type< SubsetType >, auto_action_type< SupersetType > >{};
               static_assert(always_false< SubsetType, SupersetType >, "Action types do not match");
               return false;
            }
         }
         if constexpr(not observation_type_is_void< SubsetType >()) {
            if constexpr(not std::is_same_v<
                            auto_observation_type< SubsetType >,
                            auto_observation_type< SupersetType > >) {
               static_assert(
                  always_false< SubsetType, SupersetType >(), "Observation types do not match"
               );
               return false;
            }
         }
         if constexpr(not info_state_type_is_void< SubsetType >()) {
            if constexpr(not std::is_same_v<
                            auto_info_state_type< SubsetType >,
                            auto_info_state_type< SupersetType > >) {
               static_assert(
                  always_false< SubsetType, SupersetType >(), "Infostate types do not match"
               );
               return false;
            }
         }
         if constexpr(not public_state_type_is_void< SubsetType >()) {
            if constexpr(not std::is_same_v<
                            auto_public_state_type< SubsetType >,
                            auto_public_state_type< SupersetType > >) {
               static_assert(
                  always_false< SubsetType, SupersetType >, "Publicstate types do not match"
               );
               return false;
            }
         }
         if constexpr(not world_state_type_is_void< SubsetType >()) {
            if constexpr(not std::is_same_v<
                            auto_world_state_type< SubsetType >,
                            auto_world_state_type< SupersetType > >) {
               static_assert(
                  always_false< SubsetType, SupersetType >, "Worldstate types do not match"
               );
               return false;
            }
         }
         return true;
      }
   }

  public:
   static constexpr bool value = eval();
};

template < typename T, typename U >
constexpr inline bool fosg_traits_partial_match_v = fosg_traits_partial_match< T, U >::value;

}  // namespace nor

#endif  // NOR_FOSG_TRAITS_HPP
