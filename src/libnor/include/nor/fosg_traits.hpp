
#ifndef NOR_FOSG_TRAITS_HPP
#define NOR_FOSG_TRAITS_HPP

#include "common/common.hpp"
#include "nor/concepts/has.hpp"

namespace nor {

template < typename Env >
struct fosg_traits;

template < typename HeadT, typename... TailTs >
struct action_type_trait {
   using type = typename action_type_trait< TailTs... >::type;
};

template < typename HeadT, typename... TailTs >
   requires(concepts::has::trait::action_type< HeadT >)
struct action_type_trait< HeadT, TailTs... > {
   using type = typename HeadT::action_type;
};

template < typename T >
struct action_type_trait< T > {
   using type = void;
};

template < typename T >
   requires(concepts::has::trait::action_type< T >)
struct action_type_trait< T > {
   using type = typename T::action_type;
};

template < typename... Ts >
using action_type_trait_t = typename action_type_trait< Ts... >::type;

////

template < typename HeadT, typename... TailTs >
struct chance_outcome_type_trait {
   using type = typename chance_outcome_type_trait< TailTs... >::type;
};

template < typename HeadT, typename... TailTs >
   requires(concepts::has::trait::chance_outcome_type< HeadT >)
struct chance_outcome_type_trait< HeadT, TailTs... > {
   using type = typename HeadT::chance_outcome_type;
};

template < typename T >
struct chance_outcome_type_trait< T > {
   using type = void;
};

template < typename T >
   requires(concepts::has::trait::chance_outcome_type< T >)
struct chance_outcome_type_trait< T > {
   using type = typename T::chance_outcome_type;
};

template < typename... Ts >
using chance_outcome_type_trait_t = typename chance_outcome_type_trait< Ts... >::type;

////

template < typename HeadT, typename... TailTs >
struct action_variant_type_trait {
   using type = typename action_variant_type_trait< TailTs... >::type;
};

template < typename HeadT, typename... TailTs >
   requires(concepts::has::trait::action_type< HeadT >)
struct action_variant_type_trait< HeadT, TailTs... > {
   using type = typename HeadT::action_variant_type;
};

template < typename T >
struct action_variant_type_trait< T > {
   using type = void;
};

template < typename T >
   requires(concepts::has::trait::action_type< T >)
struct action_variant_type_trait< T > {
   using type = typename T::action_type;
};

template < typename... Ts >
using action_type_trait_t = typename action_type_trait< Ts... >::type;

////

template < typename HeadT, typename... TailTs >
struct action_policy_type_trait {
   using type = typename action_policy_type_trait< TailTs... >::type;
};

template < typename HeadT, typename... TailTs >
   requires(concepts::has::trait::action_policy_type< HeadT >)
struct action_policy_type_trait< HeadT, TailTs... > {
   using type = typename HeadT::action_policy_type;
};

template < typename T >
struct action_policy_type_trait< T > {
   using type = void;
};

template < typename T >
   requires(concepts::has::trait::action_policy_type< T >)
struct action_policy_type_trait< T > {
   using type = typename T::action_policy_type;
};

template < typename... Ts >
using action_policy_type_trait_t = typename action_policy_type_trait< Ts... >::type;

////

template < typename HeadT, typename... TailTs >
struct chance_distribution_type_trait {
   using type = typename chance_distribution_type_trait< TailTs... >::type;
};

template < typename HeadT, typename... TailTs >
   requires(concepts::has::trait::chance_distribution_type< HeadT >)
struct chance_distribution_type_trait< HeadT, TailTs... > {
   using type = typename HeadT::chance_distribution_type;
};

template < typename T >
struct chance_distribution_type_trait< T > {
   using type = void;
};

template < typename T >
   requires(concepts::has::trait::chance_distribution_type< T >)
struct chance_distribution_type_trait< T > {
   using type = typename T::chance_distribution_type;
};

template < typename... Ts >
using chance_distribution_type_trait_t = typename chance_distribution_type_trait< Ts... >::type;

////

template < typename HeadT, typename... TailTs >
struct info_state_type_trait {
   using type = typename info_state_type_trait< TailTs... >::type;
};

template < typename HeadT, typename... TailTs >
   requires(concepts::has::trait::info_state_type< HeadT >)
struct info_state_type_trait< HeadT, TailTs... > {
   using type = typename HeadT::info_state_type;
};

template < typename T >
struct info_state_type_trait< T > {
   using type = void;
};

template < typename T >
   requires(concepts::has::trait::info_state_type< T >)
struct info_state_type_trait< T > {
   using type = typename T::info_state_type;
};

template < typename... Ts >
using info_state_type_trait_t = typename info_state_type_trait< Ts... >::type;

////

template < typename HeadT, typename... TailTs >
struct observation_type_trait {
   using type = typename observation_type_trait< TailTs... >::type;
};

template < typename HeadT, typename... TailTs >
   requires(concepts::has::trait::observation_type< HeadT >)
struct observation_type_trait< HeadT, TailTs... > {
   using type = typename HeadT::observation_type;
};

template < typename T >
struct observation_type_trait< T > {
   using type = void;
};

template < typename T >
   requires(concepts::has::trait::observation_type< T >)
struct observation_type_trait< T > {
   using type = typename T::observation_type;
};

template < typename... Ts >
using observation_type_trait_t = typename observation_type_trait< Ts... >::type;

////

template < typename HeadT, typename... TailTs >
struct public_state_type_trait {
   using type = typename public_state_type_trait< TailTs... >::type;
};

template < typename HeadT, typename... TailTs >
   requires(concepts::has::trait::public_state_type< HeadT >)
struct public_state_type_trait< HeadT, TailTs... > {
   using type = typename HeadT::public_state_type;
};

template < typename T >
struct public_state_type_trait< T > {
   using type = void;
};

template < typename T >
   requires(concepts::has::trait::public_state_type< T >)
struct public_state_type_trait< T > {
   using type = typename T::public_state_type;
};

template < typename... Ts >
using public_state_type_trait_t = typename public_state_type_trait< Ts... >::type;

////

template < typename HeadT, typename... TailTs >
struct world_state_type_trait {
   using type = typename world_state_type_trait< TailTs... >::type;
};

template < typename HeadT, typename... TailTs >
   requires(concepts::has::trait::world_state_type< HeadT >)
struct world_state_type_trait< HeadT, TailTs... > {
   using type = typename HeadT::world_state_type;
};

template < typename T >
struct world_state_type_trait< T > {
   using type = void;
};

template < typename T >
   requires(concepts::has::trait::world_state_type< T >)
struct world_state_type_trait< T > {
   using type = typename T::world_state_type;
};

template < typename... Ts >
using world_state_type_trait_t = typename world_state_type_trait< Ts... >::type;

////

template < typename... Ts >
struct action_variant_type_generator {
   using type = void;
};

template < typename Action, typename ChanceOutcome >
   requires(not std::is_void_v< Action > or not std::is_void_v< ChanceOutcome >)
struct action_variant_type_generator< Action, ChanceOutcome > {
   using type = std::variant<
      std::conditional_t< std::is_void_v< Action >, std::monostate, ActionHolder< Action > >,
      std::conditional_t<
         std::is_void_v< ChanceOutcome >,
         std::monostate,
         ChanceOutcomeHolder< ChanceOutcome > > >;
};

template < typename Action, typename ChanceOutcome >
using action_variant_type_generator_t = typename action_variant_type_generator<
   Action,
   ChanceOutcome >::type;

////

template < typename T >
struct fosg_auto_traits {
   // auto traits will first select the relevant trait from fosg_traits< T > if it exists in
   // fosg_traits< T > and only if not look for a trait definiton within T. If the trait class is
   // also not specialized/does not hold the type either, then 'void' is the assigned type.
   using action_type = action_type_trait_t< fosg_traits< T >, T >;
   using chance_outcome_type = chance_outcome_type_trait_t< fosg_traits< T >, T >;
   using action_policy_type = action_policy_type_trait_t< fosg_traits< T >, T >;
   using chance_distribution_type = chance_distribution_type_trait_t< fosg_traits< T >, T >;
   using observation_type = observation_type_trait_t< fosg_traits< T >, T >;
   using info_state_type = info_state_type_trait_t< fosg_traits< T >, T >;
   using public_state_type = public_state_type_trait_t< fosg_traits< T >, T >;
   using world_state_type = world_state_type_trait_t< fosg_traits< T >, T >;

   using action_variant_type = action_variant_type_generator_t< action_type, chance_outcome_type >;
};

template < typename T >
using auto_action_type = typename fosg_auto_traits< T >::action_type;
template < typename T >
using auto_chance_outcome_type = typename fosg_auto_traits< T >::chance_outcome_type;
template < typename T >
using auto_action_policy_type = typename fosg_auto_traits< T >::action_policy_type;
template < typename T >
using auto_chance_distribution_type = typename fosg_auto_traits< T >::chance_distribution_type;
template < typename T >
using auto_observation_type = typename fosg_auto_traits< T >::observation_type;
template < typename T >
using auto_info_state_type = typename fosg_auto_traits< T >::info_state_type;
template < typename T >
using auto_public_state_type = typename fosg_auto_traits< T >::public_state_type;
template < typename T >
using auto_world_state_type = typename fosg_auto_traits< T >::world_state_type;
template < typename T >
using auto_action_variant_type = action_variant_type_generator_t<
   auto_action_type< T >,
   auto_chance_outcome_type< T > >;

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
                  always_false< SubsetType, SupersetType >, "Observation types do not match"
               );
               return false;
            }
         }
         if constexpr(not info_state_type_is_void< SubsetType >()) {
            if constexpr(not std::is_same_v<
                            auto_info_state_type< SubsetType >,
                            auto_info_state_type< SupersetType > >) {
               static_assert(
                  always_false< SubsetType, SupersetType >, "Infostate types do not match"
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
