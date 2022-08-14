
#ifndef NOR_CONCRETE_HPP
#define NOR_CONCRETE_HPP

#include <concepts>
#include <range/v3/all.hpp>
#include <utility>
#include <vector>

#include "has.hpp"
#include "is.hpp"
#include "nor/fosg_traits.hpp"
#include "nor/game_defs.hpp"

namespace nor::concepts {

template < typename T >
// clang-format off
concept iterable =
/**/  has::method::begin< T >
   && has::method::begin< const T >
   && has::method::end< T >
   && has::method::end< const T >;
// clang-format on

template <
   typename Map,
   typename KeyType = typename Map::key_type,
   typename MappedType = typename Map::mapped_type >
concept map = iterable< Map > && requires(Map m, KeyType key, MappedType mapped) {
                                    typename Map::key_type;
                                    typename Map::mapped_type;

                                    m.find(key);
                                    {
                                       m[key]
                                       } -> std::same_as< MappedType& >;
                                    {
                                       m.at(key)
                                       } -> std::same_as< MappedType& >;
                                 } and requires(const Map m, KeyType key, MappedType mapped) {
                                          {
                                             m.at(key)
                                             } -> std::same_as< const MappedType& >;
                                       };

template < typename T >
concept action = is::hashable< T > && std::equality_comparable< T >;

template < typename T >
concept chance_outcome = is::hashable< T > && std::equality_comparable< T >;

template < typename T >
concept observation = is::hashable< T > && std::equality_comparable< T >;

template < typename T, typename Observation = typename fosg_auto_traits< T >::observation_type >
// clang-format off
concept public_state =
/**/  observation< Observation >
   && is::sized< T >
   && is::hashable< T >
   && std::is_copy_constructible_v< T >
   && std::equality_comparable< T >
   && has::method::append<
         T,
         Observation&,
         Observation
      >
   && has::method::getitem_r<
         T,
         Observation&,
         size_t
      >;
// clang-format on

template < typename T, typename Observation = typename fosg_auto_traits< T >::observation_type >
// clang-format off
concept info_state =
/**/  public_state< T, Observation >
   && has::method::player< const T >;
// clang-format on

template < typename T >
concept world_state = std::is_move_constructible_v< T > and is::copyable_someway< T >;

template < typename T, typename Action = typename fosg_auto_traits< T >::action_type >
// clang-format off
concept action_policy =
      is::sized< T >
   && iterable< T >
   && has::method::getitem_r< T, double&, Action >
   && has::method::at_r< const T, double, Action >;
// clang-format on

namespace detail {

template <
   typename T,
   typename Infostate,
   typename Action,
   typename ActionPolicy = typename T::action_policy_type >
// clang-format off
concept state_policy_base =
/**/  info_state< Infostate,  typename fosg_auto_traits< Infostate >::observation_type  >
   && action_policy< ActionPolicy >
   && has::trait::action_policy_type< T >;
// clang-format on

}  // namespace detail

template <
   typename T,
   typename Infostate,
   typename Action,
   typename ActionPolicy = typename T::action_policy_type >
// clang-format off
concept memorizing_state_policy =
/**/  detail::state_policy_base< T, Infostate, Action, ActionPolicy >
   && has::method::getitem_r<
         T,
         ActionPolicy&,
         const std::pair< Infostate, std::vector< Action > >&
      >;
// clang-format on

template <
   typename T,
   typename Infostate,
   typename Action,
   typename ActionPolicy = typename T::action_policy_type >
// clang-format off
concept evaluating_state_policy =
/**/  detail::state_policy_base< T, Infostate, Action, ActionPolicy >
   && has::method::getitem_r<
         T,
         ActionPolicy,
         const std::pair<Infostate, std::vector< Action > >&
      >;
// clang-format on

template <
   typename T,
   typename Infostate,
   typename Action,
   typename ActionPolicy = typename T::action_policy_type >
// clang-format off
concept state_policy =
/**/  memorizing_state_policy< T, Infostate, Action, ActionPolicy >
   or evaluating_state_policy< T, Infostate, Action, ActionPolicy >;
// clang-format on

template < typename T, typename Infostate, typename ActionPolicy = typename T::action_policy_type >
// clang-format off
concept default_state_policy =
/**/  info_state< Infostate, typename fosg_auto_traits< Infostate >::observation_type >
   && action_policy< ActionPolicy >
   && has::trait::action_policy_type< T >
   && (has::method::getitem_r<
         T,
         ActionPolicy,
         const std::pair<
            Infostate,
            std::vector<typename fosg_auto_traits<ActionPolicy>::action_type>
         >&
      >
      or has::method::getitem_r<
         T const,
         ActionPolicy,
         const std::pair<
            Infostate,
            std::vector<typename fosg_auto_traits<ActionPolicy>::action_type>
         >&
      >);
// clang-format on

template < typename T, typename Worldstate, typename Action >
// clang-format off
concept chance_distribution =
/**/  world_state< Worldstate >
   && has::method::getitem_r<
         T,
         double,
         const std::pair<
            Worldstate,
            Action
         >&
      >;
// clang-format on

template < typename Env >
// clang-format off
concept deterministic_env =
/**/ requires (Env e) {
         // checks if the stochasticity function is static and constexpr
         {std::bool_constant< (Env::stochasticity(), true) >() } -> std::same_as<std::true_type>;
         // checks if the function is also giving the correct value
         requires(
            Env::stochasticity() == Stochasticity::deterministic
         );
      };
// clang-format on

template <
   typename Env,
   typename Action = typename nor::fosg_auto_traits< Env >::action_type,
   typename Observation = typename nor::fosg_auto_traits< Env >::observation_type,
   typename Infostate = typename nor::fosg_auto_traits< Env >::info_state_type,
   typename Publicstate = typename nor::fosg_auto_traits< Env >::public_state_type,
   typename Worldstate = typename nor::fosg_auto_traits< Env >::world_state_type >
// clang-format off
concept fosg =
/**/  action< Action >
   && observation< Observation >
   && info_state< Infostate >
   && public_state< Publicstate >
   && world_state< Worldstate >
   && is::copyable_someway< Worldstate >
   && has::method::actions< Env, Worldstate >
   && has::method::transition< Env, Worldstate& >
   && has::method::private_observation< Env, Worldstate, Observation >
   && has::method::public_observation< Env, Worldstate, Observation >
   && has::method::private_observation< Env, Action, Observation >
   && has::method::public_observation< Env, Action, Observation >
   && has::method::reward< const Env, Worldstate >
   && has::method::is_terminal< Env, Worldstate >
   && has::method::is_partaking< Env, Worldstate >
   && has::method::active_player< Env >
   && has::method::players< Env, Worldstate >
   && has::method::max_player_count< Env >
   && has::method::player_count< Env >
   && has::method::stochasticity< Env >
   && has::method::turn_dynamic< Env >;
// clang-format on

template <
   typename Env,
   typename Action = typename nor::fosg_auto_traits< Env >::action_type,
   typename Observation = typename nor::fosg_auto_traits< Env >::observation_type,
   typename Infostate = typename nor::fosg_auto_traits< Env >::info_state_type,
   typename Publicstate = typename nor::fosg_auto_traits< Env >::public_state_type,
   typename Worldstate = typename nor::fosg_auto_traits< Env >::world_state_type,
   typename ChanceOutcomeType = std::conditional_t<
      deterministic_env< Env >,
      std::monostate,
      typename fosg_auto_traits< Env >::chance_outcome_type > >
// clang-format off
concept supports_open_history = has::method::open_history< Env, Worldstate, Action, ChanceOutcomeType >;

template <
   typename Env,
   typename Action = typename nor::fosg_auto_traits< Env >::action_type,
   typename Worldstate = typename nor::fosg_auto_traits< Env >::world_state_type,
   typename ChanceOutcomeType = std::conditional_t<
      deterministic_env< Env >,
      std::monostate,
      typename fosg_auto_traits< Env >::chance_outcome_type > >
// clang-format off
concept supports_private_history =
      has::method::private_history< Env, Worldstate, Action, ChanceOutcomeType >
   && has::method::public_history< Env, Worldstate, Action, ChanceOutcomeType > ;

template <
   typename Env,
   typename Action = typename nor::fosg_auto_traits< Env >::action_type,
   typename Worldstate = typename nor::fosg_auto_traits< Env >::world_state_type,
   typename ChanceOutcomeType = std::conditional_t<
      deterministic_env< Env >,
      std::monostate,
      typename fosg_auto_traits< Env >::chance_outcome_type > >
// clang-format off
concept supports_all_histories =
      supports_private_history< Env, Action, Worldstate, ChanceOutcomeType>
   && supports_open_history< Env, Action, Worldstate, ChanceOutcomeType>;

template <
   typename Env,
   typename Action = typename nor::fosg_auto_traits< Env >::action_type,
   typename Observation = typename nor::fosg_auto_traits< Env >::observation_type,
   typename Infostate = typename nor::fosg_auto_traits< Env >::info_state_type,
   typename Publicstate = typename nor::fosg_auto_traits< Env >::public_state_type,
   typename Worldstate = typename nor::fosg_auto_traits< Env >::world_state_type >
// clang-format off
concept deterministic_fosg =
/**/  fosg< Env, Action, Observation, Infostate, Publicstate, Worldstate >
   && deterministic_env< Env >;
// clang-format on

template < typename Env, typename Policy, typename AveragePolicy >
// clang-format off
concept tabular_cfr_requirements =
   concepts::fosg< Env >
   && fosg_traits_partial_match< Policy, Env >::value
   && fosg_traits_partial_match< AveragePolicy, Env >::value
   && concepts::memorizing_state_policy<
       Policy,
       typename fosg_auto_traits< Env >::info_state_type,
       typename fosg_auto_traits< Env >::action_type
      >
   && concepts::memorizing_state_policy<
         AveragePolicy,
         typename fosg_auto_traits< Env >::info_state_type,
         typename fosg_auto_traits< Env >::action_type
      >;

}  // namespace nor::concepts

#endif  // NOR_CONCRETE_HPP
