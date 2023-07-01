
#ifndef NOR_CONCRETE_HPP
#define NOR_CONCRETE_HPP

#include <concepts>
#include <range/v3/all.hpp>
// #include <ranges>
#include <utility>
#include <vector>

#include "has.hpp"
#include "is.hpp"
#include "nor/fosg_traits.hpp"
#include "nor/fwd.hpp"
#include "nor/game_defs.hpp"
#include "traits.hpp"

namespace nor::concepts {

template < typename T, typename... Args >
concept brace_initializable = requires(Args&&... args) { T{std::forward< Args >(args)...}; };

template < typename T >
// clang-format off
concept iterable = ranges::range<T> and ranges::range<const T>;
// clang-format on

template < typename Map >
concept map =
   iterable< Map > && has::trait::key_type< Map > && has::trait::mapped_type< Map >
   && requires(Map m, typename Map::key_type key, typename Map::mapped_type mapped) {
         m.emplace(key, mapped);
         m.find(key);
         {
            m[key]
         } -> std::same_as< typename Map::mapped_type& >;
         {
            m.at(key)
         } -> std::same_as< typename Map::mapped_type& >;
      } and requires(const Map m, typename Map::key_type key, typename Map::mapped_type mapped) {
         {
            m.at(key)
         } -> std::convertible_to< typename Map::mapped_type >;
      };

template < typename Map, typename KeyType, typename MappedType >
concept map_specced = map< Map > && std::same_as< KeyType, typename Map::key_type >
                      && std::same_as< MappedType, typename Map::mapped_type >;

template < typename MapLike >
concept mapping = (not common::is_specialization_v< MapLike, ranges::ref_view >) and ranges::
   detail::kv_pair_like_< ranges::range_reference_t< MapLike > >;
//   // the first condition is merely a patch for a (IMO) bug associated with the range
//   // library: https://stackoverflow.com/q/74263486/6798071
//   and requires(MapLike m) {
//      // has to be key-value-like
//      // to iterate over values and
//      // keys only repsectively
////      ranges::views::keys(m);
////      ranges::views::values(m);
//   };

template < typename MapLike, typename KeyType >
concept maps = mapping< MapLike > and requires(MapLike m) {
   requires std::convertible_to<  // given key type has to be
                                  // convertible to actual key type
      KeyType,
      std::remove_cvref_t< decltype(*(ranges::views::keys(m).begin())) > >;
};

template < typename MapLike, typename MappedType = double >
concept mapping_of = requires(MapLike m) {
   mapping< MapLike >;
   // mapped type has to be convertible to the value type
   requires std::is_convertible_v<
      MappedType,
      std::remove_cvref_t< decltype(*(ranges::views::values(m).begin())) > >;
};

template < typename T >
concept action = is::hashable< T > && std::equality_comparable< T >;

template < typename T >
concept chance_outcome = is::hashable< T > && std::equality_comparable< T >;

template < typename T >
concept observation = is::hashable< T > && std::equality_comparable< T >;

template < typename T, typename Observation = auto_observation_type< T > >
// clang-format off
concept public_state =
/**/  observation< Observation >
   && is::sized< T >
   && is::hashable< T >
   && std::copy_constructible< T >
   && std::equality_comparable< T >
   && has::method::update<
         T,
         void,
         ObservationHolder< Observation >  // public observation
      >
   && has::method::getitem_r<
         T,
         ObservationHolder< Observation >,
         size_t
      >;
// clang-format on

template < typename T, typename Observation = auto_observation_type< T > >
// clang-format off
concept info_state =
/**/  observation< Observation >
   && has::method::player< const T >
   && is::sized< T >
   && is::hashable< T >
   && std::copy_constructible< T >
   && std::equality_comparable< T >
   && has::method::update<
         T,
         void,
         ObservationHolder< Observation >, // public observation
         ObservationHolder< Observation >  // private observation
      >
   && has::method::getitem_r<
         T,
         std::pair<
            ObservationHolder< Observation >,
            ObservationHolder< Observation >
         >,
         size_t
      >
   && has::method::latest<
         T,
         std::pair<
            ObservationHolder< Observation >,
            ObservationHolder< Observation >
         >
      >;
// clang-format on

template < typename T >
concept world_state = std::is_move_constructible_v< T > and is::copyable_someway< T >;

template < typename T, typename Action = auto_action_type< T > >
// clang-format off
concept action_policy_view =
      is::sized< T >
   && has::method::at_r< const T, double, Action >;
// clang-format on

template < typename T, typename Action = auto_action_type< T > >
// clang-format off
concept action_policy =
      action_policy_view<T, Action>
   && iterable< T >
   && has::method::getitem_r< T, double&, ActionHolder< Action > >;
// clang-format on

template <
   typename T,
   typename Infostate,
   typename Action,
   typename ActionPolicy = auto_action_policy_type< T > >
// clang-format off
concept default_state_policy =
/**/  info_state< Infostate, auto_observation_type< Infostate > >
   && action_policy< ActionPolicy >
   && has::method::call_r<
         T const,
         ActionPolicy,
         const Infostate&,
         const std::span< ActionHolder< Action > >&
      >;
// clang-format on

namespace detail {

template <
   typename T,
   typename Infostate = auto_info_state_type< T >,
   typename Action = auto_action_type< T >,
   typename ActionPolicy = auto_action_policy_type< T > >
// clang-format off
concept reference_state_policy_base =
/**/  info_state< Infostate,  auto_observation_type< Infostate >  >
   && action_policy< ActionPolicy >;
// clang-format on

template <
   typename T,
   typename Infostate = auto_info_state_type< T >,
   typename Action = auto_action_type< T >,
   typename ActionPolicy = auto_action_policy_type< T > >
// clang-format off
concept value_state_policy_base =
/**/  info_state< Infostate,  auto_observation_type< Infostate >  >
   && action_policy_view< ActionPolicy >;
// clang-format on

}  // namespace detail

template <
   typename T,
   typename Infostate = auto_info_state_type< T >,
   typename Action = auto_action_type< T > >
// clang-format off
concept state_policy_view =
/**/  info_state< Infostate,  auto_observation_type< Infostate >  >
   && has::method::at_r<
         const T,
         typename T::action_policy_type,
         const Infostate&
      >;
// clang-format on

template <
   typename T,
   typename Infostate = auto_info_state_type< T >,
   typename Action = auto_action_type< T >,
   typename ActionPolicy = auto_action_policy_type< T > >
// clang-format off
concept reference_state_policy_no_default =
/**/  detail::reference_state_policy_base< T, Infostate, Action, ActionPolicy >
   && has::method::call_r<
         T,
         ActionPolicy&,
         const Infostate&
      >
   && has::method::at_r<
         const T,
         const ActionPolicy&,
         const Infostate&
      >;
// clang-format on

template <
   typename T,
   typename DefaultPolicy,
   typename Infostate = auto_info_state_type< T >,
   typename Action = auto_action_type< T >,
   typename ActionPolicy = auto_action_policy_type< T > >
// clang-format off
concept reference_state_policy =
/**/  reference_state_policy_no_default< T, Infostate, Action, ActionPolicy >
   && default_state_policy< DefaultPolicy, Infostate, Action, ActionPolicy >
   && has::method::call_r<
         T,
         ActionPolicy&,
         const Infostate&,
         const std::vector< Action >&,
         DefaultPolicy
      >
;
// clang-format on

/// The value state policy concept returns values, instead of references upon queries to its bracket
/// operator. Such queries cannot be written back to, in order to change the state policy.
/// E.g. neural network based policies would fall under this concept
template <
   typename T,
   typename Infostate = auto_info_state_type< T >,
   typename Action = auto_action_type< T >,
   typename ActionPolicy = auto_action_policy_type< T > >
// clang-format off
concept value_state_policy_no_default =
/**/  detail::value_state_policy_base< T, Infostate, Action, ActionPolicy >
   && has::method::call_r<
         T,
         ActionPolicy,
         const Infostate&
      >
   && has::method::at_r<
         const T,
         ActionPolicy,
         const Infostate&
      >;
// clang-format on

template <
   typename T,
   typename DefaultPolicy,
   typename Infostate = auto_info_state_type< T >,
   typename Action = auto_action_type< T >,
   typename ActionPolicy = auto_action_policy_type< T > >
// clang-format off
concept value_state_policy =
/**/  value_state_policy_no_default< T, Infostate, Action, ActionPolicy >
   && default_state_policy< DefaultPolicy, Infostate, Action, ActionPolicy >
   && has::method::call_r<
         T,
         ActionPolicy,
         const Infostate&,
         DefaultPolicy
      >;
// clang-format on

template <
   typename T,
   typename Infostate,
   typename Action,
   typename ActionPolicy = auto_action_policy_type< T > >
// clang-format off
concept state_policy_no_default =
/**/  reference_state_policy_no_default< T, Infostate, Action, ActionPolicy >
   or value_state_policy_no_default< T, Infostate, Action, ActionPolicy >;
// clang-format on

template <
   typename T,
   typename DefaultPolicy,
   typename Infostate,
   typename Action,
   typename ActionPolicy = auto_action_policy_type< T > >
// clang-format off
concept state_policy =
/**/  reference_state_policy< T,DefaultPolicy, Infostate, Action, ActionPolicy >
   or value_state_policy< T , DefaultPolicy, Infostate, Action, ActionPolicy >;
// clang-format on

template < typename T, typename Worldstate, typename Action >
// clang-format off
concept chance_distribution =
/**/  world_state< Worldstate >
   && has::method::call_r<
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
/**/ requires {
         // checks if the stochasticity function is static
         Env::stochasticity();
         // checks if the stochasticity function is constexpr
         requires common::is_constexpr([] { Env::stochasticity();});
         // checks if the function is also giving the correct value
         requires is::deterministic< Env >;
      };
// clang-format on

template <
   typename Env,
   typename Worldstate = auto_world_state_type< Env >,
   typename Outcome = auto_chance_outcome_type< Env > >
// clang-format off
concept stochastic_env =
/**/  not deterministic_env< Env >
   && has::method::chance_actions< Env, Worldstate, Outcome  >
   && has::method::chance_probability< Env, Worldstate, Outcome >;
// clang-format on

template <
   typename Env,
   typename Action = auto_action_type< Env >,
   typename Observation = auto_observation_type< Env >,
   typename Infostate = auto_info_state_type< Env >,
   typename Publicstate = auto_public_state_type< Env >,
   typename Worldstate = auto_world_state_type< Env >,
   typename Outcome = auto_chance_outcome_type< Env > >
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
   && has::method::private_observation< Env, Worldstate, Action, Observation >
   && has::method::public_observation< Env, Worldstate, Action, Observation >
   && has::method::reward< Env, Worldstate >
   && has::method::is_terminal< Env, Worldstate >
   && has::method::is_partaking< Env, Worldstate >
   && has::method::active_player< Env >
   && has::method::players< Env, Worldstate >
   && has::method::max_player_count< Env >
   && has::method::player_count< Env >
   && has::method::stochasticity< const Env >
   && has::method::serialized< const Env >
   && has::method::unrolled< const Env >
   && (not std::same_as< Action, Outcome >)
   && (deterministic_env < Env > or stochastic_env< Env, Worldstate, Outcome >);
// clang-format on

template <
   typename Env,
   typename Action = auto_action_type< Env >,
   typename Observation = auto_observation_type< Env >,
   typename Infostate = auto_info_state_type< Env >,
   typename Publicstate = auto_public_state_type< Env >,
   typename Worldstate = auto_world_state_type< Env >,
   typename ChanceOutcomeType = std::
      conditional_t< deterministic_env< Env >, std::monostate, auto_chance_outcome_type< Env > > >
// clang-format off
concept supports_open_history =
      has::method::open_history< Env, Worldstate, Action, ChanceOutcomeType >;
// clang-format on

template <
   typename Env,
   typename Action = auto_action_type< Env >,
   typename Worldstate = auto_world_state_type< Env >,
   typename ChanceOutcomeType = std::
      conditional_t< deterministic_env< Env >, std::monostate, auto_chance_outcome_type< Env > > >
// clang-format off
concept supports_private_history =
      has::method::private_history< Env, Worldstate, Action, ChanceOutcomeType >
   && has::method::public_history< Env, Worldstate, Action, ChanceOutcomeType > ;
// clang-format on

template <
   typename Env,
   typename Action = auto_action_type< Env >,
   typename Worldstate = auto_world_state_type< Env >,
   typename ChanceOutcomeType = std::
      conditional_t< deterministic_env< Env >, std::monostate, auto_chance_outcome_type< Env > > >
// clang-format off
concept supports_all_histories =
      supports_private_history< Env, Action, Worldstate, ChanceOutcomeType>
   && supports_open_history< Env, Action, Worldstate, ChanceOutcomeType>;
// clang-format on

// template < typename Env >
// concept supports_adhoc_infostate = supports_adhoc_infostate< Env >;

template <
   typename Env,
   typename Action = auto_action_type< Env >,
   typename Observation = auto_observation_type< Env >,
   typename Infostate = auto_info_state_type< Env >,
   typename Publicstate = auto_public_state_type< Env >,
   typename Worldstate = auto_world_state_type< Env > >
// clang-format off
concept deterministic_fosg =
/**/  fosg< Env, Action, Observation, Infostate, Publicstate, Worldstate >
   && deterministic_env< Env >;
// clang-format on

template <
   typename Env,
   typename Worldstate = auto_world_state_type< Env >,
   typename Infostate = auto_info_state_type< Env >,
   typename Publicstate = auto_public_state_type< Env >,
   typename Observation = auto_observation_type< Env >,
   typename Action = auto_action_type< Env > >
// clang-format off
concept stochastic_fosg =
/**/  fosg< Env, Action, Observation, Infostate, Publicstate, Worldstate >
   && deterministic_env< Env >;
// clang-format on

template <
   typename Env,
   typename Policy,
   typename AveragePolicy,
   typename DefaultPolicy,
   typename DefaultAveragePolicy >
// clang-format off
concept tabular_cfr_requirements =
   fosg< Env >
   && fosg_traits_partial_match< Policy, Env >::value
   && fosg_traits_partial_match< AveragePolicy, Env >::value
   && reference_state_policy<
         Policy,
         DefaultPolicy,
         auto_info_state_type< Env >,
         auto_action_type< Env >
      >
   && reference_state_policy<
         AveragePolicy,
         DefaultAveragePolicy,
         auto_info_state_type< Env >,
         auto_action_type< Env >
      >;

}

#endif  // NOR_CONCRETE_HPP
