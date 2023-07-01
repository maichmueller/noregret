
#ifndef NOR_CONCEPTS_TRAITS_HPP
#define NOR_CONCEPTS_TRAITS_HPP

namespace nor::concepts::has::trait {

template < typename T >
concept action_policy_type = requires(T t) { typename T::action_policy_type; };

template < typename T >
concept chance_outcome_type = requires(T t) { typename T::chance_outcome_type; };

template < typename T >
concept chance_distribution_type = requires(T t) { typename T::chance_distribution_type; };

template < typename T >
concept action_type = requires(T t) { typename T::action_type; };

template < typename T >
concept observation_type = requires(T t) { typename T::observation_type; };

template < typename T >
concept info_state_type = requires(T t) { typename T::info_state_type; };

template < typename T >
concept public_state_type = requires(T t) { typename T::public_state_type; };

template < typename T >
concept world_state_type = requires(T t) { typename T::world_state_type; };

template < typename T >
concept key_type = requires(T t) { typename T::key_type; };

template < typename T >
concept mapped_type = requires(T t) { typename T::mapped_type; };

template < typename T >
concept value_type = requires(T t) { typename T::value_type; };

}  // namespace nor::concepts::has::trait

#endif  // NOR_CONCEPTS_TRAITS_HPP
