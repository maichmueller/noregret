#ifndef NOR_UTILS_HPP
#define NOR_UTILS_HPP

#include <iterator>
#include <type_traits>

#include "common/common.hpp"
#include "nor/concepts.hpp"
#include "nor/game_defs.hpp"
#include "player_informed_type.hpp"

namespace nor::utils {

constexpr auto is_chance_player_pred = [](Player player) { return player == Player::chance; };
constexpr auto is_actual_player_pred = [](Player player) {
   // this comparison also excludes 'player' being 'unknown' (currently set to -2)
   return static_cast< int >(player) > static_cast< int >(Player::chance);
};
constexpr auto is_actual_player_filter = ranges::views::filter(is_actual_player_pred);

/// an empty struct to refer to special cases
struct empty {};

struct hashable_empty {
   constexpr bool operator==(const hashable_empty&) { return true; }
};

}  // namespace nor::utils

namespace std {

template <>
struct hash< nor::utils::hashable_empty > {
   size_t operator()(const nor::utils::hashable_empty&) const { return 0; }
};
}  // namespace std

namespace nor::utils {

template < class >
inline constexpr bool always_false_v = false;

template < bool condition, typename T >
inline std::conditional_t< condition, T&&, T& > move_if(T& obj)
{
   if constexpr(condition) {
      return std::move(obj);
   } else {
      return obj;
   }
}

template < template < typename > class UnaryPredicate, typename T >
inline std::conditional_t< UnaryPredicate< T >::value, T&&, T& > move_if(T& obj)
{
   if constexpr(UnaryPredicate< T >::value) {
      return std::move(obj);
   } else {
      return obj;
   }
}

template < typename T >
auto clone_any_way(const T& obj)
{
   if constexpr(concepts::has::method::clone< T >) {
      return std::unique_ptr< T >(obj.clone());
   } else if constexpr(concepts::has::method::copy< T >) {
      return std::make_unique< T >(obj.copy());
   } else if constexpr(std::is_copy_constructible_v< T >) {
      return std::make_unique< T >(obj);
   } else {
      static_assert(always_false_v< T >, "No cloning/copying method available in given type.");
   }
}

template < typename T >
   requires concepts::is::smart_pointer_like< T > or concepts::is::pointer< T >
            or concepts::is::specialization< T, std::reference_wrapper >
auto clone_any_way(const T& obj)
{
   if constexpr(concepts::is::specialization< T, std::reference_wrapper >) {
      return clone_any_way(obj.get());
   } else {
      return clone_any_way(*obj);
   }
}

template < typename Derived, typename Base, typename Deleter >
   requires std::is_same_v< Deleter, std::default_delete< Base > >
std::unique_ptr< Derived > static_unique_ptr_downcast(std::unique_ptr< Base, Deleter >&& p) noexcept
{
   if constexpr(std::is_same_v< Derived, Base >) {
      return p;
   } else {
      auto d = static_cast< Derived* >(p.release());
      return std::unique_ptr< Derived >(d);
   }
}

template < typename Derived, typename DerivedDeleter, typename Base, typename Deleter >
   requires std::convertible_to< Deleter, DerivedDeleter >
std::unique_ptr< Derived, DerivedDeleter > static_unique_ptr_downcast(
   std::unique_ptr< Base, Deleter >&& p
) noexcept
{
   if constexpr(std::is_same_v< Derived, Base >) {
      return p;
   } else {
      auto d = static_cast< Derived* >(p.release());
      return std::unique_ptr< Derived, DerivedDeleter >(d, std::move(p.get_deleter()));
   }
}

template < typename Derived, typename Base, typename Deleter >
std::unique_ptr< Derived, Deleter > dynamic_unique_ptr_cast(std::unique_ptr< Base, Deleter >&& p)
{
   if(Derived* result = dynamic_cast< Derived* >(p.get())) {
      p.release();
      return std::unique_ptr< Derived, Deleter >(result, std::move(p.get_deleter()));
   }
   return std::unique_ptr< Derived, Deleter >(nullptr, p.get_deleter());
}

template < typename Iter >
class ConstView {
  public:
   static_assert(
      concepts::is::const_iter< Iter >,
      "ConstView can be constructed from const_iterators only."
   );

   ConstView(Iter begin, Iter end) : m_begin(begin), m_end(end) {}

   [[nodiscard]] auto begin() const { return m_begin; }
   [[nodiscard]] auto end() const { return m_end; }

  private:
   Iter m_begin;
   Iter m_end;
};

template < typename Iter >
auto advance(Iter&& iter, typename Iter::difference_type n)
{
   Iter it = std::forward< Iter >(iter);
   std::advance(it, n);
   return it;
}

template < typename Key, typename Value, std::size_t Size >
struct CEMap {
   std::array< std::pair< Key, Value >, Size > data;

   [[nodiscard]] constexpr Value at(const Key& key) const
   {
      const auto itr = std::find_if(begin(data), end(data), [&key](const auto& v) {
         return v.first == key;
      });
      if(itr != end(data)) {
         return itr->second;
      } else {
         throw std::range_error("Not Found");
      }
   }
};

template < typename Key, typename Value, std::size_t Size >
struct CEBijection {
   std::array< std::pair< Key, Value >, Size > data;

   template < typename T >
      requires std::is_same_v< T, Key > or std::is_same_v< T, Value >
   [[nodiscard]] constexpr auto at(const T& elem) const
   {
      const auto itr = std::find_if(begin(data), end(data), [&elem](const auto& v) {
         if constexpr(std::is_same_v< T, Key >) {
            return v.first == elem;
         } else {
            return v.second == elem;
         }
      });
      if(itr != end(data)) {
         if constexpr(std::is_same_v< T, Key >) {
            return itr->second;
         } else {
            return itr->first;
         }
      } else {
         throw std::range_error("Not Found");
      }
   }
};

constexpr CEBijection< Player, std::string_view, 28 > player_name_bij = {
   std::pair{Player::chance, "chance"},     std::pair{Player::alex, "alex"},
   std::pair{Player::bob, "bob"},           std::pair{Player::cedric, "cedric"},
   std::pair{Player::dexter, "dexter"},     std::pair{Player::emily, "emily"},
   std::pair{Player::florence, "florence"}, std::pair{Player::gustavo, "gustavo"},
   std::pair{Player::henrick, "henrick"},   std::pair{Player::ian, "ian"},
   std::pair{Player::julia, "julia"},       std::pair{Player::kelvin, "kelvin"},
   std::pair{Player::lea, "lea"},           std::pair{Player::michael, "michael"},
   std::pair{Player::norbert, "norbert"},   std::pair{Player::oscar, "oscar"},
   std::pair{Player::pedro, "pedro"},       std::pair{Player::quentin, "quentin"},
   std::pair{Player::rosie, "rosie"},       std::pair{Player::sophia, "sophia"},
   std::pair{Player::tristan, "tristan"},   std::pair{Player::ulysses, "ulysses"},
   std::pair{Player::victoria, "victoria"}, std::pair{Player::william, "william"},
   std::pair{Player::xavier, "xavier"},     std::pair{Player::yusuf, "yusuf"},
   std::pair{Player::zoey, "zoey"},         std::pair{Player::unknown, "unknown"}
};

constexpr CEBijection< Stochasticity, std::string_view, 3 > stochasticity_name_bij = {
   std::pair{Stochasticity::deterministic, "deterministic"},
   std::pair{Stochasticity::sample, "sample"},
   std::pair{Stochasticity::choice, "choice"}
};

}  // namespace nor::utils

namespace common {
template <>
inline std::string to_string(const nor::Player& e)
{
   return std::string(nor::utils::player_name_bij.at(e));
}
template <>
inline std::string to_string(const nor::Stochasticity& e)
{
   return std::string(nor::utils::stochasticity_name_bij.at(e));
}

template <>
struct printable< nor::Player >: std::true_type {};
template <>
struct printable< nor::Stochasticity >: std::true_type {};

template <>
inline nor::Player from_string< nor::Player >(std::string_view str)
{
   return nor::utils::player_name_bij.at(str);
}

}  // namespace common

template < nor::concepts::is::enum_ Enum, typename T >
   requires nor::concepts::is::any_of< Enum, nor::Player, nor::Stochasticity >
inline std::string operator+(const T& other, Enum e)
{
   return std::string_view(other) + common::to_string(e);
}
template < nor::concepts::is::enum_ Enum, typename T >
   requires nor::concepts::is::any_of< Enum, nor::Player, nor::Stochasticity >
inline std::string operator+(Enum e, const T& other)
{
   return common::to_string(e) + std::string_view(other);
}

namespace nor {

// Thse are the GTest fixes to suppress their printer errors which cannot lookup the defintions
// inside nor for some reason
inline auto& operator<<(std::ostream& os, nor::Player e)
{
   os << common::to_string(e);
   return os;
}
inline auto& operator<<(std::ostream& os, nor::Stochasticity e)
{
   os << common::to_string(e);
   return os;
}

#ifndef NEW_EMPTY_TYPE
   #define NEW_EMPTY_TYPE decltype([]() {})
#endif

namespace detail {

template < typename Infostate >
auto update_infostate(Infostate& infostate_ref_or_ptr, auto public_obs, auto private_obs)
{
   auto update_impl = [&](auto& contained_istate) {
      contained_istate.update(std::move(public_obs), std::move(private_obs));
   };

   if constexpr(concepts::is::smart_pointer_like< Infostate > or std::is_pointer_v< Infostate >) {
      update_impl(*infostate_ref_or_ptr);
   } else if constexpr(concepts::is::specialization< Infostate, std::reference_wrapper >) {
      update_impl(infostate_ref_or_ptr.get());
   } else {
      update_impl(infostate_ref_or_ptr);
   }
}

}  // namespace detail

/**
 * @brief Fills the infostate of each player with the current observations from the intermittent
 * buffers.
 *
 * @tparam Env
 * @tparam Worldstate
 * @tparam Infostate
 * @tparam Observation
 * @param env
 * @param observation_buffer
 * @param infostate_map
 * @param action_or_outcome
 * @param state
 * @return
 */

template <
   typename ObsBufferMap,
   typename InformationStateMap,
   typename Env,
   typename Worldstate = typename fosg_auto_traits< std::remove_cvref_t< Env > >::world_state_type,
   typename Infostate = typename fosg_auto_traits< std::remove_cvref_t< Env > >::info_state_type,
   typename Observation =
      typename fosg_auto_traits< std::remove_cvref_t< Env > >::observation_type >
// clang-format off
requires concepts::fosg< std::remove_cvref_t< Env >>
   and concepts::map_specced< ObsBufferMap, Player, std::vector< std::pair< Observation, Observation > > >
   and (
         concepts::map_specced< InformationStateMap, Player, sptr< Infostate > >
         or concepts::map_specced< InformationStateMap, Player, uptr< Infostate > >
         or concepts::map_specced< InformationStateMap, Player, std::reference_wrapper< Infostate > >
         or concepts::map_specced< InformationStateMap, Player, Infostate* >
         or concepts::map_specced< InformationStateMap, Player, Infostate >
      )
// clang-format on
void next_infostate_and_obs_buffers_inplace(
   Env&& env,
   ObsBufferMap& observation_buffer,
   InformationStateMap& infostate_map,
   const Worldstate& state,
   const auto& action_or_outcome,
   const Worldstate& next_state
)
{
   // the public observation will be given to every player, so we can establish it outside the loop
   auto public_obs = env.public_observation(state, action_or_outcome, next_state);

   auto active_player = env.active_player(next_state);
   for(auto player : env.players(next_state)) {
      if(player == Player::chance) {
         continue;
      }
      if(player != active_player) {
         // for all but the active player we simply append action and state observation to the
         // buffer. They will be written to an actual infostate once that player becomes the
         // active player again
         auto& player_obs_buffer = observation_buffer[player];
         player_obs_buffer.emplace_back(
            public_obs, env.private_observation(player, state, action_or_outcome, next_state)
         );
      } else {
         // for the active player we first append all recent actions and state observations to the
         // info state, and then follow it up by adding the current action and state
         // observations

         // we are taking the reference here to the position of this infostate in the map, in order
         // to replace it later without needing to refetch it.
         auto& infostate_holder = infostate_map.at(active_player);
         // we consume these observations by moving them into the appendix of the infostates. The
         // cleared observation buffer is still returned and reused, but is now empty.
         auto& obs_history = observation_buffer[active_player];
         for(auto& obs : obs_history) {
            detail::update_infostate(infostate_holder, std::move(obs.first), std::move(obs.second));
         }
         obs_history.clear();
         detail::update_infostate(
            infostate_holder,
            public_obs,
            env.private_observation(player, state, action_or_outcome, next_state)
         );
      }
   }
}

template <
   typename ObsBufferMap,
   typename InformationStateMap,
   typename Env,
   typename Worldstate = typename fosg_auto_traits< std::remove_cvref_t< Env > >::world_state_type,
   typename Infostate = typename fosg_auto_traits< std::remove_cvref_t< Env > >::info_state_type,
   typename Observation =
      typename fosg_auto_traits< std::remove_cvref_t< Env > >::observation_type >
// clang-format off
requires concepts::fosg< std::remove_cvref_t< Env >>
   and concepts::map_specced< ObsBufferMap, Player, std::vector< std::pair< Observation, Observation > > >
   and (
         concepts::map_specced< InformationStateMap, Player, sptr< Infostate > >
         or concepts::map_specced< InformationStateMap, Player, uptr< Infostate > >
         or concepts::map_specced< InformationStateMap, Player, std::reference_wrapper< Infostate > >
         or concepts::map_specced< InformationStateMap, Player, Infostate* >
         or concepts::map_specced< InformationStateMap, Player, Infostate >
      )
// clang-format on
auto next_infostate_and_obs_buffers(
   Env&& env,
   ObsBufferMap observation_buffer,
   InformationStateMap infostate_map,
   const Worldstate& state,
   const auto& action_or_outcome,
   const Worldstate& next_state
)
{
   using mapped_infostate_type = typename InformationStateMap::mapped_type;
   // if the infostate types are referenced values or reference_wrappers then we need to actually
   // copy their pointed to contents.

   // Note that the caller needs to be aware of potential memory leaks occuring from this function!

   if constexpr(std::same_as< mapped_infostate_type, std::reference_wrapper< Infostate > >) {
      for(auto& [player, mapped] : infostate_map) {
         mapped = std::ref(new Infostate(mapped.get()));
      }
   } else if constexpr(std::same_as< mapped_infostate_type, Infostate* >) {
      for(auto& [player, mapped] : infostate_map) {
         mapped = new Infostate(*mapped);
      }
   } else if constexpr(std::same_as< mapped_infostate_type, sptr< Infostate > >) {
      for(auto& [player, mapped] : infostate_map) {
         mapped = std::make_shared< Infostate >(*mapped);
      }
   } else if constexpr(std::same_as< mapped_infostate_type, uptr< Infostate > >) {
      for(auto& [player, mapped] : infostate_map) {
         mapped = std::make_unique< Infostate >(utils::clone_any_way(mapped));
      }
   }

   next_infostate_and_obs_buffers_inplace(
      env, observation_buffer, infostate_map, state, action_or_outcome, next_state
   );
   return std::tuple{std::move(observation_buffer), std::move(infostate_map)};
}

template < typename Env, typename Worldstate >
// clang-format off
requires(
      concepts::fosg< std::remove_cvref_t< Env > >
      and not concepts::is::smart_pointer_like< Worldstate >
      and not concepts::is::pointer< Worldstate >
)
// clang-format on
uptr< Worldstate > child_state(Env&& env, const Worldstate& state, const auto& action_or_outcome)
{
   // clone the current world state first before transitioniong it with this action
   uptr< Worldstate > next_wstate_uptr = utils::static_unique_ptr_downcast< Worldstate >(
      utils::clone_any_way(state)
   );
   // move the new world state forward by the current action
   env.transition(*next_wstate_uptr, action_or_outcome);
   return next_wstate_uptr;
}

template < ranges::range Policy >
auto& normalize_action_policy_inplace(Policy& policy)
{
   auto sum = ranges::accumulate(
      /*range=*/policy | ranges::views::values,
      /*init_value=*/std::remove_cvref_t< decltype(*(ranges::views::values(policy).begin())) >{0},
      /*operation=*/std::plus{}
   );
   for(auto& [action, prob] : policy) {
      prob /= sum;
   }
   return policy;
};

template < ranges::range Policy >
auto normalize_action_policy(const Policy& policy)
{
   Policy copy = policy;
   normalize_action_policy_inplace(copy);
   return copy;
};

template < ranges::range Policy >
auto& normalize_state_policy_inplace(Policy& policy)
{
   for(auto& [_, action_policy] : policy) {
      normalize_action_policy_inplace(action_policy);
   }
   return policy;
};

template < ranges::range Policy >
auto normalize_state_policy(const Policy& policy)
{
   auto copy = policy;
   normalize_state_policy_inplace(copy);
   return copy;
};

}  // namespace nor

namespace common {

template <>
inline std::string to_string< std::monostate >(const std::monostate&)
{
   throw std::logic_error("A monostate should not be converted to string.");
   return "";
}

}  // namespace common

#endif  // NOR_UTILS_HPP
