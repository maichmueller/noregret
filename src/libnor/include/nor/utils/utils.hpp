#ifndef NOR_UTILS_HPP
#define NOR_UTILS_HPP

#include <iterator>
#include <type_traits>

#include "common/common.hpp"
#include "nor/concepts.hpp"
#include "nor/game_defs.hpp"
#include "nor/tag.hpp"
#include "player_informed_type.hpp"

namespace nor {

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

}  // namespace nor

namespace std {

template <>
struct hash< nor::hashable_empty > {
   size_t operator()(const nor::hashable_empty&) const { return 0; }
};
}  // namespace std

namespace nor {

template < class >
inline constexpr bool always_false_v = false;

template < bool condition, typename T >
inline decltype(auto) move_if(T&& obj)
{
   if constexpr(condition) {
      return std::move(std::forward< T >(obj));
   } else {
      return std::forward< T >(obj);
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
auto clone(const T& obj)
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
auto clone(const T& obj)
{
   if constexpr(concepts::is::specialization< T, std::reference_wrapper >) {
      return clone(obj.get());
   } else {
      return clone(*obj);
   }
}

template <
   typename Env,
   typename WorldstateHolderType,
   typename WstateHolderOut = WorldstateHolderType >
// clang-format off
requires(
      concepts::fosg< std::remove_cvref_t< Env > >
      and common::is_specialization_v<WorldstateHolderType, WorldstateHolder>
      and common::is_specialization_v<WstateHolderOut, WorldstateHolder>
)
// clang-format on
WstateHolderOut
child_state(Env&& env, const WorldstateHolderType& state, const auto& action_or_outcome)
{
   // clone the current world state first before transitioniong it with this action
   auto next_wstate = state.template copy< WstateHolderOut >();
   // move the new world state forward by the current action
   env.transition(next_wstate, action_or_outcome);
   return next_wstate;
}

/// overload for declaring the out-holder type as the sole input to the templates
template < typename WstateHolderOut, typename Env, typename WorldstateHolderType >
WstateHolderOut
child_state(Env&& env, const WorldstateHolderType& state, const auto& action_or_outcome)
{
   return child_state< Env, WorldstateHolderType, WstateHolderOut >(
      std::forward< Env >(env), state, action_or_outcome
   );
}

template < typename WstateHolderOut, typename Env, typename WorldstateType >
// clang-format off
requires(
      concepts::fosg< std::remove_cvref_t< Env > >
      and common::is_specialization_v<WstateHolderOut, WorldstateHolder>
)
// clang-format on
WstateHolderOut child_state(Env&& env, const WorldstateType& state, const auto& action_or_outcome)
{
   // clone the current world state first before transitioniong it with this action
   auto next_wstate = WstateHolderOut{state};
   // move the new world state forward by the current action
   env.transition(next_wstate, action_or_outcome);
   return next_wstate;
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
   std::pair{Player::zoey, "zoey"},         std::pair{Player::unknown, "unknown"}};

constexpr CEBijection< Stochasticity, std::string_view, 3 > stochasticity_name_bij = {
   std::pair{Stochasticity::deterministic, "deterministic"},
   std::pair{Stochasticity::sample, "sample"},
   std::pair{Stochasticity::choice, "choice"}};

}  // namespace nor

namespace common {
template <>
inline std::string to_string(const nor::Player& e)
{
   return std::string(nor::player_name_bij.at(e));
}
template <>
inline std::string to_string(const nor::Stochasticity& e)
{
   return std::string(nor::stochasticity_name_bij.at(e));
}

template <>
struct printable< nor::Player >: std::true_type {};
template <>
struct printable< nor::Stochasticity >: std::true_type {};

template <>
inline nor::Player from_string< nor::Player >(std::string_view str)
{
   return nor::player_name_bij.at(str);
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
   nor::concepts::mapping ObsBufferMap,
   nor::concepts::mapping InformationStateMap,
   typename Env,
   concepts::world_state WorldstateType,
   concepts::is::specialization< InfostateHolder > InfostateHolderType = std::remove_cvref_t<
      decltype(*(ranges::views::values(std::declval< InformationStateMap& >()).begin())) >,
   concepts::is::specialization< ObservationHolder > ObservationHolderType =
      std::remove_cvref_t< decltype(std::get< 0 >(
         // assume: ObsBufferMap holds a container with std::pair like mapped_type whose pair
         // entries are the obs holders
         *((*(ranges::views::values(std::declval< const ObsBufferMap& >()).begin())).begin())
      )) > >
// clang-format off
requires concepts::fosg< std::remove_cvref_t< Env > >
   and concepts::map_specced<
      ObsBufferMap,
      Player,
      std::vector< std::pair< ObservationHolderType, ObservationHolderType > >
   >
   and concepts::map_specced< InformationStateMap, Player, InfostateHolderType >
// clang-format on
void next_infostate_and_obs_buffers_inplace(
   Env&& env,
   ObsBufferMap& observation_buffer,
   InformationStateMap& infostate_map,
   const WorldstateType& state,
   const auto& action_or_outcome,
   const WorldstateType& next_state
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
            infostate_holder.update(std::move(obs.first), std::move(obs.second));
         }
         obs_history.clear();
         infostate_holder.update(
            public_obs, env.private_observation(player, state, action_or_outcome, next_state)
         );
      }
   }
}

template <
   nor::concepts::mapping ObsBufferMap,
   nor::concepts::mapping InfostateMap,
   typename Env,
   concepts::world_state WorldstateType,
   typename InfostateHolderType = std::remove_cvref_t<
      decltype(*(ranges::views::values(std::declval< const InfostateMap& >()).begin())) >,
   typename ObservationHolderType = std::remove_cvref_t< decltype(std::get< 0 >(
      // assume: ObsBufferMap is a container with std::pair like mapped_type whose pair
      // entries are the obs holders
      *((*(ranges::views::values(std::declval< const ObsBufferMap& >()).begin())).begin())
   )) > >
// clang-format off
requires concepts::fosg< std::remove_cvref_t< Env > >
   and concepts::map_specced<
      ObsBufferMap,
      Player,
      std::vector< std::pair< ObservationHolderType, ObservationHolderType > >
   >
   and concepts::map_specced< InfostateMap, Player, InfostateHolderType >
   and common::is_specialization_v< InfostateHolderType, InfostateHolder >
   and common::is_specialization_v< ObservationHolderType, ObservationHolder >
// clang-format on
auto next_infostate_and_obs_buffers(
   Env&& env,
   const ObsBufferMap& observation_buffer,
   const InfostateMap& infostate_map,
   const WorldstateType& state,
   const auto& action_or_outcome,
   const WorldstateType& next_state
)
{
   ObsBufferMap new_obs_buffer{};
   if constexpr(requires(ObsBufferMap m) {
                   m.reserve(size_t(0));
                } and nor::concepts::is::sized< ObsBufferMap >) {
      new_obs_buffer.reserve(observation_buffer.size());
   }
   InfostateMap new_infostate_map{};
   if constexpr(requires(InfostateMap m) {
                   m.reserve(size_t(0));
                } and nor::concepts::is::sized< InfostateMap >) {
      new_infostate_map.reserve(infostate_map.size());
   }
   for(const auto& [player, mapped] : observation_buffer) {
      new_obs_buffer.emplace(
         player, ranges::to_vector(mapped | ranges::views::transform([&](const auto& pair) {
                                      const auto& [pub_obs, priv_obs] = pair;
                                      return std::pair{pub_obs.copy(), priv_obs.copy()};
                                   }))
      );
   }
   for(const auto& [player, mapped] : infostate_map) {
      new_infostate_map.emplace(player, mapped.copy());
   }
   next_infostate_and_obs_buffers_inplace(
      env, new_obs_buffer, new_infostate_map, state, action_or_outcome, next_state
   );
   return std::tuple{std::move(new_obs_buffer), std::move(new_infostate_map)};
}

template < ranges::range Policy >
auto& normalize_action_policy_inplace(Policy& policy)
{
   auto sum = ranges::accumulate(
      /*range=*/policy | ranges::views::values,
      /*init_value=*/
      std::remove_cvref_t< decltype(*(ranges::views::values(policy).begin())) >{0},
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

template < typename ConvertToType, typename Container, typename tag_dispatch >
auto to_holder_vector(Container&& container, tag_dispatch)
{
   using contained_type = std::remove_cvref_t< decltype(*container.begin()) >;
   static_assert(
      std::convertible_to< contained_type, ConvertToType >,
      "Contained value type of the container is not convertible to the given type."
   );

   using HolderType = common::switch_<
      common::case_< std::same_as< tag_dispatch, tag::action >, ActionHolder< ConvertToType > >,
      common::case_<
         std::same_as< tag_dispatch, tag::chance_outcome >,
         ChanceOutcomeHolder< ConvertToType > >,
      common::case_<
         std::same_as< tag_dispatch, tag::observation >,
         ObservationHolder< ConvertToType > >,
      common::
         case_< std::same_as< tag_dispatch, tag::infostate >, InfostateHolder< ConvertToType > >,
      common::case_<
         std::same_as< tag_dispatch, tag::publicstate >,
         PublicstateHolder< ConvertToType > >,
      common::case_<
         std::same_as< tag_dispatch, tag::worldstate >,
         WorldstateHolder< ConvertToType > > >::type;

   return ranges::to_vector(ranges::views::transform(container, [](const auto& elem) {
      return HolderType{elem};
   }));
}

template < typename To >
constexpr auto static_to = [](const auto& t) { return static_cast< To >(t); };

// has to be imported because otherwise the holder specializations are not used for some reason
// TODO: perhaps there is a better solution to the problem for ActionHolder, ObservationHolder,
// etc.
//  specialization
using ::common::deref;

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
