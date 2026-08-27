
#ifndef NOR_DARK_HEX_ENVIRONMENT_HPP
#define NOR_DARK_HEX_ENVIRONMENT_HPP

#include <optional>
#include <string>
#include <vector>

#include "common/common.hpp"
#include "dark_hex/state.hpp"
#include "dark_hex/utils.hpp"
#include "nor/fosg_states.hpp"
#include "nor/fosg_traits.hpp"
#include "nor/game_defs.hpp"
#include "nor/utils/player_informed_type.hpp"

namespace nor::games::dark_hex {

using namespace ::dark_hex;

inline auto to_dark_hex_player(const nor::Player& player)
{
   return static_cast< ::dark_hex::Player >(player);
}
inline auto to_nor_player(const ::dark_hex::Player& player)
{
   return static_cast< nor::Player >(player);
}

/**
 * @brief Compact observation of dark hex.
 *
 * The game's public observations carry no payload whatsoever: neither player ever learns
 * anything about the opponent's attempts from the public channel alone. All information a
 * player receives arrives privately:
 * - 'stone_placed': the observer's own attempt succeeded (his fresh stone sits at 'cell_index'),
 * - 'attempt_rejected': the observer's own attempt failed because the target cell was occupied
 *   (by either player); no stone was placed,
 * - Kind::none: the empty/no-op observation (every event of the other player, and every public
 *   broadcast -- of which this game has none).
 */
struct Observation {
   enum class Kind : uint8_t { none = 0, stone_placed, attempt_rejected };

   Kind kind = Kind::none;
   /// meaningful iff kind != none: the targeted board cell of the observer's own attempt
   uint8_t cell_index = 0;

   friend bool operator==(const Observation&, const Observation&) = default;
};

}  // namespace nor::games::dark_hex

/// has to be visible before the DefaultPublicstate/DefaultInfostate bases below, whose
/// 'observation' concept requirement probes the availability of this specialization
namespace std {

template <>
struct hash< nor::games::dark_hex::Observation > {
   size_t operator()(const nor::games::dark_hex::Observation& obs) const noexcept
   {
      size_t seed = 0;
      common::hash_combine(seed, std::hash< unsigned >{}(unsigned(obs.kind)));
      if(obs.kind != nor::games::dark_hex::Observation::Kind::none) {
         common::hash_combine(seed, std::hash< unsigned >{}(unsigned(obs.cell_index)));
      }
      return seed;
   }
};

}  // namespace std

namespace nor::games::dark_hex {

class Publicstate: public DefaultPublicstate< Publicstate, Observation > {
   using base = DefaultPublicstate< Publicstate, Observation >;
   using base::base;

   friend base;

   size_t _hash_impl() const
   {
      size_t seed = 0;
      for(const auto& observation : history()) {
         common::hash_combine(seed, std::hash< Observation >{}(observation));
      }
      return seed;
   }
};

class Infostate: public DefaultInfostate< Infostate, Observation > {
   using base = DefaultInfostate< Infostate, Observation >;
   using base::base;

   friend base;

   size_t _hash_impl() const
   {
      size_t seed = 0;
      for(const auto& [public_obs, private_obs] : history()) {
         common::hash_combine(seed, std::hash< Observation >{}(public_obs));
         common::hash_combine(seed, std::hash< Observation >{}(private_obs));
      }
      return seed;
   }
};

/**
 * @brief The FOSG environment adapter of (deterministic) dark hex.
 *
 * The referee feedback of a rejected attempt is private information created by the environment
 * and is therefore delivered through `private_observation` keyed to the acting player only.
 */
class Environment {
  public:
   // nor fosg typedefs
   using world_state_type = State;
   using info_state_type = Infostate;
   using public_state_type = Publicstate;
   using action_type = Move;
   using chance_outcome_type = std::monostate;
   using observation_type = Observation;
   using action_variant_type = action_variant_type_generator_t< action_type, chance_outcome_type >;
   // nor fosg traits
   static constexpr size_t max_player_count() { return 2; }
   static constexpr size_t player_count() { return 2; }
   static constexpr bool serialized() { return true; }
   static constexpr bool unrolled() { return true; }
   static constexpr Stochasticity stochasticity() { return Stochasticity::deterministic; }

  public:
   Environment() = default;
   explicit Environment(Config config) : m_config(std::move(config)) {}

   [[nodiscard]] const Config& config() const { return m_config; }

   ///////////////////////////////////
   /// API: transitions            ///
   ///////////////////////////////////

   std::vector< action_type > actions(Player player, const world_state_type& wstate) const
   {
      return wstate.actions(to_dark_hex_player(player));
   }

   void transition(world_state_type& worldstate, const action_type& action) const
   {
      worldstate.apply_action(action);
   }

   [[nodiscard]] world_state_type initial_world_state() const { return world_state_type(m_config); }

   /////////////////////////////////
   /// API: players and payoffs  ///
   /////////////////////////////////

   /**
    * NOTE: returns the Player::none sentinel (mapped to nor::Player::unknown) on terminal states.
    * Transitions INTO terminal states still query `active_player(next_wstate)` during the
    * observation-buffer flush, so this must never throw or crash there.
    */
   [[nodiscard]] Player active_player(const world_state_type& wstate) const
   {
      return to_nor_player(wstate.active_player());
   }

   static inline std::vector< Player > players(const world_state_type&)
   {
      return {Player::alex, Player::bob};
   }

   static bool is_terminal(const world_state_type& wstate) { return wstate.terminal(); }

   static constexpr bool is_partaking(const world_state_type&, Player) { return true; }

   static double reward(Player player, const world_state_type& wstate)
   {
      return wstate.payoff(to_dark_hex_player(player));
   }

   ////////////////////////////////
   /// API: observations        ///
   ////////////////////////////////

   /**
    * Only the actor himself learns the outcome of his attempt (placement confirmation or
    * rejection). Everybody else -- and the public channel -- receive nothing.
    */
   observation_type private_observation(
      Player observer,
      const world_state_type& wstate,
      const action_type& action,
      const world_state_type& /*next_wstate*/
   ) const
   {
      if(to_dark_hex_player(observer) != wstate.active_player()) {
         return observation_type{};
      }
      if(wstate.is_occupied(action.cell_index)) {
         // resolved from the PRE-transition occupancy: this very attempt targeted a taken cell
         return observation_type{
            .kind = Observation::Kind::attempt_rejected, .cell_index = action.cell_index};
      }
      return observation_type{
         .kind = Observation::Kind::stone_placed, .cell_index = action.cell_index};
   }

   observation_type public_observation(
      const world_state_type& /*wstate*/,
      const action_type& /*action*/,
      const world_state_type& /*next_wstate*/
   ) const
   {
      // dark hex has an entirely silent public channel
      return observation_type{};
   }

   observation_type tiny_repr(const world_state_type&) const { return observation_type{}; }

   ////////////////////////////////
   /// API: histories           ///
   ////////////////////////////////

   /**
    * chronological sequence of all attempts masked to what `player` can observe: his own
    * attempts in full, everything else hidden (nullopt).
    */
   [[nodiscard]] std::vector< PlayerInformedType< std::optional< action_variant_type > > >
   private_history(Player player, const world_state_type& wstate) const
   {
      return _masked_history(wstate, [&](const auto& record) {
         return record.actor == to_dark_hex_player(player);
      });
   }

   /// chronological sequence with every attempt masked out (the public view of dark hex)
   [[nodiscard]] std::vector< PlayerInformedType< std::optional< action_variant_type > > >
   public_history(const world_state_type& wstate) const
   {
      return _masked_history(wstate, [](const auto&) { return false; });
   }

   /// the fully open history in which even the hidden attempts are revealed
   [[nodiscard]] std::vector< PlayerInformedType< action_variant_type > > open_history(
      const world_state_type& wstate
   ) const
   {
      std::vector< PlayerInformedType< action_variant_type > > out;
      out.reserve(wstate.move_log().size());
      for(const auto& entry : wstate.move_log()) {
         out.emplace_back(action_variant_type{Move{entry.cell_index}}, to_nor_player(entry.actor));
      }
      out.shrink_to_fit();
      return out;
   }

  private:
   template < typename VisibleFor >
   [[nodiscard]] std::vector< PlayerInformedType< std::optional< action_variant_type > > >
   _masked_history(const world_state_type& wstate, VisibleFor&& visible_for) const
   {
      std::vector< PlayerInformedType< std::optional< action_variant_type > > > out;
      out.reserve(wstate.move_log().size());
      for(const auto& entry : wstate.move_log()) {
         if(visible_for(entry)) {
            out.emplace_back(
               action_variant_type{Move{entry.cell_index}}, to_nor_player(entry.actor)
            );
         } else {
            out.emplace_back(std::nullopt, to_nor_player(entry.actor));
         }
      }
      out.shrink_to_fit();
      return out;
   }

  private:
   Config m_config{};
};

}  // namespace nor::games::dark_hex

namespace common {

template <>
inline std::string to_string(const nor::games::dark_hex::Observation& value)
{
   namespace dh = nor::games::dark_hex;
   switch(value.kind) {
      case dh::Observation::Kind::stone_placed:
         return fmt::format("placed@{}", unsigned(value.cell_index));
      case dh::Observation::Kind::attempt_rejected:
         return fmt::format("rejected@{}", unsigned(value.cell_index));
      default: return "-";
   }
}

}  // namespace common

COMMON_ENABLE_PRINT(nor::games::dark_hex, Observation);

namespace nor {

template <>
struct fosg_traits< games::dark_hex::Infostate > {
   using observation_type = nor::games::dark_hex::Observation;
};

template <>
struct fosg_traits< games::dark_hex::Environment > {
   using world_state_type = nor::games::dark_hex::State;
   using info_state_type = nor::games::dark_hex::Infostate;
   using public_state_type = nor::games::dark_hex::Publicstate;
   using action_type = nor::games::dark_hex::Move;
   using chance_outcome_type = std::monostate;
   using observation_type = nor::games::dark_hex::Observation;
};

}  // namespace nor

namespace std {

template < typename StateType >
   requires common::
      is_any_v< StateType, nor::games::dark_hex::Publicstate, nor::games::dark_hex::Infostate >
   struct hash< StateType > {
   size_t operator()(const StateType& state) const noexcept { return state.hash(); }
};

}  // namespace std

#endif  // NOR_DARK_HEX_ENVIRONMENT_HPP
