
#ifndef NOR_BATTLESHIP_ENVIRONMENT_HPP
#define NOR_BATTLESHIP_ENVIRONMENT_HPP

#include <functional>
#include <optional>
#include <range/v3/all.hpp>
#include <string>
#include <vector>

#include "battleship/state.hpp"
#include "battleship/utils.hpp"
#include "common/common.hpp"
#include "nor/fosg_states.hpp"
#include "nor/fosg_traits.hpp"
#include "nor/game_defs.hpp"
#include "nor/utils/player_informed_type.hpp"

namespace battleship {

/// the referee's public verdict of a shot
enum class Result : uint8_t { miss = 0, hit, sink };

}  // namespace battleship

namespace nor::games::battleship {

using namespace ::battleship;

inline auto to_battleship_player(const nor::Player& player)
{
   return static_cast< ::battleship::Player >(player);
}
inline auto to_nor_player(const ::battleship::Player& player)
{
   return static_cast< nor::Player >(player);
}

/**
 * @brief Compact observation of battleship.
 *
 * Public observations are either
 * - the bare fact that a player secretly placed a ship (no positional payload), or
 * - a shot together with its referee result (hit / miss / sink).
 *
 * Private observations carry the placed ship's cells and are only ever delivered to the
 * placer himself. The 'none' kind marks observation-less events.
 */
struct Observation {
   enum class Kind : uint8_t { none = 0, hidden_placement, shot };

   Kind kind = Kind::none;
   /// who caused this event
   ::nor::Player actor = ::nor::Player::unknown;
   /// for shots: the fired cell
   std::optional< Cell > target{};
   /// for shots: the referee result
   std::optional< ::battleship::Result > result{};
   /// for private placement confirmations: the two cells of the freshly placed ship
   std::array< std::optional< Cell >, 2 > placed_cells{};

   friend bool operator==(const Observation&, const Observation&) = default;
};

}  // namespace nor::games::battleship

/// has to be visible before the DefaultPublicstate/DefaultInfostate bases below, whose
/// 'observation' concept requirement probes the availability of this specialization
namespace std {

template <>
struct hash< nor::games::battleship::Observation > {
   size_t operator()(const nor::games::battleship::Observation& obs) const noexcept
   {
      size_t seed = 0;
      common::hash_combine(seed, std::hash< unsigned >{}(unsigned(obs.kind)));
      common::hash_combine(seed, std::hash< int >{}(int(obs.actor)));
      if(obs.target.has_value()) {
         common::hash_combine(seed, std::hash< nor::games::battleship::Cell >{}(*obs.target));
      }
      if(obs.result.has_value()) {
         common::hash_combine(seed, std::hash< unsigned >{}(unsigned(*obs.result)));
      }
      for(const auto& cell : obs.placed_cells) {
         if(cell.has_value()) {
            common::hash_combine(seed, std::hash< nor::games::battleship::Cell >{}(*cell));
         }
      }
      return seed;
   }
};

}  // namespace std

namespace nor::games::battleship {

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
 * @brief The FOSG environment adapter of (deterministic) battleship.
 */
class Environment {
  public:
   // nor fosg typedefs
   using world_state_type = State;
   using info_state_type = Infostate;
   using public_state_type = Publicstate;
   using action_type = Action;
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
      return wstate.actions(to_battleship_player(player));
   }

   void transition(world_state_type& worldstate, const action_type& action) const
   {
      worldstate.apply_action(action);
   }

   [[nodiscard]] world_state_type initial_world_state() const { return world_state_type(m_config); }

   /////////////////////////////////
   /// API: players and payoffs  ///
   /////////////////////////////////

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
      return wstate.payoff(to_battleship_player(player));
   }

   ////////////////////////////////
   /// API: observations        ///
   ////////////////////////////////

   observation_type private_observation(
      Player observer,
      const world_state_type& wstate,
      const action_type& action,
      const world_state_type& /*next_wstate*/
   ) const
   {
      const auto* place = std::get_if< Place >(&action);
      if(place == nullptr or to_battleship_player(observer) != wstate.active_player()) {
         // only the placer himself learns where his own ship sits
         return observation_type{};
      }
      return observation_type{
         .kind = Observation::Kind::hidden_placement,
         .actor = observer,
         .placed_cells = {place->a, place->b}};
   }

   observation_type public_observation(
      const world_state_type& wstate,
      const action_type& action,
      const world_state_type& /*next_wstate*/
   ) const
   {
      const auto actor = active_player(wstate);
      if(std::holds_alternative< Place >(action)) {
         // a secret ship was positioned somewhere; its position stays hidden
         return observation_type{.kind = Observation::Kind::hidden_placement, .actor = actor};
      }
      return _shot_observation(wstate, std::get< Fire >(action), actor);
   }

   observation_type tiny_repr(const world_state_type& wstate) const
   {
      // the observation type carries no string payload; the phase identity is all we expose
      (void) wstate;
      return observation_type{};
   }

  private:
   /// the public referee verdict of `fire` shot from pre-transition state `wstate` by `actor`
   [[nodiscard]] observation_type
   _shot_observation(const world_state_type& wstate, const Fire& fire, ::nor::Player actor) const
   {
      const auto shooter = to_battleship_player(actor);
      const auto foe = opponent(shooter);
      if(not wstate.occupies(foe, fire.target)) {
         return observation_type{
            .kind = Observation::Kind::shot,
            .actor = actor,
            .target = fire.target,
            .result = ::battleship::Result::miss};
      }
      // the shot hits; it sinks a ship iff its partner cell was already hit earlier
      bool sunk = false;
      const auto& fleet = wstate.fleet_cells(foe);
      for(size_t i = 0; i + 1 < fleet.size() and not sunk; i += 2) {
         if(fire.target == fleet[i] and wstate.was_hit(shooter, fleet[i + 1])) {
            sunk = true;
         }
         if(fire.target == fleet[i + 1] and wstate.was_hit(shooter, fleet[i])) {
            sunk = true;
         }
      }
      return observation_type{
         .kind = Observation::Kind::shot,
         .actor = actor,
         .target = fire.target,
         .result = sunk ? ::battleship::Result::sink : ::battleship::Result::hit};
   }

  public:
   ////////////////////////////////
   /// API: histories           ///
   ////////////////////////////////

   /**
    * chronological sequence of secret placements + publicly resolved shots. Each entry is
    * masked to what `player` can observe (nullopt for hidden entries).
    */
   [[nodiscard]] std::vector< PlayerInformedType< std::optional< action_variant_type > > >
   private_history(Player player, const world_state_type& wstate) const
   {
      return _masked_history(wstate, [&](const auto& record) {
         return std::holds_alternative< Place >(record.action) and record.actor == player;
      });
   }

   /// chronological sequence with all hidden placements masked out
   [[nodiscard]] std::vector< PlayerInformedType< std::optional< action_variant_type > > >
   public_history(const world_state_type& wstate) const
   {
      return _masked_history(wstate, [](const auto&) { return false; });
   }

   /// the fully open history in which even the secret placements are revealed
   [[nodiscard]] std::vector< PlayerInformedType< action_variant_type > > open_history(
      const world_state_type& wstate
   ) const
   {
      std::vector< PlayerInformedType< action_variant_type > > out;
      out.reserve(_history_size(wstate));
      for(const auto& record : _action_records(wstate)) {
         out.emplace_back(action_variant_type{record.action}, record.actor);
      }
      out.shrink_to_fit();
      return out;
   }

  private:
   struct ActionRecord {
      ::nor::Player actor;
      Action action;
   };

   [[nodiscard]] size_t _history_size(const world_state_type& wstate) const
   {
      return wstate.ships_placed(::battleship::Player::one)
             + wstate.ships_placed(::battleship::Player::two)
             + wstate.shots_used(::battleship::Player::one)
             + wstate.shots_used(::battleship::Player::two);
   }

   /**
    * Reconstructs the chronological action sequence from the state's aggregate data. The
    * secret placements are exchanged strictly alternating ship-by-ship (player one first),
    * then both players alternate firing their shots (again player one first).
    */
   [[nodiscard]] std::vector< ActionRecord > _action_records(const world_state_type& wstate) const
   {
      std::vector< ActionRecord > out;
      out.reserve(_history_size(wstate));
      const auto n_placements = std::max(
         wstate.ships_placed(::battleship::Player::one),
         wstate.ships_placed(::battleship::Player::two)
      );
      for(size_t ship = 0; ship < n_placements; ++ship) {
         for(auto actor : {::battleship::Player::one, ::battleship::Player::two}) {
            const auto& fleet = wstate.fleet_cells(actor);
            if(2 * ship + 1 < fleet.size()) {
               out.push_back(ActionRecord{
                  to_nor_player(actor), Place{fleet[2 * ship], fleet[2 * ship + 1]}});
            }
         }
      }
      const auto n_shots = std::max(
         wstate.shots_used(::battleship::Player::one), wstate.shots_used(::battleship::Player::two)
      );
      for(size_t round = 0; round < n_shots; ++round) {
         for(auto actor : {::nor::Player::alex, ::nor::Player::bob}) {
            const auto& log = wstate.shot_log(to_battleship_player(actor));
            if(round < log.size()) {
               out.push_back(ActionRecord{actor, Fire{log[round]}});
            }
         }
      }
      return out;
   }

   template < typename VisibleFor >
   [[nodiscard]] std::vector< PlayerInformedType< std::optional< action_variant_type > > >
   _masked_history(const world_state_type& wstate, VisibleFor&& visible_for) const
   {
      std::vector< PlayerInformedType< std::optional< action_variant_type > > > out;
      out.reserve(_history_size(wstate));
      for(const auto& record : _action_records(wstate)) {
         if(visible_for(record)) {
            out.emplace_back(action_variant_type{record.action}, record.actor);
         } else {
            out.emplace_back(std::nullopt, record.actor);
         }
      }
      out.shrink_to_fit();
      return out;
   }

  private:
   Config m_config{};
};

}  // namespace nor::games::battleship

namespace common {

template <>
inline std::string to_string(const nor::games::battleship::Observation& value)
{
   namespace bs = nor::games::battleship;
   switch(value.kind) {
      case bs::Observation::Kind::hidden_placement: {
         std::string out = fmt::format("{}:places?", common::to_string(value.actor));
         for(const auto& cell : value.placed_cells) {
            if(cell.has_value()) {
               out += "=" + common::to_string(*cell);
            }
         }
         return out;
      }
      case bs::Observation::Kind::shot: {
         std::string out = fmt::format(
            "{}:fires@{}", common::to_string(value.actor), common::to_string(*value.target)
         );
         if(value.result.has_value()) {
            switch(*value.result) {
               case ::battleship::Result::miss: out += "(miss)"; break;
               case ::battleship::Result::hit: out += "(hit)"; break;
               default: out += "(sink)"; break;
            }
         }
         return out;
      }
      default: return "-";
   }
}

template <>
inline std::string to_string(const ::battleship::Result& value)
{
   switch(value) {
      case ::battleship::Result::miss: return "miss";
      case ::battleship::Result::hit: return "hit";
      default: return "sink";
   }
}

}  // namespace common

COMMON_ENABLE_PRINT(nor::games::battleship, Observation);

namespace nor {

template <>
struct fosg_traits< games::battleship::Infostate > {
   using observation_type = nor::games::battleship::Observation;
};

template <>
struct fosg_traits< games::battleship::Environment > {
   using world_state_type = nor::games::battleship::State;
   using info_state_type = nor::games::battleship::Infostate;
   using public_state_type = nor::games::battleship::Publicstate;
   using action_type = nor::games::battleship::Action;
   using chance_outcome_type = std::monostate;
   using observation_type = nor::games::battleship::Observation;
};

}  // namespace nor

namespace std {

template < typename StateType >
   requires common::
      is_any_v< StateType, nor::games::battleship::Publicstate, nor::games::battleship::Infostate >
   struct hash< StateType > {
   size_t operator()(const StateType& state) const noexcept { return state.hash(); }
};

}  // namespace std

#endif  // NOR_BATTLESHIP_ENVIRONMENT_HPP
