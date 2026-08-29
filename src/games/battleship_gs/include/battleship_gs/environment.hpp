
#ifndef NOR_BATTLESHIP_GS_ENVIRONMENT_HPP
#define NOR_BATTLESHIP_GS_ENVIRONMENT_HPP

#include <functional>
#include <optional>
#include <ranges>
#include <string>
#include <variant>
#include <vector>

#include "battleship_gs/state.hpp"
#include "battleship_gs/utils.hpp"
#include "common/common.hpp"
#include "nor/fosg_states.hpp"
#include "nor/fosg_traits.hpp"
#include "nor/game_defs.hpp"
#include "nor/utils/player_informed_type.hpp"

namespace battleship_gs {

/// the referee's public verdict of a shot
enum class Result : uint8_t { miss = 0, hit, sink };

}  // namespace battleship_gs

namespace nor::games::battleship_gs {

using namespace ::battleship_gs;

inline auto to_battleship_gs_player(const nor::Player& player)
{
   return static_cast< ::battleship_gs::Player >(player);
}
inline auto to_nor_player(const ::battleship_gs::Player& player)
{
   return static_cast< nor::Player >(player);
}

/**
 * @brief Compact observation of general-sum battleship.
 *
 * Public observations are either
 * - the bare fact that a player secretly placed a ship (no positional payload), or
 * - a shot together with its referee result (hit / miss / sink; the identity of a destroyed ship
 *   is never revealed, Appendix E.1).
 *
 * Private observations carry the placed ship's cells and are only ever delivered to the placer
 * himself. The 'none' kind marks observation-less events.
 */
struct Observation {
   enum class Kind : uint8_t { none = 0, hidden_placement, shot };

   Kind kind = Kind::none;
   /// who caused this event
   ::nor::Player actor = ::nor::Player::unknown;
   /// for shots: the fired cell
   std::optional< Cell > target{};
   /// for shots: the referee result
   std::optional< ::battleship_gs::Result > result{};
   /// for private placement confirmations: the cells of the freshly placed ship
   std::array< std::optional< Cell >, max_ship_length > placed_cells{};

   friend bool operator==(const Observation&, const Observation&) = default;
};

}  // namespace nor::games::battleship_gs

/// has to be visible before the DefaultPublicstate/DefaultInfostate bases below, whose
/// 'observation' concept requirement probes the availability of this specialization
namespace std {

template <>
struct hash< nor::games::battleship_gs::Observation > {
   size_t operator()(const nor::games::battleship_gs::Observation& obs) const noexcept
   {
      size_t seed = 0;
      common::hash_combine(seed, std::hash< unsigned >{}(unsigned(obs.kind)));
      common::hash_combine(seed, std::hash< int >{}(int(obs.actor)));
      if(obs.target.has_value()) {
         common::hash_combine(seed, std::hash< nor::games::battleship_gs::Cell >{}(*obs.target));
      }
      if(obs.result.has_value()) {
         common::hash_combine(seed, std::hash< unsigned >{}(unsigned(*obs.result)));
      }
      for(const auto& cell : obs.placed_cells) {
         if(cell.has_value()) {
            common::hash_combine(seed, std::hash< nor::games::battleship_gs::Cell >{}(*cell));
         }
      }
      return seed;
   }
};

}  // namespace std

namespace nor::games::battleship_gs {

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
 * @brief The FOSG environment adapter of the (deterministic) general-sum battleship benchmark
 * (Farina et al. 2019, App. E.1).
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
      return wstate.actions(to_battleship_gs_player(player));
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
      return wstate.payoff(to_battleship_gs_player(player));
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
      if(place == nullptr or to_battleship_gs_player(observer) != wstate.active_player()) {
         // only the placer himself learns where his own ship sits
         return observation_type{};
      }
      observation_type obs{
         .kind = Observation::Kind::hidden_placement,
         .actor = observer,
      };
      const auto length = wstate.pending_ship_length(wstate.active_player());
      for(size_t i : std::views::iota(size_t{0}, length)) {
         obs.placed_cells.at(i) = place->cell(i);
      }
      return obs;
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

   observation_type tiny_repr(const world_state_type&) const { return observation_type{}; }

   ////////////////////////////////
   /// API: histories           ///
   ////////////////////////////////

   /**
    * chronological sequence of secret placements + publicly resolved shots. Each entry is masked
    * to what `player` can observe (nullopt for hidden entries).
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
      return wstate.ships_placed(::battleship_gs::Player::one)
             + wstate.ships_placed(::battleship_gs::Player::two)
             + wstate.shots_used(::battleship_gs::Player::one)
             + wstate.shots_used(::battleship_gs::Player::two);
   }

   /**
    * Reconstructs the chronological action sequence from the state's aggregate data. The secret
    * placements are exchanged strictly alternating ship-by-ship in fleet order (player one
    * first), then both players alternate firing their shots (again player one first).
    */
   [[nodiscard]] std::vector< ActionRecord > _action_records(const world_state_type& wstate) const
   {
      std::vector< ActionRecord > out;
      out.reserve(_history_size(wstate));
      const auto n_placements = std::max(
         wstate.ships_placed(::battleship_gs::Player::one),
         wstate.ships_placed(::battleship_gs::Player::two)
      );
      for(size_t ship : std::views::iota(size_t{0}, n_placements)) {
         for(auto actor : {::battleship_gs::Player::one, ::battleship_gs::Player::two}) {
            const auto& fleet = wstate.fleet_cells(actor);
            if(ship < fleet.size()) {
               const auto& cells = fleet.at(ship);
               const bool horizontal = cells.size() < 2 or cells.front().row == cells.at(1).row;
               if(horizontal) {
                  out.push_back(ActionRecord{
                     to_nor_player(actor), Place{cells.front(), int8_t{0}, int8_t{1}}});
               } else {
                  out.push_back(ActionRecord{
                     to_nor_player(actor), Place{cells.front(), int8_t{1}, int8_t{0}}});
               }
            }
         }
      }
      const auto n_shots = std::max(
         wstate.shots_used(::battleship_gs::Player::one),
         wstate.shots_used(::battleship_gs::Player::two)
      );
      for(size_t round : std::views::iota(size_t{0}, n_shots)) {
         for(auto actor : {::nor::Player::alex, ::nor::Player::bob}) {
            const auto& log = wstate.shot_log(to_battleship_gs_player(actor));
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
   /// the public referee verdict of `fire` shot from pre-transition state `wstate` by `actor`
   [[nodiscard]] observation_type
   _shot_observation(const world_state_type& wstate, const Fire& fire, ::nor::Player actor) const
   {
      namespace bgs = ::battleship_gs;
      const auto shooter = to_battleship_gs_player(actor);
      const auto foe = opponent(shooter);
      if(not wstate.occupies(foe, fire.target)) {
         return observation_type{
            .kind = Observation::Kind::shot,
            .actor = actor,
            .target = fire.target,
            .result = bgs::Result::miss};
      }
      // the shot hits; it sinks its containing ship iff every other cell of that ship was hit
      // earlier (the identity of a sunk ship stays unrevealed)
      const auto& fleet = wstate.fleet_cells(foe);
      const bool sunk = std::ranges::any_of(fleet, [&](const auto& ship) {
         if(std::ranges::find(ship, fire.target) == ship.end()) {
            return false;
         }
         return std::ranges::all_of(ship, [&](Cell cell) {
            return cell == fire.target or wstate.was_hit(shooter, cell);
         });
      });
      return observation_type{
         .kind = Observation::Kind::shot,
         .actor = actor,
         .target = fire.target,
         .result = sunk ? bgs::Result::sink : bgs::Result::hit};
   }

  private:
   Config m_config{};
};

}  // namespace nor::games::battleship_gs

namespace common {

// NOTE: the Result printer has to precede the Observation printer, whose body references it
// (non-dependent name -> must be visible at definition time)
template <>
inline std::string to_string(const ::battleship_gs::Result& value)
{
   switch(value) {
      case ::battleship_gs::Result::miss: return "miss";
      case ::battleship_gs::Result::hit: return "hit";
      default: return "sink";
   }
}

template <>
inline std::string to_string(const nor::games::battleship_gs::Observation& value)
{
   namespace bgs = nor::games::battleship_gs;
   switch(value.kind) {
      case bgs::Observation::Kind::hidden_placement: {
         std::string out = fmt::format("{}:places?", common::to_string(value.actor));
         for(const auto& cell : value.placed_cells) {
            if(cell.has_value()) {
               out += "=" + common::to_string(*cell);
            }
         }
         return out;
      }
      case bgs::Observation::Kind::shot: {
         std::string out = fmt::format(
            "{}:fires@{}", common::to_string(value.actor), common::to_string(*value.target)
         );
         if(value.result.has_value()) {
            out += fmt::format("({})", common::to_string(*value.result));
         }
         return out;
      }
      default: return "-";
   }
}

}  // namespace common

COMMON_ENABLE_PRINT(nor::games::battleship_gs, Observation);

namespace std {

template < typename StateType >
   requires common::is_any_v<
      StateType,
      nor::games::battleship_gs::Publicstate,
      nor::games::battleship_gs::Infostate >
struct hash< StateType > {
   size_t operator()(const StateType& state) const noexcept { return state.hash(); }
};

}  // namespace std

#endif  // NOR_BATTLESHIP_GS_ENVIRONMENT_HPP
