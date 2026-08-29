
#ifndef NOR_CENTIPEDE_ENVIRONMENT_HPP
#define NOR_CENTIPEDE_ENVIRONMENT_HPP

#include <optional>
#include <utility>
#include <vector>

#include "centipede/state.hpp"
#include "centipede/utils.hpp"
#include "common/common.hpp"
#include "nor/fosg_states.hpp"
#include "nor/fosg_traits.hpp"
#include "nor/game_defs.hpp"
#include "nor/utils/player_informed_type.hpp"

namespace nor::games::centipede {

using namespace ::centipede;

inline auto to_centipede_player(const nor::Player& player)
{
   return static_cast< ::centipede::Player >(player);
}
inline auto to_nor_player(const ::centipede::Player& player)
{
   return static_cast< nor::Player >(player);
}

/**
 * @brief Compact observation of the centipede game.
 *
 * The game has PERFECT information: every take/push decision is publicly observed by both
 * players, so all payloads below are public and private observations are always empty (infosets
 * coincide with public states). The terminal transition additionally announces how the game ended
 * and the resulting coin split; payoffs themselves are read off the world state via reward().
 */
struct Observation {
   /// who moved
   std::optional< nor::Player > moved_by{};
   /// the move that was played
   std::optional< Move > move{};
   /// post-transition round index of the acting player for the NEXT decision (public echo)
   std::optional< size_t > next_round{};
   /// who acts next (none <=> terminal)
   std::optional< nor::Player > next_active{};
   /// public announcement of how the game ended (terminal transitions only)
   std::optional< TerminalCause > terminal_cause{};
   /// final coin holdings in (one, two) order (terminal transitions only)
   std::optional< std::pair< uint64_t, uint64_t > > coins{};

   friend bool operator==(const Observation&, const Observation&) = default;
};

}  // namespace nor::games::centipede

/// has to be visible before the DefaultPublicstate/DefaultInfostate bases below, whose
/// 'observation' concept requirement probes the availability of this specialization
namespace std {

template <>
struct hash< nor::games::centipede::Observation > {
   size_t operator()(const nor::games::centipede::Observation& obs) const noexcept
   {
      namespace cp = nor::games::centipede;
      size_t seed = 0;
      if(obs.moved_by.has_value()) {
         common::hash_combine(seed, std::hash< int >{}(1 + int(*obs.moved_by)));
      }
      if(obs.move.has_value()) {
         common::hash_combine(seed, std::hash< bool >{}(obs.move->take));
      }
      if(obs.next_round.has_value()) {
         common::hash_combine(seed, std::hash< size_t >{}(7u + *obs.next_round));
      }
      if(obs.next_active.has_value()) {
         common::hash_combine(seed, std::hash< int >{}(31 + int(*obs.next_active)));
      }
      if(obs.terminal_cause.has_value()) {
         common::hash_combine(seed, std::hash< unsigned >{}(97u + unsigned(*obs.terminal_cause)));
      }
      if(obs.coins.has_value()) {
         common::hash_combine(
            seed,
            std::hash< unsigned long long >{}(
               211ull + obs.coins->first * 4096ull + obs.coins->second
            )
         );
      }
      return seed;
   }
};

}  // namespace std

namespace nor::games::centipede {

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
 * @brief The FOSG environment adapter of the centipede game.
 *
 * REWARD MODEL: GENERAL-SUM -- reward(player) returns that player's OWN doubled-pile holding.
 * As with Shapley's game, use nash_conv(..., constant_sum=false) / per-player best-response gaps
 * for equilibrium-quality reporting; exploitability()'s zero-sum normalization is not meaningful
 * here.
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

   Environment() = default;
   explicit Environment(Config config) : m_config(config) {}

   [[nodiscard]] const Config& config() const { return m_config; }

   ///////////////////////////////////
   /// API: transitions            ///
   ///////////////////////////////////

   [[nodiscard]] std::vector< action_type > actions(Player player, const world_state_type& wstate)
      const
   {
      return wstate.actions(to_centipede_player(player));
   }

   void transition(world_state_type& worldstate, const action_type& action) const
   {
      worldstate.apply_action(action);
   }

   void transition(world_state_type& worldstate, action_type&& action) const
   {
      worldstate.apply_action(std::move(action));
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

   static std::vector< Player > players(const world_state_type&)
   {
      return {Player::alex, Player::bob};
   }

   static bool is_terminal(const world_state_type& wstate) { return wstate.terminal(); }

   static constexpr bool is_partaking(const world_state_type&, Player) { return true; }

   /// alex is player one, bob is player two; each receives his own coin holding (general-sum)
   static double reward(Player player, const world_state_type& wstate)
   {
      return wstate.payoff(to_centipede_player(player));
   }

   ////////////////////////////////
   /// API: observations        ///
   ////////////////////////////////

   observation_type private_observation(
      Player /*observer*/,
      const world_state_type& /*wstate*/,
      const action_type& /*action*/,
      const world_state_type& /*next_wstate*/
   ) const
   {
      // perfect information: moves are public, nothing is ever privately revealed
      return observation_type{};
   }

   observation_type public_observation(
      const world_state_type& wstate,
      const action_type& action,
      const world_state_type& next_wstate
   ) const
   {
      observation_type obs{
         .moved_by = to_nor_player(wstate.active_player()),
         .move = action,
         .next_round = next_wstate.round(),
         .next_active = to_nor_player(next_wstate.active_player())};
      if(next_wstate.terminal()) {
         obs.terminal_cause = next_wstate.terminal_cause();
         obs.coins = std::pair{
            next_wstate.coin_holdings(::centipede::Player::one),
            next_wstate.coin_holdings(::centipede::Player::two)};
      }
      return obs;
   }

   observation_type tiny_repr(const world_state_type&) const { return observation_type{}; }

   ////////////////////////////////
   /// API: histories           ///
   ////////////////////////////////

   /**
    * chronological move sequence attributed to its movers. Nothing is masked (perfect
    * information): the private history coincides with the public/open ones up to the
    * nullopt-wrapping convention of hidden entries (there are none).
    */
   [[nodiscard]] std::vector< PlayerInformedType< std::optional< action_variant_type > > >
   private_history(Player /*player*/, const world_state_type& wstate) const
   {
      std::vector< PlayerInformedType< std::optional< action_variant_type > > > out;
      out.reserve(wstate.round() + size_t(wstate.terminal()));
      for(size_t r = 0; r < wstate.round(); ++r) {
         const auto mover = r % 2 == 0 ? Player::alex : Player::bob;
         out.emplace_back(action_variant_type{Move{false}}, mover);
      }
      if(wstate.terminal_cause() == TerminalCause::taken) {
         const auto mover = wstate.round() % 2 == 0 ? Player::alex : Player::bob;
         out.emplace_back(action_variant_type{Move{true}}, mover);
      }
      out.shrink_to_fit();
      return out;
   }

   /// identical to the private history: every move is public knowledge
   [[nodiscard]] std::vector< PlayerInformedType< std::optional< action_variant_type > > >
   public_history(const world_state_type& wstate) const
   {
      return private_history(Player::alex, wstate);
   }

   [[nodiscard]] std::vector< PlayerInformedType< action_variant_type > > open_history(
      const world_state_type& wstate
   ) const
   {
      std::vector< PlayerInformedType< action_variant_type > > out;
      out.reserve(wstate.round() + size_t(wstate.terminal()));
      for(size_t r = 0; r < wstate.round(); ++r) {
         const auto mover = r % 2 == 0 ? Player::alex : Player::bob;
         out.emplace_back(action_variant_type{Move{false}}, mover);
      }
      if(wstate.terminal_cause() == TerminalCause::taken) {
         const auto mover = wstate.round() % 2 == 0 ? Player::alex : Player::bob;
         out.emplace_back(action_variant_type{Move{true}}, mover);
      }
      out.shrink_to_fit();
      return out;
   }

   Config m_config{};
};

}  // namespace nor::games::centipede

namespace common {

template <>
inline std::string to_string(const nor::games::centipede::Observation& value)
{
   std::string out;
   if(value.moved_by.has_value()) {
      out += fmt::format("moved:{},", common::to_string(*value.moved_by));
   }
   if(value.move.has_value()) {
      out += fmt::format("{},", value.move->take ? "take" : "push");
   }
   if(value.next_round.has_value()) {
      out += fmt::format("round:{},", *value.next_round);
   }
   if(value.next_active.has_value()) {
      out += fmt::format("next:{},", common::to_string(*value.next_active));
   }
   if(value.terminal_cause.has_value()) {
      out += fmt::format("terminal:{},", common::to_string(*value.terminal_cause));
   }
   if(value.coins.has_value()) {
      out += fmt::format("coins:({},{})", value.coins->first, value.coins->second);
   }
   if(out.empty()) {
      return "-";
   }
   out.pop_back();
   return out;
}

}  // namespace common

COMMON_ENABLE_PRINT(nor::games::centipede, Observation);

namespace std {

template < typename StateType >
   requires common::
      is_any_v< StateType, nor::games::centipede::Publicstate, nor::games::centipede::Infostate >
   struct hash< StateType > {
   size_t operator()(const StateType& state) const noexcept { return state.hash(); }
};

}  // namespace std

#endif  // NOR_CENTIPEDE_ENVIRONMENT_HPP
