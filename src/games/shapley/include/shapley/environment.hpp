
#ifndef NOR_SHAPLEY_ENVIRONMENT_HPP
#define NOR_SHAPLEY_ENVIRONMENT_HPP

#include <optional>
#include <utility>
#include <vector>

#include "common/common.hpp"
#include "nor/fosg_states.hpp"
#include "nor/fosg_traits.hpp"
#include "nor/game_defs.hpp"
#include "nor/utils/player_informed_type.hpp"
#include "shapley/state.hpp"
#include "shapley/utils.hpp"

namespace nor::games::shapley {

using namespace ::shapley;

inline auto to_shapley_player(const nor::Player& player)
{
   return static_cast< ::shapley::Player >(player);
}
inline auto to_nor_player(const ::shapley::Player& player)
{
   return static_cast< nor::Player >(player);
}

/**
 * @brief Compact observation of Shapley's game.
 *
 * Information structure (commit-commit with hidden values):
 * - commit phases: the fact that someone committed is public, the strategy stays hidden;
 * - the committer privately echoes his fresh play;
 * - at the fused resolve both plays become public together with the terminal announcement.
 */
struct Observation {
   /// a commitment happened by this player; its value stays hidden (commit phases)
   std::optional< nor::Player > committed_by{};
   /// the observer's own freshly committed play (private observation of the committer only)
   std::optional< Play > own_play{};
   /// both plays of the resolved joint move in (one, two) order (public, resolve transition only)
   std::optional< std::pair< Play, Play > > revealed_plays{};

   friend bool operator==(const Observation&, const Observation&) = default;
};

}  // namespace nor::games::shapley

/// has to be visible before the DefaultPublicstate/DefaultInfostate bases below, whose
/// 'observation' concept requirement probes the availability of this specialization
namespace std {

template <>
struct hash< nor::games::shapley::Observation > {
   size_t operator()(const nor::games::shapley::Observation& obs) const noexcept
   {
      namespace sh = nor::games::shapley;
      size_t seed = 0;
      if(obs.committed_by.has_value()) {
         common::hash_combine(seed, std::hash< int >{}(1 + int(*obs.committed_by)));
      }
      if(obs.own_play.has_value()) {
         common::hash_combine(
            seed, std::hash< unsigned >{}(11u + unsigned(obs.own_play->strategy))
         );
      }
      if(obs.revealed_plays.has_value()) {
         common::hash_combine(
            seed,
            std::hash< unsigned >{}(
               31u + 4u * unsigned(obs.revealed_plays->first.strategy)
               + unsigned(obs.revealed_plays->second.strategy)
            )
         );
      }
      return seed;
   }
};

}  // namespace std

namespace nor::games::shapley {

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
 * @brief The FOSG environment adapter of Shapley's game.
 *
 * The simultaneous normal-form move is sequentialized CommitP1 -> CommitP2 with the deterministic
 * resolve fused into the CommitP2 application (see State's Phase documentation). REWARD MODEL:
 * GENERAL-SUM -- reward(player) returns that player's OWN canonical bimatrix payoff, not a
 * zero-sum-centered difference. Exploitability-style normalized metrics are therefore NOT
 * meaningful for this game; use nash_conv(..., constant_sum=false) (general-sum-safe sum of
 * per-player best-response improvements) instead.
 */
class Environment {
  public:
   // nor fosg typedefs
   using world_state_type = State;
   using info_state_type = Infostate;
   using public_state_type = Publicstate;
   using action_type = Play;
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

   ///////////////////////////////////
   /// API: transitions            ///
   ///////////////////////////////////

   [[nodiscard]] std::vector< action_type > actions(Player player, const world_state_type& wstate)
      const
   {
      return wstate.actions(to_shapley_player(player));
   }

   void transition(world_state_type& worldstate, const action_type& action) const
   {
      worldstate.apply_action(action);
   }

   void transition(world_state_type& worldstate, action_type&& action) const
   {
      worldstate.apply_action(std::move(action));
   }

   [[nodiscard]] world_state_type initial_world_state() const { return world_state_type{}; }

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

   /// alex is player one, bob is player two; each receives his own bimatrix entry (general-sum)
   static double reward(Player player, const world_state_type& wstate)
   {
      return wstate.payoff(to_shapley_player(player));
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
      // only the committer echoes his play
      if(to_shapley_player(observer) != wstate.active_player()) {
         return observation_type{};
      }
      return observation_type{.own_play = action};
   }

   observation_type public_observation(
      const world_state_type& wstate,
      const action_type& /*action*/,
      const world_state_type& next_wstate
   ) const
   {
      // the fused resolve happens INSIDE the commit_p2 application, so both plays are
      // only guaranteed to be present on the POST-transition state; keying the
      // announcement off terminality of 'next_wstate' keeps every edge query well-
      // defined (pre-edge commit_p2 states still hide player two's unmade play)
      if(next_wstate.terminal()) {
         return observation_type{
            .revealed_plays = std::pair{
               *next_wstate.committed_play(::shapley::Player::one),
               *next_wstate.committed_play(::shapley::Player::two)}};
      }
      switch(wstate.phase()) {
         case Phase::commit_p1: {
            // commitment event is public knowledge while the play's value is withheld until
            // the fused resolve
            return observation_type{.committed_by = Player::alex};
         }
         case Phase::commit_p2: {
            break;
         }
      }
      return observation_type{};
   }

   observation_type tiny_repr(const world_state_type&) const { return observation_type{}; }

   ////////////////////////////////
   /// API: histories           ///
   ////////////////////////////////

   /**
    * chronological sequence of commitments masked to what `player` can observe: his/her own play
    * in full, the opponent's hidden (nullopt). Two entries total (CommitP1, CommitP2-with-fused-
    * resolve).
    */
   [[nodiscard]] std::vector< PlayerInformedType< std::optional< action_variant_type > > >
   private_history(Player player, const world_state_type& wstate) const
   {
      const auto self = to_shapley_player(player);
      std::vector< PlayerInformedType< std::optional< action_variant_type > > > out;
      out.reserve(2);
      out.emplace_back(
         self == ::shapley::Player::one ? std::optional< action_variant_type >{action_variant_type{
            *wstate.committed_play(::shapley::Player::one)}}
                                        : std::nullopt,
         Player::alex
      );
      out.emplace_back(
         self == ::shapley::Player::two ? std::optional< action_variant_type >{action_variant_type{
            *wstate.committed_play(::shapley::Player::two)}}
                                        : std::nullopt,
         Player::bob
      );
      out.shrink_to_fit();
      return out;
   }

   /// chronological sequence with every commitment masked out (the public view)
   [[nodiscard]] std::vector< PlayerInformedType< std::optional< action_variant_type > > >
   public_history(const world_state_type& wstate) const
   {
      std::vector< PlayerInformedType< std::optional< action_variant_type > > > out;
      out.reserve(2);
      out.emplace_back(std::nullopt, Player::alex);
      out.emplace_back(std::nullopt, Player::bob);
      out.shrink_to_fit();
      return out;
   }

   /// the fully open history in which even the hidden commitments are revealed
   [[nodiscard]] std::vector< PlayerInformedType< action_variant_type > > open_history(
      const world_state_type& wstate
   ) const
   {
      std::vector< PlayerInformedType< action_variant_type > > out;
      out.reserve(2);
      out.emplace_back(
         action_variant_type{*wstate.committed_play(::shapley::Player::one)}, Player::alex
      );
      out.emplace_back(
         action_variant_type{*wstate.committed_play(::shapley::Player::two)}, Player::bob
      );
      out.shrink_to_fit();
      return out;
   }
};

}  // namespace nor::games::shapley

namespace common {

template <>
inline std::string to_string(const nor::games::shapley::Observation& value)
{
   std::string out;
   if(value.committed_by.has_value()) {
      out += fmt::format("committed:{},", common::to_string(*value.committed_by));
   }
   if(value.own_play.has_value()) {
      out += fmt::format("own_play:{},", common::to_string(*value.own_play));
   }
   if(value.revealed_plays.has_value()) {
      out += fmt::format(
         "plays:({},{})",
         common::to_string(value.revealed_plays->first),
         common::to_string(value.revealed_plays->second)
      );
   }
   if(out.empty()) {
      return "-";
   }
   out.pop_back();
   return out;
}

}  // namespace common

COMMON_ENABLE_PRINT(nor::games::shapley, Observation);

namespace std {

template < typename StateType >
   requires common::
      is_any_v< StateType, nor::games::shapley::Publicstate, nor::games::shapley::Infostate >
   struct hash< StateType > {
   size_t operator()(const StateType& state) const noexcept { return state.hash(); }
};

}  // namespace std

#endif  // NOR_SHAPLEY_ENVIRONMENT_HPP
