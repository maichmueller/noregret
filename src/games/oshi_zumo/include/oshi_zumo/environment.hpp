
#ifndef NOR_OSHI_ZUMO_ENVIRONMENT_HPP
#define NOR_OSHI_ZUMO_ENVIRONMENT_HPP

#include <optional>
#include <string>
#include <utility>
#include <vector>

#include "common/common.hpp"
#include "nor/fosg_states.hpp"
#include "nor/fosg_traits.hpp"
#include "nor/game_defs.hpp"
#include "nor/utils/player_informed_type.hpp"
#include "oshi_zumo/state.hpp"
#include "oshi_zumo/utils.hpp"

namespace nor::games::oshi_zumo {

using namespace ::oshi_zumo;

inline auto to_oz_player(const nor::Player& player)
{
   return static_cast< ::oshi_zumo::Player >(player);
}
inline auto to_nor_player(const ::oshi_zumo::Player& player)
{
   return static_cast< nor::Player >(player);
}

/**
 * @brief Compact observation of Oshi-Zumo.
 *
 * The game is perfect-information between rounds (OpenSpiel marks it
 * GameType::Information::kPerfectInformation and renders the full board for everybody,
 * oshi_zumo.cc:33/152-160) -- the only hidden information is the CURRENT round's uncommitted /
 * committed-but-unresolved bid. Information structure:
 * - commit phases: the fact that someone committed is public, the amount stays hidden;
 * - the committer privately echoes his fresh bid;
 * - at the fused resolve both bids become public together with the resulting wrestler position,
 *   each player additionally privately echoes his post-resolve purse;
 * - the terminal cause is announced publicly.
 */
struct Observation {
   /// a commitment happened by this player; its value stays hidden (commit phases)
   std::optional< nor::Player > committed_by{};
   /// the observer's own freshly committed bid (private observation of the committer only)
   std::optional< Bid > own_bid{};
   /// observer-private purse AFTER a resolution (both purses are publicly derivable from the
   /// revealed bids, so this is an echo rather than exclusive information)
   std::optional< uint32_t > own_purse{};
   /// both bids of the resolved round in (one, two) order (public)
   std::optional< std::pair< Bid, Bid > > revealed_bids{};
   /// post-resolve wrestler position (public)
   std::optional< int16_t > wrestler_position{};
   /// public announcement of how the game ended (terminal transitions only); payoffs themselves
   /// are read off the world state via reward()
   std::optional< TerminalCause > terminal_cause{};

   friend bool operator==(const Observation&, const Observation&) = default;
};

}  // namespace nor::games::oshi_zumo

/// has to be visible before the DefaultPublicstate/DefaultInfostate bases below, whose
/// 'observation' concept requirement probes the availability of this specialization
namespace std {

template <>
struct hash< nor::games::oshi_zumo::Observation > {
   size_t operator()(const nor::games::oshi_zumo::Observation& obs) const noexcept
   {
      namespace oz = nor::games::oshi_zumo;
      size_t seed = 0;
      if(obs.committed_by.has_value()) {
         common::hash_combine(seed, std::hash< int >{}(1 + int(*obs.committed_by)));
      }
      if(obs.own_bid.has_value()) {
         common::hash_combine(seed, std::hash< unsigned >{}(17u + unsigned(obs.own_bid->amount)));
      }
      if(obs.own_purse.has_value()) {
         common::hash_combine(seed, std::hash< unsigned >{}(53u + unsigned(*obs.own_purse)));
      }
      if(obs.revealed_bids.has_value()) {
         common::hash_combine(
            seed,
            std::hash< unsigned long >{}(
               97ul + 4096ul * unsigned long(obs.revealed_bids->first.amount)
               + unsigned long(obs.revealed_bids->second.amount)
            )
         );
      }
      if(obs.wrestler_position.has_value()) {
         common::hash_combine(seed, std::hash< int >{}(131 + int(*obs.wrestler_position)));
      }
      if(obs.terminal_cause.has_value()) {
         common::hash_combine(seed, std::hash< unsigned >{}(211u + unsigned(*obs.terminal_cause)));
      }
      return seed;
   }
};

}  // namespace std

namespace nor::games::oshi_zumo {

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
 * @brief The FOSG environment adapter of (deterministic) Oshi-Zumo.
 *
 * The simultaneous bids of each round are sequentialized CommitP1 -> CommitP2 with the
 * deterministic resolve fused into the CommitP2 application (see State's Phase documentation).
 */
class Environment {
  public:
   // nor fosg typedefs
   using world_state_type = State;
   using info_state_type = Infostate;
   using public_state_type = Publicstate;
   using action_type = Bid;
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
      return wstate.actions(to_oz_player(player));
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

   /// alex is player one, bob is player two
   static double reward(Player player, const world_state_type& wstate)
   {
      return wstate.payoff(to_oz_player(player));
   }

   ////////////////////////////////
   /// API: observations        ///
   ////////////////////////////////

   observation_type private_observation(
      Player observer,
      const world_state_type& wstate,
      const action_type& action,
      const world_state_type& next_wstate
   ) const
   {
      const auto self = to_oz_player(observer);
      switch(wstate.phase()) {
         case Phase::commit_p1: {
            // only the committer echoes his bid; the resolve has not happened yet
            if(self != ::oshi_zumo::Player::one) {
               return observation_type{};
            }
            return observation_type{.own_bid = action};
         }
         case Phase::commit_p2: {
            // fused resolve: each side privately echoes its own bid and post-resolve purse
            return observation_type{
               .own_bid = self == ::oshi_zumo::Player::one
                             ? *wstate.committed_bid(::oshi_zumo::Player::one)
                             : action,
               .own_purse = next_wstate.coins(self)};
         }
      }
      return observation_type{};
   }

   observation_type public_observation(
      const world_state_type& wstate,
      const action_type& action,
      const world_state_type& next_wstate
   ) const
   {
      switch(wstate.phase()) {
         case Phase::commit_p1: {
            // commitment event is public knowledge while the bid's value is withheld until resolve
            return observation_type{.committed_by = Player::alex};
         }
         case Phase::commit_p2: {
            // fused resolve publishes both bids and the resulting wrestler position
            observation_type obs{
               .revealed_bids = std::pair{*wstate.committed_bid(::oshi_zumo::Player::one), action},
               .wrestler_position = next_wstate.wrestler_pos()};
            if(next_wstate.terminal()) {
               obs.terminal_cause = next_wstate.terminal_cause();
            }
            return obs;
         }
      }
      return observation_type{};
   }

   observation_type tiny_repr(const world_state_type&) const { return observation_type{}; }

   ////////////////////////////////
   /// API: histories           ///
   ////////////////////////////////

   /**
    * chronological sequence of commitments masked to what `player` can observe: his/her own
    * bids in full, everything else hidden (nullopt). Two entries per resolved round (CommitP1,
    * CommitP2-with-fused-resolve), plus one pending CommitP1 entry while a resolution is awaited.
    */
   [[nodiscard]] std::vector< PlayerInformedType< std::optional< action_variant_type > > >
   private_history(Player player, const world_state_type& wstate) const
   {
      const auto self = to_oz_player(player);
      std::vector< PlayerInformedType< std::optional< action_variant_type > > > out;
      out.reserve(2 * wstate.round() + size_t(wstate.phase() == Phase::commit_p2));
      for(size_t r = 0; r < wstate.round(); ++r) {
         const auto& record = wstate.rounds()[r];
         out.emplace_back(
            self == ::oshi_zumo::Player::one
               ? std::optional< action_variant_type >{action_variant_type{Bid{record.bid_one}}}
               : std::nullopt,
            Player::alex
         );
         out.emplace_back(
            self == ::oshi_zumo::Player::two
               ? std::optional< action_variant_type >{action_variant_type{Bid{record.bid_two}}}
               : std::nullopt,
            Player::bob
         );
      }
      if(not wstate.terminal() && wstate.phase() == Phase::commit_p2) {
         out.emplace_back(
            self == ::oshi_zumo::Player::one
               ? std::optional< action_variant_type >{action_variant_type{
                  *wstate.committed_bid(::oshi_zumo::Player::one)}}
               : std::nullopt,
            Player::alex
         );
      }
      out.shrink_to_fit();
      return out;
   }

   /// chronological sequence with every commitment masked out (the public view)
   [[nodiscard]] std::vector< PlayerInformedType< std::optional< action_variant_type > > >
   public_history(const world_state_type& wstate) const
   {
      std::vector< PlayerInformedType< std::optional< action_variant_type > > > out;
      const size_t n_rounds = wstate.round();
      out.reserve(2 * n_rounds + size_t(wstate.phase() == Phase::commit_p2));
      for(size_t r = 0; r < n_rounds; ++r) {
         out.emplace_back(std::nullopt, Player::alex);
         out.emplace_back(std::nullopt, Player::bob);
      }
      if(not wstate.terminal() && wstate.phase() == Phase::commit_p2) {
         out.emplace_back(std::nullopt, Player::alex);
      }
      out.shrink_to_fit();
      return out;
   }

   /// the fully open history in which even the hidden commitments are revealed
   [[nodiscard]] std::vector< PlayerInformedType< action_variant_type > > open_history(
      const world_state_type& wstate
   ) const
   {
      std::vector< PlayerInformedType< action_variant_type > > out;
      out.reserve(2 * wstate.round() + size_t(wstate.phase() == Phase::commit_p2));
      for(size_t r = 0; r < wstate.round(); ++r) {
         const auto& record = wstate.rounds()[r];
         out.emplace_back(action_variant_type{Bid{record.bid_one}}, Player::alex);
         out.emplace_back(action_variant_type{Bid{record.bid_two}}, Player::bob);
      }
      if(not wstate.terminal() && wstate.phase() == Phase::commit_p2) {
         out.emplace_back(
            action_variant_type{*wstate.committed_bid(::oshi_zumo::Player::one)}, Player::alex
         );
      }
      out.shrink_to_fit();
      return out;
   }

  private:
   Config m_config{};
};

}  // namespace nor::games::oshi_zumo

namespace common {

template <>
inline std::string to_string(const nor::games::oshi_zumo::Observation& value)
{
   namespace oz = nor::games::oshi_zumo;
   std::string out;
   if(value.committed_by.has_value()) {
      out += fmt::format("committed:{},", common::to_string(*value.committed_by));
   }
   if(value.own_bid.has_value()) {
      out += fmt::format("own_bid:{},", value.own_bid->amount);
   }
   if(value.own_purse.has_value()) {
      out += fmt::format("purse:{},", *value.own_purse);
   }
   if(value.revealed_bids.has_value()) {
      out += fmt::format(
         "bids:({},{})", value.revealed_bids->first.amount, value.revealed_bids->second.amount
      );
   }
   if(value.wrestler_position.has_value()) {
      out += fmt::format("pos:{},", *value.wrestler_position);
   }
   if(value.terminal_cause.has_value()) {
      out += fmt::format("terminal:{},", common::to_string(*value.terminal_cause));
   }
   if(out.empty()) {
      return "-";
   }
   out.pop_back();
   return out;
}

}  // namespace common

COMMON_ENABLE_PRINT(nor::games::oshi_zumo, Observation);

namespace nor {

template <>
struct fosg_traits< games::oshi_zumo::Infostate > {
   using observation_type = nor::games::oshi_zumo::Observation;
};

template <>
struct fosg_traits< games::oshi_zumo::Environment > {
   using world_state_type = nor::games::oshi_zumo::State;
   using info_state_type = nor::games::oshi_zumo::Infostate;
   using public_state_type = nor::games::oshi_zumo::Publicstate;
   using action_type = nor::games::oshi_zumo::Bid;
   using chance_outcome_type = std::monostate;
   using observation_type = nor::games::oshi_zumo::Observation;
};

}  // namespace nor

namespace std {

template < typename StateType >
   requires common::
      is_any_v< StateType, nor::games::oshi_zumo::Publicstate, nor::games::oshi_zumo::Infostate >
   struct hash< StateType > {
   size_t operator()(const StateType& state) const noexcept { return state.hash(); }
};

}  // namespace std

#endif  // NOR_OSHI_ZUMO_ENVIRONMENT_HPP
