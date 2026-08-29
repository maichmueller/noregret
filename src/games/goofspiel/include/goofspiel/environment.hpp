
#ifndef NOR_GOOFSPIEL_ENVIRONMENT_HPP
#define NOR_GOOFSPIEL_ENVIRONMENT_HPP

#include <optional>
#include <ranges>
#include <string>
#include <utility>
#include <vector>

#include "common/common.hpp"
#include "goofspiel/state.hpp"
#include "goofspiel/utils.hpp"
#include "nor/fosg_states.hpp"
#include "nor/fosg_traits.hpp"
#include "nor/game_defs.hpp"
#include "nor/utils/player_informed_type.hpp"

namespace nor::games::goofspiel {

using namespace ::goofspiel;

inline auto to_goofspiel_player(const nor::Player& player)
{
   return static_cast< ::goofspiel::Player >(player);
}
inline auto to_nor_player(const ::goofspiel::Player& player)
{
   return static_cast< nor::Player >(player);
}

/**
 * @brief Compact observation type of goofspiel.
 *
 * An observation is exactly one of
 * - a freshly revealed prize card (public chance observation),
 * - the fact that some player committed a bid whose value is withheld (public),
 * - one's own freshly committed bid (private observation of its committer),
 * - the pair of revealed bids together with the round outcome (public; public-reveal mode only,
 *   where bids are published right after each resolve), or
 * - the mere outcome of a resolved round (public; limited-information mode).
 *
 * All payloads are optional and at most the semantically matching subset is set per instance.
 */
struct Observation {
   /// the prize card revealed by chance
   std::optional< uint8_t > prize{};
   /// set iff this transition was a bid commitment by the denoted player while its value stays
   /// hidden (limited-information mode always, public-reveal mode until the resolve)
   std::optional< ::goofspiel::Player > committed_by{};
   /// the observer's own fresh bid (private observation of the committing player only)
   std::optional< Bid > own_bid{};
   /// both bids of the resolved round in (one, two) order (public-reveal mode only)
   std::optional< std::pair< Bid, Bid > > revealed_bids{};
   /// the announced result of the resolved round
   std::optional< RoundOutcome > outcome{};

   friend bool operator==(const Observation&, const Observation&) = default;
};

}  // namespace nor::games::goofspiel

/// has to be visible before the DefaultPublicstate/DefaultInfostate bases below, whose
/// 'observation' concept requirement probes the availability of this specialization
namespace std {

template <>
struct hash< nor::games::goofspiel::Observation > {
   size_t operator()(const nor::games::goofspiel::Observation& obs) const noexcept
   {
      size_t seed = 0;
      if(obs.prize.has_value()) {
         common::hash_combine(seed, std::hash< unsigned >{}(1u + unsigned(*obs.prize)));
      }
      if(obs.committed_by.has_value()) {
         common::hash_combine(seed, std::hash< int >{}(7 + int(*obs.committed_by)));
      }
      if(obs.own_bid.has_value()) {
         common::hash_combine(seed, std::hash< unsigned >{}(13u + unsigned(obs.own_bid->card)));
      }
      if(obs.revealed_bids.has_value()) {
         common::hash_combine(
            seed,
            std::hash< unsigned >{}(
               29u + 16u * unsigned(obs.revealed_bids->first.card)
               + unsigned(obs.revealed_bids->second.card)
            )
         );
      }
      if(obs.outcome.has_value()) {
         common::hash_combine(seed, std::hash< int >{}(57 + int(*obs.outcome)));
      }
      return seed;
   }
};

}  // namespace std

namespace nor::games::goofspiel {

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
 * @brief The FOSG environment adapter of goofspiel wrapping goofspiel::State.
 */
class Environment {
  public:
   // nor fosg typedefs
   using world_state_type = State;
   using info_state_type = Infostate;
   using public_state_type = Publicstate;
   using action_type = Bid;
   using chance_outcome_type = PrizeCard;
   using observation_type = Observation;
   using action_variant_type = action_variant_type_generator_t< action_type, chance_outcome_type >;
   // nor fosg traits
   static constexpr size_t max_player_count() { return 2; }
   static constexpr size_t player_count() { return 2; }
   static constexpr bool serialized() { return true; }
   static constexpr bool unrolled() { return true; }
   static constexpr Stochasticity stochasticity() { return Stochasticity::choice; }

  public:
   Environment() = default;
   explicit Environment(GoofspielConfig config) : m_config(std::move(config)) {}

   [[nodiscard]] const GoofspielConfig& config() const { return m_config; }

   ///////////////////////////////////
   /// API: transitions and chance ///
   ///////////////////////////////////

   std::vector< action_type > actions(Player, const world_state_type& wstate) const
   {
      return wstate.actions();
   }

   inline std::vector< chance_outcome_type > chance_actions(const world_state_type& wstate) const
   {
      return wstate.chance_actions();
   }

   inline double
   chance_probability(const world_state_type& wstate, const chance_outcome_type& outcome) const
   {
      return wstate.chance_probability(outcome);
   }

   /// B7 (PCS) chance classification: EVERY goofspiel chance event is public.
   /// Prize reveals are observed by both players, and the resolve confirmation
   /// (sentinel PrizeCard{0}) publishes the round result (revealed bids and/or
   /// the announced winner). The outcome argument is irrelevant for the class.
   static bool public_chance_event(const world_state_type&, const chance_outcome_type&) noexcept
   {
      return true;
   }

   void transition(world_state_type& worldstate, const action_type& action) const
   {
      worldstate.apply_action(action);
   }

   void transition(world_state_type& worldstate, const chance_outcome_type& outcome) const
   {
      worldstate.apply_action(outcome);
   }

   [[nodiscard]] world_state_type initial_world_state() const { return world_state_type(m_config); }

   /////////////////////////////////
   /// API: players and payoffs  ///
   /////////////////////////////////

   [[nodiscard]] Player active_player(const world_state_type& wstate) const
   {
      return to_nor_player(wstate.active_player());
   }

   /// NOTE: the chance player is part of the returned roster (mirroring kuhn/leduc) so that the
   /// CFR machinery tracks its reach-probability contributions through the prize reveals
   static std::vector< Player > players(const world_state_type&)
   {
      return {Player::chance, Player::alex, Player::bob};
   }

   static bool is_terminal(const world_state_type& wstate) { return wstate.is_terminal(); }

   static constexpr bool is_partaking(const world_state_type&, Player player)
   {
      return player != Player::chance;
   }

   /// zero-sum across players: u(one) = score(one) - score(two) and vice versa
   static double reward(Player player, const world_state_type& wstate)
   {
      return wstate.payoff(to_goofspiel_player(player));
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
      // only the committer learns his own fresh bid
      if(to_goofspiel_player(observer) == wstate.active_player()) {
         return observation_type{.own_bid = action};
      }
      return observation_type{};
   }

   observation_type private_observation(
      Player /*observer*/,
      const world_state_type& /*wstate*/,
      const chance_outcome_type& /*outcome*/,
      const world_state_type& /*next_wstate*/
   ) const
   {
      // prize reveals are public and the resolve confirmation carries no private information
      return observation_type{};
   }

   observation_type public_observation(
      const world_state_type& wstate,
      const action_type& /*action*/,
      const world_state_type& /*next_wstate*/
   ) const
   {
      // a commitment event is public knowledge while the bid's value is withheld until resolve
      return observation_type{.committed_by = wstate.active_player()};
   }

   observation_type public_observation(
      const world_state_type& wstate,
      const chance_outcome_type& outcome,
      const world_state_type& /*next_wstate*/
   ) const
   {
      switch(wstate.phase()) {
         case Phase::prize_reveal: return observation_type{.prize = outcome.value};
         case Phase::resolve: {
            const auto bid_one = *wstate.committed_bid(::goofspiel::Player::one);
            const auto bid_two = *wstate.committed_bid(::goofspiel::Player::two);
            const auto winner = bid_one.card > bid_two.card   ? RoundOutcome::p1_wins
                                : bid_two.card > bid_one.card ? RoundOutcome::p2_wins
                                                              : RoundOutcome::tie;
            if(m_config.imp_info) {
               // limited information: only the outcome is announced
               return observation_type{.outcome = winner};
            }
            // public reveal: both bids become public right after the resolve
            return observation_type{
               .revealed_bids = std::pair{bid_one, bid_two}, .outcome = winner};
         }
         default: return observation_type{};
      }
   }

   ////////////////////////////////
   /// API: histories           ///
   ////////////////////////////////

   /// chronological sequence of prize reveals, commitments and resolves masked to what `player`
   /// can observe (nullopt payloads for hidden entries)
   [[nodiscard]] std::vector< PlayerInformedType< std::optional< action_variant_type > > >
   private_history(Player player, const world_state_type& wstate) const;

   [[nodiscard]] std::vector< PlayerInformedType< std::optional< action_variant_type > > >
   public_history(const world_state_type& wstate) const;

   [[nodiscard]] std::vector< PlayerInformedType< action_variant_type > > open_history(
      const world_state_type& wstate
   ) const;

  private:
   GoofspielConfig m_config{};
};

inline std::vector< PlayerInformedType< std::optional< Environment::action_variant_type > > >
Environment::private_history(Player player, const world_state_type& wstate) const
{
   const auto observer = to_goofspiel_player(player);
   std::vector< PlayerInformedType< std::optional< action_variant_type > > > out;
   const auto& records = wstate.rounds();
   // completed rounds
   for(size_t r = 0; r < wstate.round(); ++r) {
      const auto& record = records[r];
      // the prize reveal is public knowledge
      out.emplace_back(action_variant_type{PrizeCard{uint8_t(record.prize)}}, Player::chance);
      // bids are visible to their owner always and to everybody else from the resolve onwards in
      // public-reveal mode
      out.emplace_back(
         observer == ::goofspiel::Player::one or not m_config.imp_info
            ? std::optional< action_variant_type >{action_variant_type{
               Bid{uint8_t(record.bid_one)}}}
            : std::nullopt,
         nor::Player::alex
      );
      out.emplace_back(
         observer == ::goofspiel::Player::two or not m_config.imp_info
            ? std::optional< action_variant_type >{action_variant_type{
               Bid{uint8_t(record.bid_two)}}}
            : std::nullopt,
         nor::Player::bob
      );
      out.emplace_back(action_variant_type{PrizeCard{0}}, Player::chance);
   }
   // the ongoing partial round
   if(not wstate.is_terminal()) {
      if(wstate.phase() != Phase::prize_reveal) {
         out.emplace_back(
            action_variant_type{PrizeCard{uint8_t(records[wstate.round()].prize)}}, Player::chance
         );
      }
      if(auto bid = wstate.committed_bid(::goofspiel::Player::one); bid.has_value()) {
         out.emplace_back(
            observer == ::goofspiel::Player::one
               ? std::optional< action_variant_type >{action_variant_type{*bid}}
               : std::nullopt,
            nor::Player::alex
         );
      }
      if(auto bid = wstate.committed_bid(::goofspiel::Player::two); bid.has_value()) {
         out.emplace_back(
            observer == ::goofspiel::Player::two
               ? std::optional< action_variant_type >{action_variant_type{*bid}}
               : std::nullopt,
            nor::Player::bob
         );
      }
   }
   out.shrink_to_fit();
   return out;
}

inline std::vector< PlayerInformedType< std::optional< Environment::action_variant_type > > >
Environment::public_history(const world_state_type& wstate) const
{
   std::vector< PlayerInformedType< std::optional< action_variant_type > > > out;
   const auto& records = wstate.rounds();
   for(size_t r = 0; r < wstate.round(); ++r) {
      const auto& record = records[r];
      out.emplace_back(action_variant_type{PrizeCard{uint8_t(record.prize)}}, Player::chance);
      out.emplace_back(  // only public knowledge in public-reveal mode where resolves published it
         not m_config.imp_info ? std::optional< action_variant_type >{action_variant_type{
            Bid{uint8_t(record.bid_one)}}}
                               : std::nullopt,
         nor::Player::alex
      );
      out.emplace_back(
         not m_config.imp_info ? std::optional< action_variant_type >{action_variant_type{
            Bid{uint8_t(record.bid_two)}}}
                               : std::nullopt,
         nor::Player::bob
      );
      out.emplace_back(action_variant_type{PrizeCard{0}}, Player::chance);
   }
   if(not wstate.is_terminal()) {
      if(wstate.phase() != Phase::prize_reveal) {
         out.emplace_back(
            action_variant_type{PrizeCard{uint8_t(records[wstate.round()].prize)}}, Player::chance
         );
      }
      if(wstate.committed_bid(::goofspiel::Player::one).has_value()) {
         out.emplace_back(std::nullopt, nor::Player::alex);
      }
      if(wstate.committed_bid(::goofspiel::Player::two).has_value()) {
         out.emplace_back(std::nullopt, nor::Player::bob);
      }
   }
   out.shrink_to_fit();
   return out;
}

inline std::vector< PlayerInformedType< Environment::action_variant_type > >
Environment::open_history(const world_state_type& wstate) const
{
   std::vector< PlayerInformedType< action_variant_type > > out;
   const auto& records = wstate.rounds();
   for(size_t r = 0; r < wstate.round(); ++r) {
      const auto& record = records[r];
      out.emplace_back(action_variant_type{PrizeCard{uint8_t(record.prize)}}, Player::chance);
      out.emplace_back(action_variant_type{Bid{uint8_t(record.bid_one)}}, nor::Player::alex);
      out.emplace_back(action_variant_type{Bid{uint8_t(record.bid_two)}}, nor::Player::bob);
      out.emplace_back(action_variant_type{PrizeCard{0}}, Player::chance);
   }
   if(not wstate.is_terminal()) {
      if(wstate.phase() != Phase::prize_reveal) {
         out.emplace_back(
            action_variant_type{PrizeCard{uint8_t(records[wstate.round()].prize)}}, Player::chance
         );
      }
      if(auto bid = wstate.committed_bid(::goofspiel::Player::one); bid.has_value()) {
         out.emplace_back(action_variant_type{*bid}, nor::Player::alex);
      }
      if(auto bid = wstate.committed_bid(::goofspiel::Player::two); bid.has_value()) {
         out.emplace_back(action_variant_type{*bid}, nor::Player::bob);
      }
   }
   out.shrink_to_fit();
   return out;
}

}  // namespace nor::games::goofspiel

namespace common {

template <>
inline std::string to_string(const nor::games::goofspiel::Observation& value)
{
   if(value.prize.has_value()) {
      return fmt::format("prize:{}", unsigned(*value.prize));
   }
   if(value.committed_by.has_value()) {
      return fmt::format("committed:{}", common::to_string(*value.committed_by));
   }
   if(value.own_bid.has_value()) {
      return fmt::format("own_bid:{}", unsigned(value.own_bid->card));
   }
   if(value.revealed_bids.has_value()) {
      return fmt::format(
         "bids:({},{})",
         unsigned(value.revealed_bids->first.card),
         unsigned(value.revealed_bids->second.card)
      );
   }
   if(value.outcome.has_value()) {
      return fmt::format("outcome:{}", common::to_string(*value.outcome));
   }
   return "-";
}

}  // namespace common

COMMON_ENABLE_PRINT(nor::games::goofspiel, Observation);

namespace std {

template < typename StateType >
   requires common::
      is_any_v< StateType, nor::games::goofspiel::Publicstate, nor::games::goofspiel::Infostate >
   struct hash< StateType > {
   size_t operator()(const StateType& state) const noexcept { return state.hash(); }
};

}  // namespace std

#endif  // NOR_GOOFSPIEL_ENVIRONMENT_HPP
