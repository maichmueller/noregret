
#ifndef NOR_THREE_PLAYER_GOOFSPIEL_ENVIRONMENT_HPP
#define NOR_THREE_PLAYER_GOOFSPIEL_ENVIRONMENT_HPP

#include <optional>
#include <ranges>
#include <string>
#include <utility>
#include <vector>

#include "common/common.hpp"
#include "nor/fosg_states.hpp"
#include "nor/fosg_traits.hpp"
#include "nor/game_defs.hpp"
#include "nor/utils/player_informed_type.hpp"
#include "three_player_goofspiel/state.hpp"
#include "three_player_goofspiel/utils.hpp"

namespace nor::games::three_player_goofspiel {

using namespace ::three_player_goofspiel;

inline auto to_tpg_player(const nor::Player& player)
{
   return static_cast< ::three_player_goofspiel::Player >(player);
}
inline auto to_nor_player(const ::three_player_goofspiel::Player& player)
{
   return static_cast< nor::Player >(player);
}

/**
 * @brief Compact observation type of three-player goofspiel.
 *
 * An observation is exactly one of
 * - a freshly revealed prize card (public chance observation),
 * - the fact that some player committed a bid whose value is withheld (public),
 * - one's own freshly committed bid (private observation of its committer),
 * - the triple of revealed bids together with the round winner (public; public-reveal mode
 *   only), or
 * - the mere outcome of a resolved round (public; limited-information mode).
 *
 * Deal observations in split-deck mode are carried as private own-mask entries.
 *
 * All payloads are optional and at most the semantically matching subset is set per instance.
 */
struct Observation {
   /// the prize card revealed by chance
   std::optional< uint8_t > prize{};
   /// set iff this transition was a bid commitment by the denoted player while its value stays
   /// hidden (limited-information mode always, public-reveal mode until the resolve)
   std::optional< ::three_player_goofspiel::Player > committed_by{};
   /// the observer's own fresh bid (private observation of the committing player only)
   std::optional< Bid > own_bid{};
   /// the observer's freshly dealt hand mask (private observation during a split deal)
   std::optional< uint16_t > dealt_hand{};
   /// all bids of the resolved round in (alex, bob, cedric) order (public-reveal mode only)
   std::optional< std::array< Bid, 3 > > revealed_bids{};
   /// the announced result of the resolved round
   std::optional< RoundWinner > outcome{};

   friend bool operator==(const Observation&, const Observation&) = default;
};

}  // namespace nor::games::three_player_goofspiel

/// has to be visible before the DefaultPublicstate/DefaultInfostate bases below, whose
/// 'observation' concept requirement probes the availability of this specialization
namespace std {

template <>
struct hash< nor::games::three_player_goofspiel::Observation > {
   size_t operator()(const nor::games::three_player_goofspiel::Observation& obs) const noexcept
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
      if(obs.dealt_hand.has_value()) {
         common::hash_combine(seed, std::hash< unsigned >{}(17u + unsigned(*obs.dealt_hand)));
      }
      if(obs.revealed_bids.has_value()) {
         const auto& bids = *obs.revealed_bids;
         common::hash_combine(
            seed,
            std::hash< unsigned >{}(
               29u + 256u * unsigned(bids[0].card) + 16u * unsigned(bids[1].card)
               + unsigned(bids[2].card)
            )
         );
      }
      if(obs.outcome.has_value()) {
         common::hash_combine(seed, std::hash< int >{}(71 + int(*obs.outcome)));
      }
      return seed;
   }
};

}  // namespace std

namespace nor::games::three_player_goofspiel {

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
 * @brief The FOSG environment adapter of the three-player goofspiel wrapping
 * three_player_goofspiel::State.
 */
class Environment {
  public:
   // nor fosg typedefs
   using world_state_type = State;
   using info_state_type = Infostate;
   using public_state_type = Publicstate;
   using action_type = Bid;
   using chance_outcome_type = ChanceOutcome;
   using observation_type = Observation;
   using action_variant_type = action_variant_type_generator_t< action_type, chance_outcome_type >;
   // nor fosg traits
   static constexpr size_t max_player_count() { return 3; }
   static constexpr size_t player_count() { return 3; }
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

   /// B7 (PCS) chance classification: every chance event except the concealed deal is public to
   /// all participants.
   static bool
   public_chance_event(const world_state_type&, const chance_outcome_type& outcome) noexcept
   {
      return outcome.kind != ChanceOutcome::Kind::deal;
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

   /// NOTE: the chance player is part of the returned roster (mirroring goofspiel/kuhn/leduc) so
   /// that the CFR machinery tracks its reach-probability contributions through the reveals
   static std::vector< Player > players(const world_state_type&)
   {
      return {Player::chance, Player::alex, Player::bob, Player::cedric};
   }

   static bool is_terminal(const world_state_type& wstate) { return wstate.is_terminal(); }

   static constexpr bool is_partaking(const world_state_type&, Player player)
   {
      return player != Player::chance;
   }

   /// zero-sum across seats with the equal-split team payoff rule implemented in State::payoff
   static double reward(Player player, const world_state_type& wstate)
   {
      return wstate.payoff(to_tpg_player(player));
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
      if(to_tpg_player(observer) == wstate.active_player()) {
         return observation_type{.own_bid = action};
      }
      return observation_type{};
   }

   observation_type private_observation(
      Player observer,
      const world_state_type& /*wstate*/,
      const chance_outcome_type& outcome,
      const world_state_type& /*next_wstate*/
   ) const
   {
      switch(outcome.kind) {
         case ChanceOutcome::Kind::deal: {
            // each team member only learns his OWN dealt half; cedric learns nothing privately
            auto full = uint16_t(0);
            for(size_t v = 1; v <= m_config.deck_size; ++v) {
               full |= ::three_player_goofspiel::card_bit(uint8_t(v));
            }
            if(observer == Player::alex) {
               return observation_type{.dealt_hand = outcome.mask_a};
            }
            if(observer == Player::bob) {
               // bob's half is the complement; carried explicitly so his infostate key differs
               // from cedric's blank entry
               return observation_type{.dealt_hand = uint16_t(full & ~outcome.mask_a)};
            }
            return observation_type{};
         }
         default:
            // prize reveals and the resolve confirmation carry no private information
            return observation_type{};
      }
   }

   observation_type public_observation(
      const world_state_type& wstate,
      const action_type& /*action*/,
      const world_state_type& next_wstate
   ) const
   {
      (void) next_wstate;
      // a commitment event is public knowledge while the bid's value is withheld until resolve
      return observation_type{.committed_by = wstate.active_player()};
   }

   observation_type public_observation(
      const world_state_type& wstate,
      const chance_outcome_type& outcome,
      const world_state_type& /*next_wstate*/
   ) const
   {
      switch(outcome.kind) {
         case ChanceOutcome::Kind::deal: return observation_type{};
         case ChanceOutcome::Kind::prize: return observation_type{.prize = outcome.value};
         case ChanceOutcome::Kind::confirm: {
            const auto bid_alex = *wstate.committed_bid(::three_player_goofspiel::Player::alex);
            const auto bid_bob = *wstate.committed_bid(::three_player_goofspiel::Player::bob);
            const auto bid_cedric = *wstate.committed_bid(::three_player_goofspiel::Player::cedric);
            const auto winner = resolve_winner(bid_alex, bid_bob, bid_cedric);
            if(m_config.imp_info) {
               // limited information: only the outcome is announced
               return observation_type{.outcome = winner};
            }
            // public reveal: all three bids become public right after the resolve
            return observation_type{
               .revealed_bids = std::array{bid_alex, bid_bob, bid_cedric}, .outcome = winner};
         }
      }
      return observation_type{};
   }

   static RoundWinner resolve_winner(const Bid& bid_alex, const Bid& bid_bob, const Bid& bid_cedric)
   {
      const auto best = std::max({bid_alex.card, bid_bob.card, bid_cedric.card});
      const auto winners = size_t(bid_alex.card == best) + size_t(bid_bob.card == best)
                           + size_t(bid_cedric.card == best);
      if(winners != 1) {
         return RoundWinner::tie;
      }
      if(bid_alex.card == best) {
         return RoundWinner::alex;
      }
      return bid_bob.card == best ? RoundWinner::bob : RoundWinner::cedric;
   }

   ////////////////////////////////
   /// API: histories           ///
   ////////////////////////////////

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

}  // namespace nor::games::three_player_goofspiel

namespace nor::games::three_player_goofspiel {

inline std::vector< PlayerInformedType< std::optional< Environment::action_variant_type > > >
Environment::private_history(Player player, const world_state_type& wstate) const
{
   const auto observer = to_tpg_player(player);
   std::vector< PlayerInformedType< std::optional< action_variant_type > > > out;
   const auto& records = wstate.rounds();
   if(m_config.split_half_deal) {
      out.emplace_back(
         observer == ::three_player_goofspiel::Player::alex
            ? std::optional< action_variant_type >{action_variant_type{ChanceOutcome{
               .kind = ChanceOutcome::Kind::deal,
               .value = 0,
               .mask_a = wstate.dealt_halves().first}}}
            : std::nullopt,
         Player::chance
      );
      out.emplace_back(
         observer == ::three_player_goofspiel::Player::bob
            ? std::optional< action_variant_type >{action_variant_type{ChanceOutcome{
               .kind = ChanceOutcome::Kind::deal,
               .value = 0,
               .mask_a = wstate.dealt_halves().second}}}
            : std::nullopt,
         Player::chance
      );
   }
   // completed rounds
   for(size_t r = 0; r < wstate.round(); ++r) {
      const auto& record = records[r];
      out.emplace_back(
         action_variant_type{ChanceOutcome{
            .kind = ChanceOutcome::Kind::prize, .value = uint8_t(record.prize), .mask_a = 0}},
         Player::chance
      );
      const auto maybe_visible_bid = [&](int8_t bid_card, ::three_player_goofspiel::Player owner) {
         return observer == owner or not m_config.imp_info
                   ? std::optional< action_variant_type >{action_variant_type{
                      Bid{uint8_t(bid_card)}}}
                   : std::nullopt;
      };
      out.emplace_back(
         maybe_visible_bid(record.bid_alex, ::three_player_goofspiel::Player::alex), Player::alex
      );
      out.emplace_back(
         maybe_visible_bid(record.bid_bob, ::three_player_goofspiel::Player::bob), Player::bob
      );
      out.emplace_back(
         maybe_visible_bid(record.bid_cedric, ::three_player_goofspiel::Player::cedric),
         Player::cedric
      );
      out.emplace_back(
         action_variant_type{
            ChanceOutcome{.kind = ChanceOutcome::Kind::confirm, .value = 0, .mask_a = 0}},
         Player::chance
      );
   }
   // the ongoing partial round
   if(not wstate.is_terminal()) {
      if(wstate.phase() != Phase::deal and wstate.phase() != Phase::prize_reveal) {
         out.emplace_back(
            action_variant_type{ChanceOutcome{
               .kind = ChanceOutcome::Kind::prize,
               .value = uint8_t(records[wstate.round()].prize),
               .mask_a = 0}},
            Player::chance
         );
      }
      const auto pending_visible_bid = [&](::three_player_goofspiel::Player owner
                                       ) -> std::optional< action_variant_type > {
         auto bid = wstate.committed_bid(owner);
         if(not bid.has_value()) {
            return std::nullopt;
         }
         return observer == owner or not m_config.imp_info
                   ? std::optional< action_variant_type >{action_variant_type{*bid}}
                   : std::nullopt;
      };
      if(auto entry = pending_visible_bid(::three_player_goofspiel::Player::alex);
         entry.has_value()) {
         out.emplace_back(std::move(*entry), Player::alex);
      }
      if(auto entry = pending_visible_bid(::three_player_goofspiel::Player::bob);
         entry.has_value()) {
         out.emplace_back(std::move(*entry), Player::bob);
      }
      if(auto entry = pending_visible_bid(::three_player_goofspiel::Player::cedric);
         entry.has_value()) {
         out.emplace_back(std::move(*entry), Player::cedric);
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
   if(m_config.split_half_deal) {
      out.emplace_back(std::nullopt, Player::chance);
   }
   for(size_t r = 0; r < wstate.round(); ++r) {
      const auto& record = records[r];
      out.emplace_back(
         action_variant_type{ChanceOutcome{
            .kind = ChanceOutcome::Kind::prize, .value = uint8_t(record.prize), .mask_a = 0}},
         Player::chance
      );
      const auto revealed_or_hidden = [&](int8_t bid_card) {
         return not m_config.imp_info ? std::optional< action_variant_type >{action_variant_type{
                   Bid{uint8_t(bid_card)}}}
                                      : std::nullopt;
      };
      out.emplace_back(revealed_or_hidden(record.bid_alex), Player::alex);
      out.emplace_back(revealed_or_hidden(record.bid_bob), Player::bob);
      out.emplace_back(revealed_or_hidden(record.bid_cedric), Player::cedric);
      out.emplace_back(
         action_variant_type{
            ChanceOutcome{.kind = ChanceOutcome::Kind::confirm, .value = 0, .mask_a = 0}},
         Player::chance
      );
   }
   if(not wstate.is_terminal()) {
      if(wstate.phase() != Phase::deal and wstate.phase() != Phase::prize_reveal) {
         out.emplace_back(
            action_variant_type{ChanceOutcome{
               .kind = ChanceOutcome::Kind::prize,
               .value = uint8_t(records[wstate.round()].prize),
               .mask_a = 0}},
            Player::chance
         );
      }
      for(const auto owner :
          {::three_player_goofspiel::Player::alex,
           ::three_player_goofspiel::Player::bob,
           ::three_player_goofspiel::Player::cedric}) {
         if(wstate.committed_bid(owner).has_value()) {
            out.emplace_back(std::nullopt, to_nor_player(owner));
         }
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
   if(m_config.split_half_deal) {
      out.emplace_back(
         action_variant_type{ChanceOutcome{
            .kind = ChanceOutcome::Kind::deal, .value = 0, .mask_a = wstate.dealt_halves().first}},
         Player::chance
      );
   }
   for(size_t r = 0; r < wstate.round(); ++r) {
      const auto& record = records[r];
      out.emplace_back(
         action_variant_type{ChanceOutcome{
            .kind = ChanceOutcome::Kind::prize, .value = uint8_t(record.prize), .mask_a = 0}},
         Player::chance
      );
      out.emplace_back(action_variant_type{Bid{uint8_t(record.bid_alex)}}, Player::alex);
      out.emplace_back(action_variant_type{Bid{uint8_t(record.bid_bob)}}, Player::bob);
      out.emplace_back(action_variant_type{Bid{uint8_t(record.bid_cedric)}}, Player::cedric);
      out.emplace_back(
         action_variant_type{
            ChanceOutcome{.kind = ChanceOutcome::Kind::confirm, .value = 0, .mask_a = 0}},
         Player::chance
      );
   }
   if(not wstate.is_terminal()) {
      if(wstate.phase() != Phase::deal and wstate.phase() != Phase::prize_reveal) {
         out.emplace_back(
            action_variant_type{ChanceOutcome{
               .kind = ChanceOutcome::Kind::prize,
               .value = uint8_t(records[wstate.round()].prize),
               .mask_a = 0}},
            Player::chance
         );
      }
      const auto push_pending = [&](::three_player_goofspiel::Player owner) {
         if(auto bid = wstate.committed_bid(owner); bid.has_value()) {
            out.emplace_back(action_variant_type{*bid}, to_nor_player(owner));
         }
      };
      push_pending(::three_player_goofspiel::Player::alex);
      push_pending(::three_player_goofspiel::Player::bob);
      push_pending(::three_player_goofspiel::Player::cedric);
   }
   out.shrink_to_fit();
   return out;
}

}  // namespace nor::games::three_player_goofspiel

namespace common {

template <>
inline std::string to_string(const nor::games::three_player_goofspiel::Observation& value)
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
   if(value.dealt_hand.has_value()) {
      return fmt::format("dealt:{:x}", unsigned(*value.dealt_hand));
   }
   if(value.revealed_bids.has_value()) {
      return fmt::format(
         "bids:({},{},{})",
         unsigned((*value.revealed_bids)[0].card),
         unsigned((*value.revealed_bids)[1].card),
         unsigned((*value.revealed_bids)[2].card)
      );
   }
   if(value.outcome.has_value()) {
      return fmt::format("outcome:{}", common::to_string(*value.outcome));
   }
   return "-";
}

}  // namespace common

COMMON_ENABLE_PRINT(nor::games::three_player_goofspiel, Observation);

namespace nor {

template <>
struct fosg_traits< games::three_player_goofspiel::Infostate > {
   using observation_type = nor::games::three_player_goofspiel::Observation;
};

template <>
struct fosg_traits< games::three_player_goofspiel::Environment > {
   using world_state_type = nor::games::three_player_goofspiel::State;
   using info_state_type = nor::games::three_player_goofspiel::Infostate;
   using public_state_type = nor::games::three_player_goofspiel::Publicstate;
   using action_type = nor::games::three_player_goofspiel::Bid;
   using chance_outcome_type = nor::games::three_player_goofspiel::ChanceOutcome;
   using observation_type = nor::games::three_player_goofspiel::Observation;
};

}  // namespace nor

namespace std {

template < typename StateType >
   requires common::is_any_v<
      StateType,
      nor::games::three_player_goofspiel::Publicstate,
      nor::games::three_player_goofspiel::Infostate >
struct hash< StateType > {
   size_t operator()(const StateType& state) const noexcept { return state.hash(); }
};

}  // namespace std

#endif  // NOR_THREE_PLAYER_GOOFSPIEL_ENVIRONMENT_HPP
