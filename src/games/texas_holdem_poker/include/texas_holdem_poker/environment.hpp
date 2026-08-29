
#ifndef NOR_TEXAS_HOLDEM_POKER_ENVIRONMENT_HPP
#define NOR_TEXAS_HOLDEM_POKER_ENVIRONMENT_HPP

#include <algorithm>
#include <ranges>
#include <string>
#include <vector>

#include "common/common.hpp"
#include "nor/fosg_states.hpp"
#include "nor/fosg_traits.hpp"
#include "nor/game_defs.hpp"
#include "nor/utils/player_informed_type.hpp"
#include "texas_holdem_poker/state.hpp"
#include "texas_holdem_poker/utils.hpp"

namespace nor::games::texholdem {

using namespace ::texholdem;

inline auto to_texholdem_player(const nor::Player& player)
{
   return static_cast< ::texholdem::Player >(player);
}
inline auto to_nor_player(const ::texholdem::Player& player)
{
   return static_cast< nor::Player >(player);
}

/**
 * @brief Compact observation type of texas hold'em.
 *
 * An observation is either
 * - a betting action taken by some player (public),
 * - a revealed community card (public),
 * - the mere identity of a player receiving a face-down card (public), or
 * - the identity of a privately received hole card (private observation of its recipient).
 *
 * All payloads are optional and at most one of them is set per instance.
 */
struct Observation {
   /// a betting action that was publicly played
   std::optional< Action > action{};
   /// a revealed community card or a privately received hole card
   std::optional< Card > card{};
   /// set iff a face-down card was dealt to the denoted player while its identity stays hidden
   /// (note: an explicit qualification is required here since the unqualified 'Player' resolves
   ///  to nor::Player within this nested namespace)
   std::optional< ::texholdem::Player > hidden_deal_to{};

   friend bool operator==(const Observation&, const Observation&) = default;
};

}  // namespace nor::games::texholdem

/// has to be visible before the DefaultPublicstate/DefaultInfostate bases below, whose
/// 'observation' concept requirement probes the availability of this specialization
namespace std {

template <>
struct hash< nor::games::texholdem::Observation > {
   size_t operator()(const nor::games::texholdem::Observation& obs) const noexcept
   {
      size_t seed = 0;
      if(obs.action.has_value()) {
         common::hash_combine(seed, std::hash< nor::games::texholdem::Action >{}(*obs.action));
      }
      if(obs.card.has_value()) {
         common::hash_combine(seed, std::hash< nor::games::texholdem::Card >{}(*obs.card));
      }
      if(obs.hidden_deal_to.has_value()) {
         common::hash_combine(seed, std::hash< int >{}(int(*obs.hidden_deal_to)));
      }
      return seed;
   }
};

}  // namespace std

namespace nor::games::texholdem {

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
 * @brief The FOSG environment adapter of no-limit texas hold'em wrapping texholdem::State.
 */
class Environment {
  public:
   // nor fosg typedefs
   using world_state_type = State;
   using info_state_type = Infostate;
   using public_state_type = Publicstate;
   using action_type = Action;
   using chance_outcome_type = Card;
   using observation_type = Observation;
   using action_variant_type = action_variant_type_generator_t< action_type, chance_outcome_type >;
   // nor fosg traits
   static constexpr size_t max_player_count() { return ::texholdem::max_player_count; }
   static constexpr size_t player_count() { return std::dynamic_extent; }
   static constexpr bool serialized() { return true; }
   static constexpr bool unrolled() { return true; }
   static constexpr Stochasticity stochasticity() { return Stochasticity::choice; }

  public:
   Environment() = default;
   explicit Environment(PokerConfig config) : m_config(std::move(config)) {}

   [[nodiscard]] const PokerConfig& config() const { return m_config; }

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

   static std::vector< Player > players(const world_state_type& wstate)
   {
      auto seated_players = wstate.players();
      std::vector< Player > out;
      out.reserve(seated_players.size());
      std::ranges::copy(
         seated_players | std::views::transform([](auto p) { return to_nor_player(p); }),
         std::back_inserter(out)
      );
      return out;
   }

   static bool is_terminal(const world_state_type& wstate) { return wstate.is_terminal(); }

   static bool is_partaking(const world_state_type& wstate, Player player)
   {
      size_t seat = as_int(to_texholdem_player(player));
      return seat < wstate.config().n_players and not wstate.has_folded(::texholdem::Player(seat));
   }

   static double reward(Player player, const world_state_type& wstate)
   {
      return wstate.payoff(to_texholdem_player(player));
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
      // betting actions carry no additional private information
      return observation_type{};
   }

   observation_type private_observation(
      Player observer,
      const world_state_type& wstate,
      const chance_outcome_type& outcome,
      const world_state_type& /*next_wstate*/
   ) const
   {
      if(wstate.holes_dealt() >= hole_cards_per_player * wstate.config().n_players) {
         // board cards are fully public
         return observation_type{};
      }
      if(next_deal_recipient(wstate) == to_texholdem_player(observer)) {
         // the observer receives this hole card himself
         return observation_type{.card = outcome};
      }
      return observation_type{};
   }

   observation_type public_observation(
      const world_state_type& wstate,
      const action_type& action,
      const world_state_type& /*next_wstate*/
   ) const
   {
      (void) wstate;
      return observation_type{.action = action};
   }

   observation_type public_observation(
      const world_state_type& wstate,
      const chance_outcome_type& outcome,
      const world_state_type& /*next_wstate*/
   ) const
   {
      if(wstate.holes_dealt() < hole_cards_per_player * wstate.config().n_players) {
         // a face-down hole card: only the identity of its recipient is public knowledge
         return observation_type{.hidden_deal_to = next_deal_recipient(wstate)};
      }
      // a revealed community card
      return observation_type{.card = outcome};
   }

   ////////////////////////////////
   /// API: histories           ///
   ////////////////////////////////

   /// chronological sequence of hole card deals + street-grouped board reveals + betting actions.
   /// Each entry is masked to what `player` can observe (nullopt for hidden entries).
   [[nodiscard]] std::vector< PlayerInformedType< std::optional< action_variant_type > > >
   private_history(Player player, const world_state_type& wstate) const
   {
      std::vector< PlayerInformedType< std::optional< action_variant_type > > > out;
      out.reserve(deals_per_history(wstate) + wstate.actions_history().size());
      append_card_deals(out, wstate, [&](size_t seat) {
         return to_texholdem_player(player) == ::texholdem::Player(seat);
      });
      append_betting_actions(out, wstate, [](const ActionRecord&) { return true; });
      out.shrink_to_fit();
      return out;
   }

   [[nodiscard]] std::vector< PlayerInformedType< std::optional< action_variant_type > > >
   public_history(const world_state_type& wstate) const
   {
      std::vector< PlayerInformedType< std::optional< action_variant_type > > > out;
      out.reserve(deals_per_history(wstate) + wstate.actions_history().size());
      append_card_deals(out, wstate, [](size_t) { return false; });
      append_betting_actions(out, wstate, [](const ActionRecord&) { return true; });
      out.shrink_to_fit();
      return out;
   }

   [[nodiscard]] std::vector< PlayerInformedType< action_variant_type > > open_history(
      const world_state_type& wstate
   ) const
   {
      std::vector< PlayerInformedType< action_variant_type > > out;
      out.reserve(deals_per_history(wstate) + wstate.actions_history().size());
      // hole cards in their deal order
      for(auto [seat, round, opt_card] : hole_deal_sequence(wstate)) {
         (void) seat;
         (void) round;
         if(not opt_card.has_value()) {
            break;
         }
         out.emplace_back(action_variant_type{*opt_card}, Player::chance);
      }
      // then the chronologically interleaved board reveals + betting actions
      for(auto street : {Street::preflop, Street::flop, Street::turn, Street::river}) {
         if(street != Street::preflop) {
            for(auto opt_card : board_reveals(wstate, street)) {
               if(opt_card.has_value()) {
                  out.emplace_back(action_variant_type{*opt_card}, Player::chance);
               }
            }
         }
         for(const auto& record : wstate.actions_history()) {
            if(record.street == street) {
               out.emplace_back(action_variant_type{record.action}, to_nor_player(record.player));
            }
         }
      }
      out.shrink_to_fit();
      return out;
   }

  private:
   /// the seat which receives the pending hole card (only meaningful while holes are being dealt)
   static ::texholdem::Player next_deal_recipient(const world_state_type& wstate)
   {
      return wstate.next_deal_recipient();
   }

   /// number of hole deals + board reveals in a full hand (upper bound for reservations)
   static size_t deals_per_history(const world_state_type& wstate)
   {
      return hole_cards_per_player * wstate.config().n_players + community_card_count;
   }

   /// the (seat, round-index, card) triples of hole deals in chronological deal order; `card`
   /// is empty once the corresponding deal has not happened yet
   static std::vector< std::tuple< size_t, size_t, std::optional< Card > > > hole_deal_sequence(
      const world_state_type& wstate
   )
   {
      std::vector< std::tuple< size_t, size_t, std::optional< Card > > > out;
      size_t n_deals = hole_cards_per_player * wstate.config().n_players;
      out.reserve(n_deals);
      for(size_t i = 0; i < n_deals; ++i) {
         size_t round = i / wstate.config().n_players;
         size_t seat = size_t(
            (wstate.small_blind_pos() + int(i % wstate.config().n_players))
            % int(wstate.config().n_players)
         );
         out.emplace_back(seat, round, wstate.hole_card(::texholdem::Player(seat), round));
      }
      return out;
   }

   /// the community cards revealed at the given street (empty optionals for pending ones)
   static std::vector< std::optional< Card > >
   board_reveals(const world_state_type& wstate, Street street)
   {
      auto all_community = wstate.community_cards();
      size_t start = 0, end = 0;
      switch(street) {
         case Street::flop:
            start = 0;
            end = std::min(size_t(3), all_community.size());
            break;
         case Street::turn:
            start = 3;
            end = std::min(size_t(4), all_community.size());
            break;
         case Street::river:
            start = 4;
            end = std::min(size_t(5), all_community.size());
            break;
         default: break;
      }
      std::vector< std::optional< Card > > out;
      out.reserve(end - start);
      for(size_t i = start; i < end; ++i) {
         out.emplace_back(all_community[i]);
      }
      return out;
   }

   template < typename Container, typename VisibleFor >
   static void
   append_card_deals(Container& out, const world_state_type& wstate, VisibleFor visible_for)
   {
      for(auto [seat, round, opt_card] : hole_deal_sequence(wstate)) {
         if(not opt_card.has_value()) {
            break;  // this card has not been dealt yet --> no further history exists
         }
         if(visible_for(seat)) {
            out.emplace_back(action_variant_type{*opt_card}, Player::chance);
         } else {
            out.emplace_back(std::nullopt, Player::chance);
         }
      }
   }

   template < typename Container, typename Included >
   static void
   append_betting_actions(Container& out, const world_state_type& wstate, Included included)
   {
      for(const auto& record : wstate.actions_history()) {
         if(included(record)) {
            out.emplace_back(action_variant_type{record.action}, to_nor_player(record.player));
         } else {
            out.emplace_back(std::nullopt, to_nor_player(record.player));
         }
      }
   }

  private:
   PokerConfig m_config{};
};

}  // namespace nor::games::texholdem

namespace common {

template <>
inline std::string to_string(const nor::games::texholdem::Observation& value)
{
   if(value.action.has_value()) {
      return common::to_string(*value.action);
   }
   if(value.card.has_value()) {
      return common::to_string(*value.card);
   }
   if(value.hidden_deal_to.has_value()) {
      return fmt::format("{}:?", common::to_string(*value.hidden_deal_to));
   }
   return "-";
}

}  // namespace common

COMMON_ENABLE_PRINT(nor::games::texholdem, Observation);

namespace std {

template < typename StateType >
   requires common::
      is_any_v< StateType, nor::games::texholdem::Publicstate, nor::games::texholdem::Infostate >
   struct hash< StateType > {
   size_t operator()(const StateType& state) const noexcept { return state.hash(); }
};

}  // namespace std

#endif  // NOR_TEXAS_HOLDEM_POKER_ENVIRONMENT_HPP
