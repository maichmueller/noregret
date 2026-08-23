
#ifndef NOR_LIARS_DICE_ENVIRONMENT_HPP
#define NOR_LIARS_DICE_ENVIRONMENT_HPP

#include <optional>
#include <range/v3/all.hpp>
#include <string>
#include <vector>

#include "common/common.hpp"
#include "liars_dice/state.hpp"
#include "liars_dice/utils.hpp"
#include "nor/fosg_states.hpp"
#include "nor/fosg_traits.hpp"
#include "nor/game_defs.hpp"

namespace nor::games::liars_dice {

using namespace ::liars_dice;

inline auto to_liars_dice_player(const nor::Player& player)
{
   return static_cast< ::liars_dice::Player >(player);
}
inline auto to_nor_player(const ::liars_dice::Player& player)
{
   return static_cast< nor::Player >(player);
}

/// the public showdown payload revealed upon a challenge
struct Reveal {
   uint8_t die_one;
   uint8_t die_two;
   Outcome outcome;

   friend bool operator==(const Reveal&, const Reveal&) = default;
};

/**
 * @brief Compact observation type of liar's dice.
 *
 * An observation is either
 * - a privately received own die roll (only visible to its recipient),
 * - the mere identity of the player receiving a hidden roll (public),
 * - a publicly announced bid,
 * - or the public reveal of both dice together with the challenge resolution.
 *
 * All payloads are optional and at most one of them is set per instance.
 */
struct Observation {
   /// the privately received own roll of the observer (empty for everyone else)
   std::optional< Roll > roll{};
   /// set iff a face-down roll was dealt to the denoted player while its face stays hidden
   std::optional< ::liars_dice::Player > hidden_roll_to{};
   /// a bid announcement that was publicly played
   std::optional< Bid > bid{};
   /// the public reveal of both dice and the challenge outcome
   std::optional< Reveal > reveal{};

   friend bool operator==(const Observation&, const Observation&) = default;
};

}  // namespace nor::games::liars_dice

/// has to be visible before the DefaultPublicstate/DefaultInfostate bases below, whose
/// 'observation' concept requirement probes the availability of this specialization
namespace std {

template <>
struct hash< nor::games::liars_dice::Observation > {
   size_t operator()(const nor::games::liars_dice::Observation& obs) const noexcept
   {
      size_t seed = 0;
      if(obs.roll.has_value()) {
         common::hash_combine(seed, std::hash< nor::games::liars_dice::Roll >{}(*obs.roll));
      }
      if(obs.hidden_roll_to.has_value()) {
         common::hash_combine(seed, std::hash< int >{}(int(*obs.hidden_roll_to)));
      }
      if(obs.bid.has_value()) {
         common::hash_combine(seed, std::hash< nor::games::liars_dice::Bid >{}(*obs.bid));
      }
      if(obs.reveal.has_value()) {
         const auto& rev = *obs.reveal;
         common::hash_combine(seed, std::hash< int >{}(int(rev.die_one)));
         common::hash_combine(seed, std::hash< int >{}(int(rev.die_two)));
         common::hash_combine(seed, std::hash< int >{}(int(rev.outcome)));
      }
      return seed;
   }
};

template <>
struct hash< nor::games::liars_dice::Reveal > {
   size_t operator()(const nor::games::liars_dice::Reveal& rev) const noexcept
   {
      size_t seed = 0;
      common::hash_combine(seed, std::hash< int >{}(int(rev.die_one)));
      common::hash_combine(seed, std::hash< int >{}(int(rev.die_two)));
      common::hash_combine(seed, std::hash< int >{}(int(rev.outcome)));
      return seed;
   }
};

}  // namespace std

namespace nor::games::liars_dice {

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
 * @brief The FOSG environment adapter of single-round liar's dice wrapping liars_dice::State.
 */
class Environment {
  public:
   // nor fosg typedefs
   using world_state_type = State;
   using info_state_type = Infostate;
   using public_state_type = Publicstate;
   using action_type = Action;
   using chance_outcome_type = Roll;
   using observation_type = Observation;
   using action_variant_type = action_variant_type_generator_t< action_type, chance_outcome_type >;
   // nor fosg traits
   static constexpr size_t max_player_count() { return ::liars_dice::max_player_count; }
   static constexpr size_t player_count() { return ::liars_dice::max_player_count; }
   static constexpr bool serialized() { return true; }
   static constexpr bool unrolled() { return true; }
   static constexpr Stochasticity stochasticity() { return Stochasticity::choice; }

  public:
   Environment() = default;
   explicit Environment(DiceConfig config) : m_config(config) {}

   [[nodiscard]] const DiceConfig& config() const { return m_config; }

   ///////////////////////////////////
   /// API: transitions and chance ///
   ///////////////////////////////////

   [[nodiscard]] std::vector< action_type >
   actions(Player /*player*/, const world_state_type& wstate) const
   {
      return wstate.actions();
   }

   [[nodiscard]] std::vector< chance_outcome_type > chance_actions(const world_state_type& wstate
   ) const
   {
      return wstate.chance_actions();
   }

   [[nodiscard]] double
   chance_probability(const world_state_type& wstate, const chance_outcome_type& outcome) const
   {
      return wstate.chance_probability(outcome);
   }

   template < typename ActionT >
      requires common::is_any_v< ActionT, action_type, chance_outcome_type >
   void transition(world_state_type& worldstate, const ActionT& action) const
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

   [[nodiscard]] std::vector< Player > players(const world_state_type& wstate) const
   {
      auto seated_players = wstate.players();
      std::vector< Player > out;
      out.reserve(seated_players.size());
      ranges::copy(
         seated_players | ranges::views::transform([](auto p) { return to_nor_player(p); }),
         std::back_inserter(out)
      );
      return out;
   }

   [[nodiscard]] static bool is_terminal(const world_state_type& wstate)
   {
      return wstate.is_terminal();
   }

   static constexpr bool is_partaking(const world_state_type&, Player) { return true; }

   [[nodiscard]] static double reward(Player player, const world_state_type& wstate)
   {
      return wstate.payoff(to_liars_dice_player(player));
   }

   ////////////////////////////////
   /// API: observations        ///
   ////////////////////////////////

   [[nodiscard]] observation_type private_observation(
      Player /*observer*/,
      const world_state_type& /*wstate*/,
      const action_type& /*action*/,
      const world_state_type& /*next_wstate*/
   ) const
   {
      // bids and challenges carry no additional private information
      return observation_type{};
   }

   [[nodiscard]] observation_type private_observation(
      Player observer,
      const world_state_type& /*wstate*/,
      const chance_outcome_type& outcome,
      const world_state_type& /*next_wstate*/
   ) const
   {
      if(outcome.player == to_liars_dice_player(observer)) {
         // the observer receives this roll himself
         return observation_type{.roll = outcome};
      }
      return observation_type{};
   }

   [[nodiscard]] observation_type public_observation(
      const world_state_type& /*wstate*/,
      const action_type& action,
      const world_state_type& next_wstate
   ) const
   {
      if(action.kind == ActionType::challenge) {
         // a challenge resolves the hand: both dice are revealed publicly
         return observation_type{
            .reveal = Reveal{
               .die_one = next_wstate.die(::liars_dice::Player::one).value_or(uint8_t(0)),
               .die_two = next_wstate.die(::liars_dice::Player::two).value_or(uint8_t(0)),
               .outcome = next_wstate.challenge_outcome().value_or(Outcome::challenger_wins)}};
      }
      // every bid announcement is fully public
      return observation_type{.bid = action.bid};
   }

   [[nodiscard]] observation_type public_observation(
      const world_state_type& /*wstate*/,
      const chance_outcome_type& outcome,
      const world_state_type& /*next_wstate*/
   ) const
   {
      // a hidden roll: only the identity of its recipient is public knowledge
      return observation_type{.hidden_roll_to = outcome.player};
   }

   ////////////////////////////////
   /// API: histories           ///
   ////////////////////////////////

   /// chronological sequence of both die rolls + betting actions. Each entry is masked to what
   /// `player` can observe (nullopt for hidden entries).
   [[nodiscard]] std::vector< PlayerInformedType< std::optional< action_variant_type > > >
   private_history(Player player, const world_state_type& wstate) const
   {
      std::vector< PlayerInformedType< std::optional< action_variant_type > > > out;
      out.reserve(rolls_per_history + wstate.actions_history().size());
      append_rolls(out, wstate, [&](auto seat) { return to_liars_dice_player(player) == seat; });
      append_actions(out, wstate);
      out.shrink_to_fit();
      return out;
   }

   [[nodiscard]] std::vector< PlayerInformedType< std::optional< action_variant_type > > >
   public_history(const world_state_type& wstate) const
   {
      std::vector< PlayerInformedType< std::optional< action_variant_type > > > out;
      out.reserve(rolls_per_history + wstate.actions_history().size());
      append_rolls(out, wstate, []([[maybe_unused]] auto seat) { return false; });
      append_actions(out, wstate);
      out.shrink_to_fit();
      return out;
   }

   [[nodiscard]] std::vector< PlayerInformedType< action_variant_type > > open_history(
      const world_state_type& wstate
   ) const
   {
      std::vector< PlayerInformedType< action_variant_type > > out;
      out.reserve(rolls_per_history + wstate.actions_history().size());
      for(auto roller : {::liars_dice::Player::one, ::liars_dice::Player::two}) {
         if(auto face = wstate.die(roller)) {
            out.emplace_back(action_variant_type{Roll{roller, *face}}, Player::chance);
         }
      }
      for(const auto& record : wstate.actions_history()) {
         out.emplace_back(action_variant_type{record.action}, to_nor_player(record.player));
      }
      out.shrink_to_fit();
      return out;
   }

  private:
   static constexpr size_t rolls_per_history = 2;

   template < typename Container, typename VisibleFor >
   static void append_rolls(Container& out, const world_state_type& wstate, VisibleFor visible_for)
   {
      for(auto roller : {::liars_dice::Player::one, ::liars_dice::Player::two}) {
         auto face_opt = wstate.die(roller);
         if(not face_opt.has_value()) {
            break;  // this die has not been rolled yet --> no further history exists
         }
         if(visible_for(roller)) {
            out.emplace_back(action_variant_type{Roll{roller, *face_opt}}, Player::chance);
         } else {
            out.emplace_back(std::nullopt, Player::chance);
         }
      }
   }

   template < typename Container >
   static void append_actions(Container& out, const world_state_type& wstate)
   {
      for(const auto& record : wstate.actions_history()) {
         out.emplace_back(action_variant_type{record.action}, to_nor_player(record.player));
      }
   }

  private:
   DiceConfig m_config{};
};

}  // namespace nor::games::liars_dice

namespace common {

template <>
inline std::string to_string(const nor::games::liars_dice::Observation& value)
{
   using liars_dice_ns = nor::games::liars_dice;
   if(value.roll.has_value()) {
      return fmt::format("{}", common::to_string(*value.roll));
   }
   if(value.hidden_roll_to.has_value()) {
      return fmt::format("{}:?", common::to_string(*value.hidden_roll_to));
   }
   if(value.bid.has_value()) {
      return common::to_string(*value.bid);
   }
   if(value.reveal.has_value()) {
      return fmt::format(
         "reveal {}|{}:{}",
         int(value.reveal->die_one),
         int(value.reveal->die_two),
         value.reveal->outcome == liars_dice_ns::Outcome::bidder_wins ? "bidder" : "challenger"
      );
   }
   return "-";
}

}  // namespace common

COMMON_ENABLE_PRINT(nor::games::liars_dice, Observation);

namespace nor {

template <>
struct fosg_traits< games::liars_dice::Infostate > {
   using observation_type = nor::games::liars_dice::Observation;
};

template <>
struct fosg_traits< games::liars_dice::Environment > {
   using world_state_type = nor::games::liars_dice::State;
   using info_state_type = nor::games::liars_dice::Infostate;
   using public_state_type = nor::games::liars_dice::Publicstate;
   using action_type = nor::games::liars_dice::Action;
   using chance_outcome_type = nor::games::liars_dice::Roll;
   using observation_type = nor::games::liars_dice::Observation;
};

}  // namespace nor

namespace std {

template < typename StateType >
   requires common::
      is_any_v< StateType, nor::games::liars_dice::Publicstate, nor::games::liars_dice::Infostate >
   struct hash< StateType > {
   size_t operator()(const StateType& state) const noexcept { return state.hash(); }
};

}  // namespace std

#endif  // NOR_LIARS_DICE_ENVIRONMENT_HPP
