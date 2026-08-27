
#ifndef NOR_SHERIFF_ENVIRONMENT_HPP
#define NOR_SHERIFF_ENVIRONMENT_HPP

#include <optional>
#include <string>
#include <utility>
#include <variant>
#include <vector>

#include "common/common.hpp"
#include "nor/fosg_states.hpp"
#include "nor/fosg_traits.hpp"
#include "nor/game_defs.hpp"
#include "nor/utils/player_informed_type.hpp"
#include "sheriff/state.hpp"
#include "sheriff/utils.hpp"

namespace nor::games::sheriff {

using namespace ::sheriff;

inline auto to_sheriff_player(const nor::Player& player)
{
   return static_cast< ::sheriff::Player >(player);
}
inline auto to_nor_player(const ::sheriff::Player& player)
{
   return static_cast< nor::Player >(player);
}

/**
 * @brief Compact observation of the Sheriff game.
 *
 * Information structure (Appendix F.1: all actions are public except the cargo selection):
 * - the loading EVENT is public, its size is a private echo to the smuggler only;
 * - bribe offers and sheriff responses are public with their full payload;
 * - an inspection terminal publicly reveals the cargo contents (the sheriff looks into the
 *   trunk), while a bribe-accepted terminal keeps the cargo hidden forever;
 * - terminal_cause is announced publicly on the final transition.
 */
struct Observation {
   enum class Kind : uint8_t { none = 0, load, offer, response };

   Kind kind = Kind::none;
   /// who caused this event
   ::nor::Player actor = ::nor::Player::unknown;
   /// for load events: private echo of the smuggler's own cargo (smuggler-only observations)
   std::optional< uint32_t > own_cargo{};
   /// for offer events: the proposed bribe amount (public)
   std::optional< uint32_t > bribe_offered{};
   /// for response events: whether the pending bribe was accepted (public)
   std::optional< bool > bribe_accepted{};
   /// public reveal of the cargo on inspection terminals (both kinds)
   std::optional< uint32_t > revealed_cargo{};
   /// public announcement of how the game ended (terminal transitions only)
   std::optional< ::sheriff::TerminalCause > terminal_cause{};

   friend bool operator==(const Observation&, const Observation&) = default;
};

}  // namespace nor::games::sheriff

/// has to be visible before the DefaultPublicstate/DefaultInfostate bases below, whose
/// 'observation' concept requirement probes the availability of this specialization
namespace std {

template <>
struct hash< nor::games::sheriff::Observation > {
   size_t operator()(const nor::games::sheriff::Observation& obs) const noexcept
   {
      size_t seed = 0;
      common::hash_combine(seed, std::hash< unsigned >{}(unsigned(obs.kind)));
      common::hash_combine(seed, std::hash< int >{}(int(obs.actor)));
      if(obs.own_cargo.has_value()) {
         common::hash_combine(seed, std::hash< unsigned >{}(17u + *obs.own_cargo));
      }
      if(obs.bribe_offered.has_value()) {
         common::hash_combine(seed, std::hash< unsigned >{}(53u + *obs.bribe_offered));
      }
      if(obs.bribe_accepted.has_value()) {
         common::hash_combine(seed, std::hash< bool >{}(*obs.bribe_accepted));
      }
      if(obs.revealed_cargo.has_value()) {
         common::hash_combine(seed, std::hash< unsigned >{}(97u + *obs.revealed_cargo));
      }
      if(obs.terminal_cause.has_value()) {
         common::hash_combine(seed, std::hash< unsigned >{}(211u + unsigned(*obs.terminal_cause)));
      }
      return seed;
   }
};

}  // namespace std

namespace nor::games::sheriff {

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
 * @brief The FOSG environment adapter of the Sheriff benchmark (Farina et al. 2019, App. F).
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

   [[nodiscard]] std::vector< action_type > actions(Player player, const world_state_type& wstate)
      const
   {
      return wstate.actions(to_sheriff_player(player));
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

   /// alex is the smuggler (player one), bob is the sheriff (player two)
   static double reward(Player player, const world_state_type& wstate)
   {
      return wstate.payoff(to_sheriff_player(player));
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
      if(wstate.phase() != Phase::load or to_sheriff_player(observer) != ::sheriff::Player::one) {
         // only the smuggler himself learns his own cargo size
         return observation_type{};
      }
      return observation_type{
         .kind = Observation::Kind::load,
         .actor = observer,
         .own_cargo = std::get< Load >(action).items};
   }

   observation_type public_observation(
      const world_state_type& wstate,
      const action_type& action,
      const world_state_type& next_wstate
   ) const
   {
      const auto actor = active_player(wstate);
      switch(wstate.phase()) {
         case Phase::load: {
            // the loading event itself is public knowledge, its size stays hidden
            return observation_type{.kind = Observation::Kind::load, .actor = actor};
         }
         case Phase::offer: {
            return observation_type{
               .kind = Observation::Kind::offer,
               .actor = actor,
               .bribe_offered = std::get< Offer >(action).bribe};
         }
         case Phase::respond: {
            observation_type obs{
               .kind = Observation::Kind::response,
               .actor = actor,
               .bribe_accepted = std::get< Respond >(action).accept};
            if(next_wstate.terminal()) {
               obs.terminal_cause = next_wstate.terminal_cause();
               // an inspection opens the trunk for both sides to see; a settled bribe does not
               if(next_wstate.terminal_cause() == TerminalCause::inspection_goods) {
                  obs.revealed_cargo = next_wstate.cargo();
               } else if(next_wstate.terminal_cause() == TerminalCause::inspection_clean) {
                  obs.revealed_cargo = 0u;
               }
            }
            return obs;
         }
         case Phase::over: break;
      }
      return observation_type{};
   }

   observation_type tiny_repr(const world_state_type&) const { return observation_type{}; }

   ////////////////////////////////
   /// API: histories           ///
   ////////////////////////////////

   /**
    * chronological sequence of load + bargaining actions masked to what `player` can observe:
    * the cargo size is only ever visible to the smuggler (nullopt for the sheriff), everything
    * else is public.
    */
   [[nodiscard]] std::vector< PlayerInformedType< std::optional< action_variant_type > > >
   private_history(Player player, const world_state_type& wstate) const
   {
      return _masked_history(wstate, [&](const auto& record) {
         // every action is public knowledge except the cargo size, which only the smuggler sees
         return not std::holds_alternative< Load >(record.action) or record.actor == player;
      });
   }

   /// chronological sequence with the cargo size masked out (the public view)
   [[nodiscard]] std::vector< PlayerInformedType< std::optional< action_variant_type > > >
   public_history(const world_state_type& wstate) const
   {
      return _masked_history(wstate, [](const auto& record) {
         return not std::holds_alternative< Load >(record.action);
      });
   }

   /// the fully open history in which even the secret cargo is revealed
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
      // one entry per executed step: load + (offer + respond) per resolved round (+ pending offer)
      return size_t(wstate.cargo().has_value()) + 2u * wstate.resolved_rounds()
             + size_t(not wstate.terminal() && wstate.phase() == Phase::respond);
   }

   /// reconstructs the chronological public action sequence from the state's aggregate data
   [[nodiscard]] std::vector< ActionRecord > _action_records(const world_state_type& wstate) const
   {
      std::vector< ActionRecord > out;
      out.reserve(_history_size(wstate));
      if(wstate.cargo().has_value()) {
         out.emplace_back(Player::alex, Action{Load{*wstate.cargo()}});
      }
      for(size_t r = 0; r < wstate.resolved_rounds(); ++r) {
         const auto& record = wstate.rounds_log().at(r);
         out.emplace_back(Player::alex, Action{Offer{record.bribe}});
         out.emplace_back(Player::bob, Action{Respond{record.accepted}});
      }
      if(not wstate.terminal() && wstate.phase() == Phase::respond) {
         out.emplace_back(Player::alex, Action{Offer{*wstate.pending_bribe()}});
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

}  // namespace nor::games::sheriff

namespace common {

// NOTE: the TerminalCause printer has to precede the Observation printer, whose body
// references it (non-dependent name -> must be visible at definition time)
template <>
inline std::string to_string(const ::sheriff::TerminalCause& value)
{
   switch(value) {
      case ::sheriff::TerminalCause::bribe_accepted: return "bribe_accepted";
      case ::sheriff::TerminalCause::inspection_goods: return "inspection_goods";
      case ::sheriff::TerminalCause::inspection_clean: return "inspection_clean";
      default: return "none";
   }
}

template <>
inline std::string to_string(const nor::games::sheriff::Observation& value)
{
   namespace sh = nor::games::sheriff;
   std::string out;
   switch(value.kind) {
      case sh::Observation::Kind::load: {
         out = fmt::format("{}:loads", common::to_string(value.actor));
         if(value.own_cargo.has_value()) {
            out += fmt::format("={}", *value.own_cargo);
         }
         break;
      }
      case sh::Observation::Kind::offer: {
         out = fmt::format("{}:offers{}", common::to_string(value.actor), *value.bribe_offered);
         break;
      }
      case sh::Observation::Kind::response: {
         out = fmt::format(
            "{}:{}",
            common::to_string(value.actor),
            value.bribe_accepted.has_value() and *value.bribe_accepted ? "accepts" : "rejects"
         );
         if(value.revealed_cargo.has_value()) {
            out += fmt::format(",cargo={}", *value.revealed_cargo);
         }
         if(value.terminal_cause.has_value()) {
            out += "," + common::to_string(*value.terminal_cause);
         }
         break;
      }
      default: return "-";
   }
   return out;
}

}  // namespace common

COMMON_ENABLE_PRINT(nor::games::sheriff, Observation);

namespace nor {

template <>
struct fosg_traits< games::sheriff::Infostate > {
   using observation_type = nor::games::sheriff::Observation;
};

template <>
struct fosg_traits< games::sheriff::Environment > {
   using world_state_type = nor::games::sheriff::State;
   using info_state_type = nor::games::sheriff::Infostate;
   using public_state_type = nor::games::sheriff::Publicstate;
   using action_type = nor::games::sheriff::Action;
   using chance_outcome_type = std::monostate;
   using observation_type = nor::games::sheriff::Observation;
};

}  // namespace nor

namespace std {

template < typename StateType >
   requires common::
      is_any_v< StateType, nor::games::sheriff::Publicstate, nor::games::sheriff::Infostate >
   struct hash< StateType > {
   size_t operator()(const StateType& state) const noexcept { return state.hash(); }
};

}  // namespace std

#endif  // NOR_SHERIFF_ENVIRONMENT_HPP
