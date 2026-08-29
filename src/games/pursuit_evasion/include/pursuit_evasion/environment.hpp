
#ifndef NOR_PURSUIT_EVASION_ENVIRONMENT_HPP
#define NOR_PURSUIT_EVASION_ENVIRONMENT_HPP

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
#include "pursuit_evasion/state.hpp"
#include "pursuit_evasion/utils.hpp"

namespace nor::games::pursuit_evasion {

using namespace ::pursuit_evasion;

inline auto to_pe_player(const nor::Player& player)
{
   return static_cast< ::pursuit_evasion::Player >(player);
}
inline auto to_nor_player(const ::pursuit_evasion::Player& player)
{
   return static_cast< nor::Player >(player);
}

/**
 * @brief Compact observation of pursuit-evasion.
 *
 * Information structure (paper App. G): the attacker privately tracks his own position (and
 * whether he waited, which is exactly the event that cleaned his traces); the defender privately
 * tracks her two patrol positions and any trace-sighting events ("attacker-was-here"); the public
 * channel carries only the bare fact of a commitment during the commit phases and the terminal
 * cause announcement. In particular patrol positions NEVER reach the attacker and attacker
 * positions never reach the defender except as anonymous trace sightings.
 */
struct Observation {
   /// a commitment happened by this player; its value stays hidden (commit phases)
   std::optional< nor::Player > committed_by{};
   /// attacker-private echo of his fresh commitment (CommitA only)
   std::optional< AttMove > own_att_move{};
   /// defender-private echo of her fresh compound commitment (CommitD only)
   std::optional< DefMove > own_def_move{};
   /// attacker-private post-resolve position (fused resolve)
   std::optional< uint8_t > own_position{};
   /// attacker-private: he waited this round, clearing all his traces
   bool waited = false;
   /// defender-private post-resolve patrol positions (p1, p2)
   std::optional< std::pair< uint8_t, uint8_t > > patrol_positions{};
   /// defender-private sighting: patrol 1 ENTERED a node carrying a pre-round trace; payload is
   /// that node ("attacker-was-here")
   std::optional< uint8_t > sighting_p1{};
   /// defender-private sighting of patrol 2, analogously
   std::optional< uint8_t > sighting_p2{};
   /// public announcement of how the game ended (terminal transitions only); payoffs themselves
   /// are read off the world state via reward()
   std::optional< TerminalCause > terminal_cause{};

   friend bool operator==(const Observation&, const Observation&) = default;
};

}  // namespace nor::games::pursuit_evasion

/// has to be visible before the DefaultPublicstate/DefaultInfostate bases below, whose
/// 'observation' concept requirement probes the availability of this specialization
namespace std {

template <>
struct hash< nor::games::pursuit_evasion::Observation > {
   size_t operator()(const nor::games::pursuit_evasion::Observation& obs) const noexcept
   {
      namespace pe = nor::games::pursuit_evasion;
      size_t seed = 0;
      if(obs.committed_by.has_value()) {
         common::hash_combine(seed, std::hash< int >{}(1 + int(*obs.committed_by)));
      }
      if(obs.own_att_move.has_value()) {
         common::hash_combine(seed, std::hash< pe::AttMove >{}(*obs.own_att_move));
      }
      if(obs.own_def_move.has_value()) {
         common::hash_combine(seed, std::hash< pe::DefMove >{}(*obs.own_def_move));
      }
      if(obs.own_position.has_value()) {
         common::hash_combine(seed, std::hash< unsigned >{}(31u + unsigned(*obs.own_position)));
      }
      common::hash_combine(seed, std::hash< bool >{}(obs.waited));
      if(obs.patrol_positions.has_value()) {
         common::hash_combine(
            seed,
            std::hash< unsigned >{}(
               61u + 16u * unsigned(obs.patrol_positions->first)
               + unsigned(obs.patrol_positions->second)
            )
         );
      }
      if(obs.sighting_p1.has_value()) {
         common::hash_combine(seed, std::hash< unsigned >{}(97u + unsigned(*obs.sighting_p1)));
      }
      if(obs.sighting_p2.has_value()) {
         common::hash_combine(seed, std::hash< unsigned >{}(131u + unsigned(*obs.sighting_p2)));
      }
      if(obs.terminal_cause.has_value()) {
         common::hash_combine(seed, std::hash< unsigned >{}(163u + unsigned(*obs.terminal_cause)));
      }
      return seed;
   }
};

}  // namespace std

namespace nor::games::pursuit_evasion {

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
 * @brief The FOSG environment adapter of (deterministic) PCFR+ App. G pursuit-evasion.
 *
 * The simultaneous attacker/defender move of each round is sequentialized CommitA -> CommitD with
 * the deterministic resolve fused into the CommitD application (see State's Phase documentation).
 * Trace-sighting information created by the environment flows through `private_observation`
 * keyed to the defender only.
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

   Environment() = default;
   explicit Environment(Config config) : m_config(std::move(config)) {}

   [[nodiscard]] const Config& config() const { return m_config; }

   ///////////////////////////////////
   /// API: transitions            ///
   ///////////////////////////////////

   [[nodiscard]] std::vector< action_type > actions(Player player, const world_state_type& wstate)
      const
   {
      return wstate.actions(to_pe_player(player));
   }

   void transition(world_state_type& worldstate, const action_type& action) const
   {
      worldstate.apply_action(action);
   }

   void transition(world_state_type& worldstate, const AttMove& action) const
   {
      worldstate.apply_action(action);
   }

   void transition(world_state_type& worldstate, const DefMove& action) const
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

   /// alex is the attacker, bob the defender
   static double reward(Player player, const world_state_type& wstate)
   {
      return wstate.payoff(to_pe_player(player));
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
      if(const auto* att = std::get_if< AttMove >(&action)) {
         // CommitA: only the committer echoes his move; the resolve has not happened yet
         if(to_pe_player(observer) != wstate.active_player()) {
            return observation_type{};
         }
         return observation_type{.own_att_move = *att, .waited = att->is_wait()};
      }
      const auto& def = std::get< DefMove >(action);
      if(to_pe_player(observer) == ::pursuit_evasion::Player::one) {
         // fused resolve, attacker side: his new position and whether he cleaned his traces
         return observation_type{
            .own_position = next_wstate.attacker_node(),
            .waited = wstate.committed_attacker_move()->is_wait()};
      }
      if(to_pe_player(observer) == ::pursuit_evasion::Player::two) {
         // fused resolve, defender side: her compound echo, the fresh patrol positions and any
         // trace sightings -- computed against the PRE-round trace mask so simultaneity never
         // leaks the same-tick deposit
         observation_type obs{
            .own_def_move = def,
            .patrol_positions = std::pair{
               next_wstate.patrol_nodes()[0], next_wstate.patrol_nodes()[1]}};
         const uint16_t pre_traces = wstate.trace_mask();
         const auto [p1_from, p2_from] = wstate.patrol_nodes();
         if(def.p1 != p1_from && ((pre_traces >> def.p1) & uint16_t(1)) != 0u) {
            obs.sighting_p1 = def.p1;
         }
         if(def.p2 != p2_from && ((pre_traces >> def.p2) & uint16_t(1)) != 0u) {
            obs.sighting_p2 = def.p2;
         }
         return obs;
      }
      return observation_type{};
   }

   observation_type public_observation(
      const world_state_type& /*wstate*/,
      const action_type& action,
      const world_state_type& next_wstate
   ) const
   {
      observation_type obs{};
      if(std::holds_alternative< AttMove >(action)) {
         obs.committed_by = Player::alex;
      } else {
         obs.committed_by = Player::bob;
         if(next_wstate.terminal()) {
            obs.terminal_cause = next_wstate.terminal_cause();
         }
      }
      // otherwise an entirely silent public channel until the terminal announcement
      return obs;
   }

   observation_type tiny_repr(const world_state_type&) const { return observation_type{}; }

   ////////////////////////////////
   /// API: histories           ///
   ////////////////////////////////

   /**
    * chronological sequence of commitments masked to what `player` can observe: his/her own
    * commitments in full, everything else hidden (nullopt). Two entries per resolved round
    * (CommitA, CommitD-with-fused-resolve), plus one pending CommitA entry while a resolution is
    * awaited.
    */
   [[nodiscard]] std::vector< PlayerInformedType< std::optional< action_variant_type > > >
   private_history(Player player, const world_state_type& wstate) const
   {
      const auto self = to_pe_player(player);
      std::vector< PlayerInformedType< std::optional< action_variant_type > > > out;
      out.reserve(2 * wstate.round() + size_t(wstate.phase() == Phase::commit_defender));
      for(size_t r = 0; r < wstate.round(); ++r) {
         const auto& record = wstate.rounds()[r];
         auto att_entry = record.att_edge >= 0
                             ? std::optional< action_variant_type >{action_variant_type{
                                AttMove{uint8_t(record.att_edge)}}}
                             : std::optional< action_variant_type >{
                                action_variant_type{AttMove{AttMove::wait_edge_id}}};
         out.emplace_back(
            self == ::pursuit_evasion::Player::one ? att_entry : std::nullopt, Player::alex
         );
         auto def_entry = action_variant_type{
            DefMove{uint8_t(record.p1_to), uint8_t(record.p2_to)}};
         out.emplace_back(
            self == ::pursuit_evasion::Player::two ? std::optional< action_variant_type >{def_entry}
                                                   : std::nullopt,
            Player::bob
         );
      }
      if(not wstate.terminal() && wstate.phase() == Phase::commit_defender) {
         auto committed = wstate.committed_attacker_move();
         out.emplace_back(
            self == ::pursuit_evasion::Player::one
               ? std::optional< action_variant_type >{action_variant_type{*committed}}
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
      out.reserve(2 * n_rounds + size_t(wstate.phase() == Phase::commit_defender));
      for(size_t r = 0; r < n_rounds; ++r) {
         out.emplace_back(std::nullopt, Player::alex);
         out.emplace_back(std::nullopt, Player::bob);
      }
      if(not wstate.terminal() && wstate.phase() == Phase::commit_defender) {
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
      out.reserve(2 * wstate.round() + size_t(wstate.phase() == Phase::commit_defender));
      for(size_t r = 0; r < wstate.round(); ++r) {
         const auto& record = wstate.rounds()[r];
         const auto att_edge = record.att_edge >= 0 ? uint8_t(record.att_edge)
                                                    : AttMove::wait_edge_id;
         out.emplace_back(action_variant_type{AttMove{att_edge}}, Player::alex);
         out.emplace_back(
            action_variant_type{DefMove{uint8_t(record.p1_to), uint8_t(record.p2_to)}}, Player::bob
         );
      }
      if(not wstate.terminal() && wstate.phase() == Phase::commit_defender) {
         out.emplace_back(action_variant_type{*wstate.committed_attacker_move()}, Player::alex);
      }
      out.shrink_to_fit();
      return out;
   }

  private:
   Config m_config{};
};

}  // namespace nor::games::pursuit_evasion

namespace common {

template <>
inline std::string to_string(const nor::games::pursuit_evasion::Observation& value)
{
   namespace pe = nor::games::pursuit_evasion;
   std::string out;
   if(value.committed_by.has_value()) {
      out += fmt::format("committed:{},", common::to_string(*value.committed_by));
   }
   if(value.own_att_move.has_value()) {
      out += fmt::format("own_att:{},", common::to_string(*value.own_att_move));
   }
   if(value.own_def_move.has_value()) {
      out += fmt::format("own_def:{},", common::to_string(*value.own_def_move));
   }
   if(value.own_position.has_value()) {
      out += fmt::format("pos:{},", unsigned(*value.own_position));
   }
   if(value.waited) {
      out += "waited,";
   }
   if(value.patrol_positions.has_value()) {
      out += fmt::format(
         "patrols:({},{})",
         unsigned(value.patrol_positions->first),
         unsigned(value.patrol_positions->second)
      );
   }
   if(value.sighting_p1.has_value()) {
      out += fmt::format("sight1@{},", unsigned(*value.sighting_p1));
   }
   if(value.sighting_p2.has_value()) {
      out += fmt::format("sight2@{},", unsigned(*value.sighting_p2));
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

COMMON_ENABLE_PRINT(nor::games::pursuit_evasion, Observation);

namespace std {

template < typename StateType >
   requires common::is_any_v<
      StateType,
      nor::games::pursuit_evasion::Publicstate,
      nor::games::pursuit_evasion::Infostate >
struct hash< StateType > {
   size_t operator()(const StateType& state) const noexcept { return state.hash(); }
};

}  // namespace std

#endif  // NOR_PURSUIT_EVASION_ENVIRONMENT_HPP
