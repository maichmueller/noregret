
#ifndef NOR_NODE_HPP
#define NOR_NODE_HPP

#include <map>
#include <vector>

#include "nor/concepts.hpp"
#include "nor/game_defs.hpp"

namespace nor::rm {

/**
 * @brief the node type to represent game states in the game tree built by CFR.
 *
 * There is no concept checking for this class as any of the template types are supposed to be
 * checked within the CFR class for concept fulfillment and thus allow some flexibility in data
 * storage (e.g. not store the public states, by setting them as empty class - the minimal storage
 * cost)
 * @tparam Action
 * @tparam Infostate
 * @tparam Publicstate
 * @tparam Worldstate
 */

namespace detail {

template < typename Publicstate, typename = void >
struct CondPubstate {
   /// the public information state at this node
   Publicstate m_public_state;
};

/**
 * @brief Empty Publicstate optimization for when its storage is not required.
 *
 * This is useful e.g. when one would want to visualize the tree with private and public
 * information.
 */
template < typename Publicstate >
struct CondPubstate< Publicstate, std::enable_if_t< concepts::is::empty< Publicstate > > > {
   /// the public information state at this node
   /// In order to avoid any storage overhead we don't let the empty type have unique storage. This
   /// way there won't be individual storage of empty structs, but rather a single address for all
   [[no_unique_address]] Publicstate m_public_state;
};

}  // namespace detail

template < typename Action, typename Infostate, typename Publicstate >
struct CFRNodeData: public detail::CondPubstate< Publicstate > {
   using info_state_type = Infostate;
   using public_state_type = Publicstate;
   using cond_public_state_base = detail::CondPubstate< Publicstate >;

   CFRNodeData(
      Player player,
      std::unordered_map< Player, Infostate > info_states,
      Publicstate public_state,
      std::unordered_map< Player, double > reach_prob)
       : cond_public_state_base{std::move(public_state)},
         m_player(player),
         m_infostates(std::move(info_states)),
         m_compound_reach_prob_contribution(std::move(reach_prob))
   {
   }

   auto& publicstate() { return cond_public_state_base::m_public_state; }
   auto& infostates() { return m_infostates; }
   auto& infostate(Player player) { return m_infostates.at(player); }
   auto& infostate() { return infostate(m_player); }
   auto& value(Player player) { return m_value[player]; }
   auto& value() { return m_value; }
   auto& reach_probability_contrib(Player player)
   {
      return m_compound_reach_prob_contribution[player];
   }
   auto& reach_probability_contrib() { return m_compound_reach_prob_contribution; }
   auto& regret(const Action& action) { return m_regret[action]; }
   auto& regret() { return m_regret; }

   [[nodiscard]] auto& publicstate() const { return cond_public_state_base::m_public_state; }
   [[nodiscard]] auto& infostates() const { return m_infostates; }
   [[nodiscard]] auto& infostate(Player player) const { return m_infostates.at(player); }
   [[nodiscard]] auto& infostate() const { return infostate(m_player); }
   [[nodiscard]] auto player() const { return m_player; }
   [[nodiscard]] auto& value(Player player) const { return m_value.at(player); }
   [[nodiscard]] auto& value() const { return m_value; }
   [[nodiscard]] auto& reach_probability_contrib(Player player) const
   {
      return m_compound_reach_prob_contribution.at(player);
   }
   [[nodiscard]] auto& reach_probability_contrib() const
   {
      return m_compound_reach_prob_contribution;
   }
   [[nodiscard]] auto& regret(const Action& action) const { return m_regret[action]; }
   [[nodiscard]] auto& regret() const { return m_regret; }

  private:
   /// the active player at this node
   Player m_player;
   /// the information state of the each player at this node
   std::unordered_map< Player, Infostate > m_infostates;

   // per-player-based storage

   /// the value of each player for this node.
   /// Needs to be updated later during the traversal.
   std::unordered_map< Player, double > m_value;
   /// the compounding reach probability of each player for this node (i.e. the probability
   /// contribution of each player along the trajectory to this node multiplied together).
   /// Needs to be updated during the traversal with the current policy.
   std::unordered_map< Player, double > m_compound_reach_prob_contribution;

   // action-based storage
   // these don't need to be on a per player basis, as only the active player of this node has
   // actions to play.

   /// the cumulative regret the active player amassed with each action. Cumulative with regards to
   /// the number of CFR iterations. Defaults to 0 and should be updated later during the traversal.
   std::unordered_map< Action, double > m_regret{};
};

}  // namespace nor::rm

#endif  // NOR_NODE_HPP
