
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
   Publicstate public_state;
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
   [[no_unique_address]] Publicstate public_state;
};

}  // namespace detail

template < typename Action, typename Worldstate, typename Publicstate, typename Infostate >
struct CFRNode: public detail::CondPubstate< Publicstate > {
   using cond_public_state = detail::CondPubstate< Publicstate >;

   CFRNode(
      const Worldstate& world_state_,
      const Publicstate& public_state_,
      const Infostate& info_state_,
      Player player_,
      std::map< Player, double > reach_prob,
      CFRNode* parent = nullptr)
       : m_world_state(world_state_),
         cond_public_state(public_state_),
         m_info_state(info_state_),
         m_player(player_),
         m_reach_prob(std::move(reach_prob)),
         m_parent(parent),
         m_hash_cache(std::hash< Infostate >{}(m_info_state))
   {
   }

   template < concepts::fosg Game >
   CFRNode(const Game& g, const std::map< Player, double >& reach_prob_)
       : CFRNode(
          g.m_world_state(),
          g.public_state(g.m_world_state()),
          g.m_info_state(g.m_world_state(), g.m_player()),
          g.active_player(g.m_world_state()),
          reach_prob_)
   {
      static_assert(
         std::is_same_v< Action, typename Game::action_type >,
         "Provided action type and game action type are not the same.");
   }

   auto hash() const { return m_hash_cache; }
   auto operator==(const CFRNode& other) const { return m_world_state == other.m_world_state; }

   auto& world_state() { return m_world_state; }
   auto& public_state() { return cond_public_state::public_state; }
   auto& info_state() { return m_info_state; }
   auto player() { return player; }
   auto& value(Player player) { return m_value[player]; }
   auto& value() { return m_value; }
   auto& reach_probability(Player player) { return m_reach_prob[player]; }
   auto& reach_probability() { return m_reach_prob; }
   auto& children(const Action& action) { return m_children[action]; }
   auto& children() { return m_children; }
   auto& regret(const Action& action) { return m_regret[action]; }
   auto& regret() { return m_regret; }
   auto parent() { return m_parent; }
   [[nodiscard]] auto& world_state() const { return m_world_state; }
   [[nodiscard]] auto& public_state() const { return cond_public_state::public_state; }
   [[nodiscard]] auto& info_state() const { return m_info_state; }
   [[nodiscard]] auto player() const { return player; }
   [[nodiscard]] auto& value(Player player) const { return m_value.at(player); }
   [[nodiscard]] auto& value() const { return m_value; }
   [[nodiscard]] auto& reach_probability(Player player) const { return m_reach_prob.at(player); }
   [[nodiscard]] auto& reach_probability() const { return m_reach_prob; }
   [[nodiscard]] auto& children(const Action& action) const { return m_children[action]; }
   [[nodiscard]] auto& regret(const Action& action) const { return m_regret[action]; }
   [[nodiscard]] auto& children() const { return m_children; }
   [[nodiscard]] auto& regret() const { return m_regret; }
   [[nodiscard]] auto parent() const { return m_parent; }

  private:
   /// the overall state of the game at this node
   Worldstate m_world_state;
   /// the private information state of the active player at this node
   Infostate m_info_state;
   /// the currently active player at the world state
   Player m_player;

   // player-based storage

   /// the value of each player for this node.
   /// Defaults to 0 and should be updated later during the traversal.
   std::map< Player, double > m_value{};
   /// the reach probability of each player for this node.
   /// Defaults to 0 and should be updated later during the traversal.
   std::map< Player, double > m_reach_prob{};

   // action-based storage

   /// the children that each action maps to in the game tree.
   /// Should be filled during the traversal.
   std::map< Action, CFRNode* > m_children{};
   /// the cumulative regret the active player amassed with each action.
   /// Defaults to 0 and should be updated later during the traversal.
   std::map< Action, double > m_regret{};

   /// the parent node from which this node stems
   CFRNode* m_parent;
   /// the cached hash of the stored information state
   size_t m_hash_cache;
};

}  // namespace nor::rm

namespace std {

template < typename... Args >
struct hash< nor::rm::CFRNode< Args... > > {
   size_t operator()(const nor::rm::CFRNode< Args... >& node) const { return node.hash(); }
};
}  // namespace std

#endif  // NOR_NODE_HPP
