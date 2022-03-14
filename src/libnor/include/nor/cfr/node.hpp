
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

///**
// * The empty public state type will still need to fulfill the concept of a public state type!
// * @tparam Action
// * @tparam Observation
// */
// template < concepts::action Action, concepts::observation Observation >
// struct empty_public_state {
//   using action_type = Action;
//   using observation_type = Observation;
//
//   [[nodiscard]] size_t size() const { return 0; }
//   void append(const std::pair <Action, Observation>&) {}
//};

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
   /// In order to avoid any storage overhead we don't let the empty type have unique storage. This
   /// way there won't be individual storage of empty structs, but rather a single address for all
   [[no_unique_address]] Publicstate public_state;
};

}  // namespace detail

template < typename Action, typename Infostate, typename Publicstate >
struct CFRNode: public detail::CondPubstate< Publicstate > {
   using info_state_type = Infostate;
   using public_state_type = Publicstate;
   using cond_public_state_base = detail::CondPubstate< Publicstate >;

   CFRNode(
      Player player,
      std::vector< Action > legal_actions,
      bool is_terminal,
      std::map< Player, Infostate > info_states,
      Publicstate public_state = {},
      CFRNode* parent = nullptr)
       : cond_public_state_base{std::move(public_state)},
         m_player(player),
         m_actions(std::move(legal_actions)),
         m_terminal(is_terminal),
         m_infostates(std::move(info_states)),
         m_reach_prob(),
         m_parent(parent)
   {
   }

   CFRNode(
      Player player,
      std::vector< Action > legal_actions,
      bool is_terminal,
      std::map< Player, Infostate > info_states,
      Publicstate public_state,
      std::map< Player, double > reach_prob,
      CFRNode* parent = nullptr)
       : cond_public_state_base{std::move(public_state)},
         m_player(player),
         m_actions(std::move(legal_actions)),
         m_terminal(is_terminal),
         m_infostates(std::move(info_states)),
         m_reach_prob(std::move(reach_prob)),
         m_parent(parent)
   {
   }

   auto& public_state() { return cond_public_state_base::public_state; }
   auto player() { return m_player; }
   auto& info_states(Player player) { return m_infostates[player]; }
   auto& info_states() { return m_infostates; }
   auto& value(Player player) { return m_value[player]; }
   auto& value() { return m_value; }
   auto& reach_probability(Player player) { return m_reach_prob[player]; }
   auto& reach_probability() { return m_reach_prob; }
   auto& actions() { return m_actions; }
   auto& children(const Action& action) { return m_children[action]; }
   auto& children() { return m_children; }
   auto& regret(const Action& action) { return m_regret[action]; }
   auto& regret() { return m_regret; }
   auto parent() { return m_parent; }
   [[nodiscard]] auto& public_state() const { return cond_public_state_base::public_state; }
   [[nodiscard]] auto terminal() const { return m_terminal; }
   [[nodiscard]] auto player() const { return m_player; }
   [[nodiscard]] auto& info_states(Player player) const { return m_infostates.at(player); }
   [[nodiscard]] auto& info_states() const { return m_infostates; }
   [[nodiscard]] auto& value(Player player) const { return m_value.at(player); }
   [[nodiscard]] auto& value() const { return m_value; }
   [[nodiscard]] auto& reach_probability(Player player) const { return m_reach_prob.at(player); }
   [[nodiscard]] auto& reach_probability() const { return m_reach_prob; }
   [[nodiscard]] auto& actions() const { return m_actions; }
   [[nodiscard]] auto& children(const Action& action) const { return m_children[action]; }
   [[nodiscard]] auto& regret(const Action& action) const { return m_regret[action]; }
   [[nodiscard]] auto& children() const { return m_children; }
   [[nodiscard]] auto& regret() const { return m_regret; }
   [[nodiscard]] auto parent() const { return m_parent; }

  private:
   /// the currently active player at the world state
   Player m_player;
   /// the legal actions the active player can choose from at this state.
   std::vector< Action > m_actions;
   /// whether it is a terminal node
   bool m_terminal = false;

   // player-based storage

   /// the information states of each player at this node
   std::map< Player, Infostate > m_infostates;
   /// the value of each player for this node.
   /// Defaults to 0 and should be updated later during the traversal.
   std::map< Player, double > m_value{};
   /// the reach probability of each player for this node.
   /// Defaults to 0 and should be updated later during the traversal.
   std::map< Player, double > m_reach_prob{};


   // action-based storage
   // these don't need to be on a per player basis, as only the active player of this node has
   // actions to play.

   /// the children that each action maps to in the game tree.
   /// Should be filled during the traversal.
   std::unordered_map< Action, CFRNode* > m_children{};
   /// the cumulative regret the active player amassed with each action.
   /// Defaults to 0 and should be updated later during the traversal.
   std::unordered_map< Action, double > m_regret{};

   /// the parent node from which this node stems
   CFRNode* m_parent;
};

}  // namespace nor::rm

// namespace std {
//
// template < typename... Args >
// struct hash< nor::rm::CFRNode< Args... > > {
//    size_t operator()(const nor::rm::CFRNode< Args... >& node) const noexcept { return
//    node.hash(); }
// };
// }  // namespace std

#endif  // NOR_NODE_HPP
