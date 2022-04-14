
#ifndef NOR_NODE_HPP
#define NOR_NODE_HPP

#include <map>
#include <vector>

#include "nor/concepts.hpp"
#include "nor/game_defs.hpp"
#include "nor/utils/utils.hpp"
#include "common/common.hpp"

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

/// In order to avoid any storage overhead we don't let the empty type have unique storage. This
/// way there won't be individual storage of empty structs in the node, but rather a single address
/// for all. This reduces the memory footprint of each node by a potentially significant amount,
/// depending on compiler struct padding

template < typename Publicstate >
struct CondPubstate {
   constexpr static bool empty_optimization = false;
   /// the public information state at this node
   Publicstate publicstate;
};

template < typename Worldstate >
struct CondWorldstate {
   constexpr static bool empty_optimization = false;
   /// the world state at this node
   uptr< Worldstate > worldstate;
};

/**
 * @brief Empty Publicstate optimization for when its storage is not required.
 *
 * This is useful e.g. when one would want to visualize the tree with private and public
 * information.
 */
template < concepts::is::empty Publicstate >
struct CondPubstate< Publicstate > {
   constexpr static bool empty_optimization = true;
   [[no_unique_address]] Publicstate publicstate;
};

template < concepts::is::empty Worldstate >
struct CondWorldstate< Worldstate > {
   constexpr static bool empty_optimization = true;
   [[no_unique_address]] Worldstate worldstate;
};

}  // namespace detail

template < typename Action, typename Infostate, typename Worldstate, typename Publicstate >
struct CFRNodeData:
    public detail::CondPubstate< Publicstate >,
    detail::CondWorldstate< Worldstate > {
   using info_state_type = Infostate;
   using world_state_type = Worldstate;
   using public_state_type = Publicstate;
   using cond_public_state_base = detail::CondPubstate< public_state_type >;
   using cond_world_state_base = detail::CondWorldstate< world_state_type >;

   template < typename TrueWorldstate >
   CFRNodeData(
      Player player,
      std::unordered_map< Player, info_state_type > info_states,
      TrueWorldstate* worldstate,
      public_state_type publicstate,
      std::unordered_map< Player, double > reach_prob)
       : cond_public_state_base{std::move(publicstate)},
         cond_world_state_base{init_wstate(worldstate)},
         m_player(player),
         m_infostates(std::move(info_states)),
         m_compound_reach_prob_contribution(std::move(reach_prob))
   {
   }

   auto& worldstate() { return cond_world_state_base::worldstate; }
   auto& publicstate() { return cond_public_state_base::publicstate; }
   auto& infostates() { return m_infostates; }
   auto& infostate(Player player) { return m_infostates.at(player); }
   auto& infostate() { return infostate(m_player); }
   auto& value(Player player) { return m_value[player]; }
   auto& value() { return m_value; }
   auto& reach_probability(Player player)
   {
      return m_compound_reach_prob_contribution[player];
   }
   auto& reach_probability() { return m_compound_reach_prob_contribution; }
   auto& regret(const Action& action) { return m_regret[action]; }
   auto& regret() { return m_regret; }

   [[nodiscard]] auto& worldstate() const { return cond_world_state_base::worldstate; }
   [[nodiscard]] auto& publicstate() const { return cond_public_state_base::publicstate; }
   [[nodiscard]] auto& infostates() const { return m_infostates; }
   [[nodiscard]] auto& infostate(Player player) const { return m_infostates.at(player); }
   [[nodiscard]] auto& infostate() const { return infostate(m_player); }
   [[nodiscard]] auto player() const { return m_player; }
   [[nodiscard]] auto& value(Player player) const { return m_value.at(player); }
   [[nodiscard]] auto& value() const { return m_value; }
   [[nodiscard]] auto& reach_probability(Player player) const
   {
      return m_compound_reach_prob_contribution.at(player);
   }
   [[nodiscard]] auto& reach_probability() const
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

   template < typename TrueWorldstate >
   auto init_wstate(const TrueWorldstate* wstate_ptr)
   {
      if constexpr(cond_world_state_base::empty_optimization) {
         return Worldstate{};
      } else {
         static_assert(
            std::is_same_v< TrueWorldstate, world_state_type >,
            "Passed True worldstate differs from configured node's worldstate type. ");
         return utils::static_unique_ptr_downcast< world_state_type >(
            utils::clone_any_way(wstate_ptr));
      }
   }
};

}  // namespace nor::rm

#endif  // NOR_NODE_HPP
