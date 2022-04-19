
#ifndef NOR_NODE_HPP
#define NOR_NODE_HPP

#include <map>
#include <vector>

#include "common/common.hpp"
#include "nor/concepts.hpp"
#include "nor/game_defs.hpp"
#include "nor/utils/utils.hpp"

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
  public:
   using action_type = Action;
   using info_state_type = Infostate;
   using world_state_type = Worldstate;
   using public_state_type = Publicstate;
   using cond_public_state_base = detail::CondPubstate< public_state_type >;
   using cond_world_state_base = detail::CondWorldstate< world_state_type >;

   template < typename GivenWorldstate >
   CFRNodeData(
      std::reference_wrapper< Infostate > info_reference,
      GivenWorldstate* worldstate,
      public_state_type publicstate)
       : cond_public_state_base{std::move(publicstate)},
         cond_world_state_base{init_wstate(worldstate)},
         m_infostate_ref(info_reference)
   {
   }

   auto& worldstate() { return cond_world_state_base::worldstate; }
   auto& publicstate() { return cond_public_state_base::publicstate; }
   auto& infostate() { return m_infostate_ref.get(); }
   auto& value(Player player) { return m_value[player]; }
   auto& value() { return m_value; }

   [[nodiscard]] auto& worldstate() const { return cond_world_state_base::worldstate; }
   [[nodiscard]] auto& publicstate() const { return cond_public_state_base::publicstate; }
   [[nodiscard]] auto& infostate() const { return m_infostate_ref.get(); }
   [[nodiscard]] auto& value(Player player) const { return m_value.at(player); }
   [[nodiscard]] auto& value() const { return m_value; }

  private:
   /// the information state of the each player at this node
   /// the reference wrapper indicates non-ownership of the contained value. A raw pointer would be
   /// ambiguous.
   std::reference_wrapper< Infostate > m_infostate_ref;

   /// the value of each player for this node.
   /// Needs to be updated later during the traversal.
   std::unordered_map< Player, double > m_value;

   template < typename GivenWorldstate >
   auto init_wstate(const GivenWorldstate* wstate_ptr)
   {
      if constexpr(cond_world_state_base::empty_optimization) {
         return Worldstate{};
      } else {
         static_assert(
            std::is_same_v< GivenWorldstate, world_state_type >,
            "Passed True worldstate differs from configured node's worldstate type. ");
         return utils::static_unique_ptr_downcast< world_state_type >(
            utils::clone_any_way(wstate_ptr));
      }
   }
};

template < typename Action >
class InfostateNodeData {
  public:
   template < ranges::range ActionRange >
   InfostateNodeData(Player player, const ActionRange& actions) : m_player(player), m_regret()
   {
      for(const auto& action : actions) {
         m_regret.emplace(action, 0.);
      }
   }
   InfostateNodeData(Player player, std::unordered_map< Action, double > regret_per_action)
       : m_player(player), m_regret(std::move(regret_per_action))
   {
   }

   auto& regret(const Action& action) { return m_regret[action]; }
   auto& regret() { return m_regret; }

   [[nodiscard]] auto& regret(const Action& action) const { return m_regret[action]; }
   [[nodiscard]] auto& regret() const { return m_regret; }
   [[nodiscard]] auto player() const { return m_player; }

  private:
   /// the active player at this node
   Player m_player;
   /// the cumulative regret the active player amassed with each action. Cumulative with regards to
   /// the number of CFR iterations. Defaults to 0 and should be updated later during the traversal.
   std::unordered_map< Action, double > m_regret{};
};

}  // namespace nor::rm

#endif  // NOR_NODE_HPP
