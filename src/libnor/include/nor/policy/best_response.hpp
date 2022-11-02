
#ifndef NOR_BEST_RESPONSE_HPP
#define NOR_BEST_RESPONSE_HPP

#include <queue>

#include "default_policy.hpp"
#include "nor/concepts.hpp"
#include "nor/rm/forest.hpp"
#include "nor/rm/rm_utils.hpp"
#include "policy_view.hpp"

namespace nor {

template < concepts::info_state Infostate, concepts::action Action >
class BestResponsePolicy {
  public:
   using info_state_type = Infostate;
   using action_type = Action;

   BestResponsePolicy(Player best_response_player) : m_br_player(best_response_player) {}

   BestResponsePolicy(
      Player best_response_player,
      std::unordered_map< info_state_type, std::pair< action_type, double > > best_response_map = {}
   )
       : m_br_player(best_response_player), m_best_response(std::move(best_response_map))
   {
   }

   template < concepts::fosg Env >
   decltype(auto) allocate(
      Env& env,
      std::unordered_map< Player, StatePolicyView< info_state_type, action_type > > player_policies,
      uptr< typename fosg_auto_traits< Env >::world_state_type > root_state,
      std::unordered_map< Player, info_state_type > root_infostates = {}
   )
   {
      auto istate_tree = forest::InfostateTree(
         env, std::move(root_state), std::move(root_infostates)
      );
      istate_tree.build(m_br_player, std::move(player_policies));
      _compute_best_responses< Env >(istate_tree.root_node());
      return *this;
   }

   auto operator()(const info_state_type& infostate) const
   {
      return HashmapActionPolicy< action_type >{
         std::array{std::pair{m_best_response.at(infostate).first, 1.}}};
   }
   auto operator()(const info_state_type& infostate, auto&&...) const
   {
      return operator()(infostate);
   }

   auto& map() const { return m_best_response; }
   auto value(const info_state_type& infostate) const
   {
      return m_best_response.at(infostate).second;
   }

  private:
   Player m_br_player;
   std::unordered_map< info_state_type, std::pair< action_type, double > > m_best_response;

   template < typename Env >
   double _compute_best_responses(typename forest::InfostateTree< Env >::Node& curr_node);
};

template < concepts::info_state Infostate, concepts::action Action >
template < typename Env >
double BestResponsePolicy< Infostate, Action >::_compute_best_responses(
   typename forest::InfostateTree< Env >::Node& curr_node
)
{
   // first check if this node's value hasn't been already computed by another visit or is already
   // in the br map
   if(curr_node.state_value.has_value()) {
      return curr_node.state_value.value();
   } else if(m_best_response.contains(*(curr_node.infostate))) {
      return m_best_response[*(curr_node.infostate)].second;
   }

   auto child_traverser = [&](auto state_value_updater) {
      for(const auto& [action_variant, child_tuple] : curr_node.children) {
         const auto& [child_node_uptr, action_prob, action_value_opt] = child_tuple;
         // the action_value_optional will only hold a value if this action leads directly
         // to a terminal state. In that case we know exactly what the value of the action
         // is and can set fetch it. Otherwise, we ask the downstream information state to
         // return to us its value in a recursive manner.
         // We don't set the action_value with the computed value afterwards, since we are
         // not going to query this infostate's actions for values anymore, as by that
         // point we already know the complete infostate value to return upon being
         // queried.
         double child_value = action_value_opt.has_value()
                                 ? action_value_opt.value()
                                 : _compute_best_responses< Env >(*child_node_uptr);
         // the state value is updated depending on the given update rule
         LOGD2("child-value", child_value);
         state_value_updater(action_variant, action_prob.value(), child_value);
      }
   };

   double state_value = 0.;
   if(curr_node.active_player == m_br_player) {
      // if this is an infostate pertaining to the Best Response player then we need to
      // compute:
      // 1. value(I) = max_{a}(v(a | I))
      // 2. best_response(I) = argmax_{a}(v(a | I))
      // we return value(I) back up as this infostate's value to the BR player.
      std::optional< action_type > best_action = std::nullopt;
      state_value = std::numeric_limits< double >::lowest();
      child_traverser([&](const auto& action_variant, rm::Probability, double child_value) {
         if(child_value > state_value) {
            // if we reach here the action variant must hold a player action, otherwise there was a
            // logic error beforehand
            LOGD2(
               "BRP action",
               std::visit([](const auto& av) { return common::to_string(av); }, action_variant)
            );
            best_action = std::get< 0 >(action_variant);
            state_value = child_value;
            LOGD2("BRP new state-value", state_value);
         }
      });
      // store this value to fill the BR policy with the answer for this infostate
      m_best_response.emplace(
         std::piecewise_construct,
         std::forward_as_tuple(*(curr_node.infostate)),
         std::forward_as_tuple(best_action.value(), state_value)
      );
   } else {
      // this is an opponent infostate or chance node. In both cases we compute:
      //    value(I) = sum_{a} (policy(a | I) * v(a | I))
      // we return value(I) back up as this infostate's value to the BR player.
      child_traverser([&](const auto& action, rm::Probability action_prob, double child_value) {
         LOGD2(
            "OPP action",
            std::visit([](const auto& av) { return common::to_string(av); }, action)
         );
         LOGD2("OPP old state-value ", state_value);
         LOGD2("OPP action Prob ", action_prob.get());
         LOGD2("OPP child-value", child_value);
         state_value += action_prob.get() * child_value;
         LOGD2("OPP new state-value", state_value);
      });
   }
   // store this value to not trigger a recomputation upon visit from another Infostate.
   curr_node.state_value = state_value;
   return state_value;
}

}  // namespace nor

#endif  // NOR_BEST_RESPONSE_HPP
