
#ifndef NOR_BEST_RESPONSE_HPP
#define NOR_BEST_RESPONSE_HPP

#include "nor/concepts.hpp"

namespace nor::rm {

//template <
//   concepts::fosg Env,
//   typename CacheMap,
//   concepts::world_state Worldstate = typename fosg_auto_traits< Env >::world_state_type,
//   concepts::info_state Infostate = typename fosg_auto_traits< Env >::info_state_type,
//   concepts::info_state Action = typename fosg_auto_traits< Env >::action_type >
//   requires concepts::map< CacheMap, Infostate, Action >
//auto best_response(
//   Env& env,
//   uptr< Worldstate > wstate,
//   const sptr< Infostate >& infostate,
//   CacheMap& br_cache)
//{
//   auto existing_br_cache_it = br_cache.find(*infostate);
//   if(existing_br_cache_it != br_cache.end()) {
//      return existing_br_cache_it->second;
//   }
//   auto& cached_br = br_cache[*infostate];
//
//   Action best_action = -1;
//   double best_value = std::numeric_limits< double >::lowest();
//   // The legal actions are the same for all children, so we arbitrarily pick the
//   // first one to get the legal actions from.
//   for(const auto& action : infoset[0].first->GetChildActions()) {
//      double value = 0;
//      // Prob here is the counterfactual reach-weighted probability.
//      for(const auto& state_and_prob : infoset) {
//         if(state_and_prob.second <= prob_cut_threshold_)
//            continue;
//         HistoryNode* state_node = state_and_prob.first;
//         HistoryNode* child_node = state_node->GetChild(action).second;
//         SPIEL_CHECK_TRUE(child_node != nullptr);
//         value += state_and_prob.second * Value(child_node->GetHistory());
//      }
//      if(value > best_value) {
//         best_value = value;
//         best_action = action;
//      }
//   }
//   if(best_action == -1)
//      SpielFatalError("No action was chosen.");
//   best_response_actions_[infostate] = best_action;
//   return best_action;
//}


}  // namespace nor::rm

#endif  // NOR_BEST_RESPONSE_HPP
