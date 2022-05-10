
#ifndef NOR_RM_ALGORITHMS_HPP
#define NOR_RM_ALGORITHMS_HPP

#include <execution>
#include <named_type.hpp>
#include <range/v3/all.hpp>

#include "common/common.hpp"
#include "node.hpp"
#include "nor/concepts.hpp"
#include "nor/game_defs.hpp"
#include "nor/type_defs.hpp"
#include "nor/utils/utils.hpp"

namespace nor::rm {

/// strong-types for passing arguments around with intent
using Probability = fluent::NamedType< double, struct reach_prob_tag >;
using Weight = fluent::NamedType< double, struct weight_tag >;
using StateValue = fluent::NamedType< double, struct state_value_tag >;
using StateValueMap = fluent::
   NamedType< std::unordered_map< Player, double >, struct value_map_tag >;
using ReachProbabilityMap = fluent::
   NamedType< std::unordered_map< Player, double >, struct reach_prob_map_tag >;

template < ranges::range Policy >
auto& normalize_action_policy_inplace(Policy& policy)
{
   double sum = 0.;
   for(const auto& [action, prob] : policy) {
      sum += prob;
   }
   for(auto& [action, prob] : policy) {
      prob /= sum;
   }
   return policy;
};

template < ranges::range Policy >
auto normalize_action_policy(const Policy& policy)
{
   Policy copy = policy;
   return normalize_action_policy_inplace(copy);
};

template < ranges::range Policy >
auto& normalize_state_policy_inplace(Policy& policy)
{
   for(auto& [_, action_policy] : policy) {
      normalize_action_policy_inplace(action_policy);
   }
   return policy;
};

template < ranges::range Policy >
auto normalize_state_policy(const Policy& policy)
{
   auto copy = policy;
   return normalize_state_policy_inplace(copy);
};

template < typename MapLike >
concept kv_like_over_doubles = requires(MapLike m) {
                                  // has to be key-value-like to iterate over values only
                                  ranges::views::values(m);
                                  // value type has to be convertible to double
                                  std::is_convertible_v<
                                     decltype(*(ranges::views::values(m).begin())),
                                     double >;
                               };
/**
 * @brief computes the reach probability of the node.
 *
 * Since each player's compounding likelihood contribution is stored in the nodes themselves, the
 * actual computation is nothing more than merely multiplying all player's individual
 * contributions.
 * @param reach_probability_contributions the compounded reach probability of each player for
 * this node
 * @return the reach probability of the nde
 */
template < kv_like_over_doubles KVdouble >
[[nodiscard]] inline double reach_probability(const KVdouble& reach_probability_contributions)
{
   auto values_view = reach_probability_contributions | ranges::views::values;
   return std::reduce(values_view.begin(), values_view.end(), double(1.), std::multiplies{});
}
/**
 * @brief computes the counterfactual reach probability of the player for this node.
 *
 * The method delegates the actual computation to the overload with an already provided reach
 * probability.
 * @param node the node at which the counterfactual reach probability is to be computed
 * @param player the player for which the value is computed
 * @return the counterfactual reach probability
 */
template < kv_like_over_doubles KVdouble >
   requires requires(KVdouble m) {
               // the keys have to of type 'Player' as well
               std::is_convertible_v< decltype(*(ranges::views::keys(m).begin())), Player >;
            }
inline double cf_reach_probability(
   const Player& player,
   const KVdouble& reach_probability_contributions)
{
   auto values_view = reach_probability_contributions
                      | ranges::views::filter([&](const auto& player_rp_pair) {
                           return std::get< 0 >(player_rp_pair) != player;
                        })
                      | ranges::views::values;
   return std::reduce(values_view.begin(), values_view.end(), double(1.), std::multiplies{});
}

/**
 * @brief Performs regret-matching on the given policy with respect to the provided regret
 *
 * @tparam Action
 * @tparam Policy
 */
template < concepts::action Action, concepts::action_policy< Action > Policy >
void regret_matching(Policy& policy_map, const std::unordered_map< Action, double >& cumul_regret)
{
   // sum up the positivized regrets and store them in a new vector
   std::unordered_map< Action, double > pos_regrets;
   double pos_regret_sum{0.};
   for(const auto& [action, regret] : cumul_regret) {
      double pos_regret = std::max(0., regret);
      pos_regrets.emplace(action, pos_regret);
      pos_regret_sum += pos_regret;
   }
   // apply the new policy to the vector policy
   auto exec_policy{std::execution::par_unseq};
   if(pos_regret_sum > 0) {
      if(cumul_regret.size() != policy_map.size()) {
         throw std::invalid_argument(
            "Passed regrets and policy maps do not have the same number of elements");
      }
      std::for_each(exec_policy, policy_map.begin(), policy_map.end(), [&](auto& entry) {
         return std::get< 1 >(entry) = std::max(0., cumul_regret.at(std::get< 0 >(entry)))
                                       / pos_regret_sum;
      });
   } else {
      std::for_each(exec_policy, policy_map.begin(), policy_map.end(), [&](auto& entry) {
         return std::get< 1 >(entry) = 1. / static_cast< double >(policy_map.size());
      });
   }
}

/**
 * @brief Performs regret-matching on the given policy with respect to the provided regret
 *
 * @tparam Action
 * @tparam Policy
 */
template < typename Policy, typename RegretMap, typename ActionWrapper >
// clang-format off
requires
   concepts::map< RegretMap >
   and std::is_convertible_v< typename RegretMap::mapped_type, double>
   and std::invocable< ActionWrapper, typename fosg_auto_traits<Policy>::action_type >
   and concepts::action_policy<
      Policy
   >
// clang-format on
void regret_matching(
   Policy& policy_map,
   const RegretMap& cumul_regret,
   ActionWrapper action_wrapper =
      [](const typename fosg_auto_traits< Policy >::action_type& action) { return action; })
{
   // sum up the positivized regrets and store them in a new vector
   RegretMap pos_regrets;
   double pos_regret_sum{0.};
   for(const auto& [action, regret] : cumul_regret) {
      double pos_regret = std::max(0., regret);
      pos_regrets.emplace(action, pos_regret);
      pos_regret_sum += pos_regret;
   }
   // apply the new policy to the vector policy
   auto exec_policy{std::execution::par_unseq};
   if(pos_regret_sum > 0) {
      if(cumul_regret.size() != policy_map.size()) {
         throw std::invalid_argument(
            "Passed regrets and policy maps do not have the same number of elements");
      }
      std::for_each(exec_policy, policy_map.begin(), policy_map.end(), [&](auto& entry) {
         return std::get< 1 >(entry) = std::max(
                                          0., cumul_regret.at(action_wrapper(std::get< 0 >(entry))))
                                       / pos_regret_sum;
      });
   } else {
      std::for_each(exec_policy, policy_map.begin(), policy_map.end(), [&](auto& entry) {
         return std::get< 1 >(entry) = 1. / static_cast< double >(policy_map.size());
      });
   }
}

}  // namespace nor::rm

#endif  // NOR_RM_ALGORITHMS_HPP
