
#ifndef NOR_VANILLA_HPP
#define NOR_VANILLA_HPP

#include <queue>

#include "nor/concepts.hpp"
#include "nor/policy.hpp"

namespace nor {

/**
 * A (Vanilla) Counterfactual Regret Minimization algorithm class following the Factored-Observation
 * Stochastic Games (FOSG) formalism.
 * @tparam Game, the game type to run CFR on.
 * or a neural network type for estimating values.
 *
 */
template <
   concepts::fosg Game,
   concepts::policy< typename Game::info_state_type, typename Game::action_type > Policy >
class CFR {
   explicit CFR(Game&& game, Policy&& policy = Policy())
       : m_game(std::forward< Game >(game)),
         m_policy(std::forward< Policy >(policy)),
         m_avg_policy()
   {
   }

   /**
    * @brief Executes n iterations of the CFR algorithm in unrolled form (no recursion).
    * @param n_iters, the number of iterations to perform.
    * @return the updated policy
    */
   const Policy* iterate(int n_iters = 1);

  private:
   Game m_game;
   std::map< Player, Policy > m_policy;
   std::map< Player, Policy > m_avg_policy;
};

template <
   concepts::fosg Game,
   concepts::policy< typename Game::info_state_type, typename Game::action_type > Policy >
const Policy* CFR< Game, Policy >::iterate(int n_iters)
{
   auto& game_state = m_game.world_state();
   bool is_terminal = m_game.is_terminal(game_state);
   double value = 0;
   auto actions = m_game.actions(m_game.active_player());
   using sub_tree_tuple = std::
      tuple< Game, typename Game::action_type, Player, unsigned int, double, double >;
   std::queue<sub_tree_tuple> sub_tree_queue;
   std::vector< double > cf_value(actions.size(), 0);
   for(auto action : actions) {
      while(true) {

      }
   }

   if() {
      return m_game.reward(game_state);
   }

   return &m_policy;
}

///// helper function for building CFR from a game later maybe?
// template <typename...Args>
// CFR<Args...> make_cfr(Args&&... args);

}  // namespace nor

#endif  // NOR_VANILLA_HPP
