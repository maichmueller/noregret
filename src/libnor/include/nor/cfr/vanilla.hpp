
#ifndef NOR_VANILLA_HPP
#define NOR_VANILLA_HPP

#include "nor/concepts.hpp"

namespace nor {




/**
 * A (Vanilla) Counterfactual Regret Minimization algorithm class following the Factored-Observation
 * Stochastic Games (FOSG) formalism.
 * @tparam Game, the game type to run CFR on.
 * @tparam Valuator, the type providing the value of an istate. Eg a table for tabular CFR
 * or a neural network type for estimating values.
 *
 */
//template <
//   concepts::game Game,
//   typename Valuator,
//   concepts::action Action,
//   concepts::istate Infostate,
//   concepts::policy< Infostate, Action > Policy >
//class CFR {
//
//};

template <
   concepts::fosg Game,
   concepts::action Action,
   concepts::info_state Infostate,
   concepts::policy< Infostate, Action > Policy >
class CFR {

};



///// helper function for building CFR from a game later maybe?
//template <typename...Args>
//CFR<Args...> make_cfr(Args&&... args);


}  // namespace nor

#endif  // NOR_VANILLA_HPP
