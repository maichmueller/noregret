
#ifndef NOR_VANILLA_HPP
#define NOR_VANILLA_HPP

#include "nor/concepts.hpp"

namespace nor {

/**
 * A (Vanilla) Counterfactual Regret Minimization algorithm class following the Factored-Observation
 * Stochastic Games (FOSG) formalism.
 * @tparam Game, the game type to run CFR on.
 * @tparam Valuator, the type for providing the value of an infostate. Eg a table for tabular CFR
 * or a neural network type for estimating values.
 *
 */
template <
   typename Game,
   typename Valuator,
   concepts::action Action,
   concepts::infoset< Action > Infoset,
   concepts::policy< Infoset, Action > Policy >
class CFR {

};

}  // namespace nor

#endif  // NOR_VANILLA_HPP
