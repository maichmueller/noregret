# noregret

Python bindings for the [noregret](https://github.com/maichmueller/noregret) framework: a
factored-observation stochastic game library with a family of compiled counterfactual-regret
solvers.

The bindings are a thin, value-oriented surface over the compiled solver core. A game selects one
registered game/solver/profile combination once, and the solver then runs entirely in statically
dispatched C++ until the operation returns.

```python
import nor

game = nor.Game("kuhn_poker")
session = game.make_session("vanilla_alternating")

session.advance(1000)
print(session.stats().iteration)

policy = session.policy(nor.PolicyKind.average)
for player, info_state, actions in policy.to_entries():
    print(player, info_state.to_string(), [(a.to_string(), p) for a, p in actions])
```

## What the catalog offers

`nor.games()`, `nor.solvers()`, `nor.profiles()` and `nor.capabilities()` describe exactly the
combinations the compiler admitted -- nothing is advertised that cannot be constructed. Names and
enumerators come from C++26 reflection over the same definitions the solvers use.

```python
nor.capabilities_for(nor.GameId.leduc_poker)   # every profile this game admits
nor.profiles_for(nor.SolverId.mccfr)           # every profile in one solver family
```

Game specifications retain the concrete game's rules. Texas hold'em uses the standard 52-card
deck by default and also accepts a smaller prefix of that canonical deck when a bounded finite
variant is useful for experiments or compatibility checks; the deck must still contain every hole
and community card required by the configured player count.

## Sessions

A session exposes one coarse operation set:

| operation | meaning |
| --- | --- |
| `iterate()` | run exactly one iteration and return its root values |
| `advance(n)` | run `n` iterations, collecting nothing |
| `advance_last(n)` | run `n` iterations, return only the last result (`None` for `n == 0`) |
| `trace(n, every=1)` | run `n` iterations, collect after every `every`-th one |
| `stats()` | iteration/cycle counts and policy sizes |
| `policy(kind)` | a borrowed view of the current or average policy |

A policy handle borrows solver storage rather than copying it. It reports `valid == False` once the
session advances or is released, and raises `StaleViewError` if used anyway. `to_entries()` is the
explicit bulk copy; **its row order is unspecified** because it follows the solver's hash map.

## Pure-Python games

Subclass `nor.DynamicEnvironmentProvider` to play a game written in Python against the same
compiled solvers. Every value the provider returns is validated on its way into the solver -- at
the initial state and at every state reached afterwards -- and a violation surfaces as a
`CapabilityError` with a `code`, not as undefined behaviour.

```python
class Matching(nor.DynamicEnvironmentProvider):
    def max_player_count(self): return 2
    def player_count(self): return 2
    def stochasticity(self): return nor.Stochasticity.deterministic
    def initial_world_state(self): return nor.WorldState("matching", "root")
    def players(self, state): return [nor.Player.alex, nor.Player.bob]
    def active_player(self, state):
        return nor.Player.alex if state.identity == "root" else nor.Player.bob
    ...

session = nor.Game(Matching()).make_session("vanilla_alternating")
```

The provider's roster must be the same at every state; use `is_partaking` to mark a participant as
out of the hand.

## Errors

Every failure that the runtime can describe becomes a `nor.CapabilityError` carrying `code`, and
`game`/`solver`/`profile` when they are known. `StaleViewError`, `ClosedSessionError` and
`ReentrantSessionError` cover handle lifetime and same-session reentrancy.
