"""Shared helpers for the binding test suite.

The tests import the extension module directly rather than through the ``nor`` package, so that a
failure in the package layout shows up in the dedicated packaging test instead of disabling every
other test.
"""

import unittest

import _noregret as nor


# nanobind types accept exactly one bound base, so the shared game vocabulary lives in module level
# helpers rather than in a mixin.
#
# Every dynamic provider in this suite plays the same tiny two-player game:
# ``root`` (alex) -> ``after`` (bob) -> ``terminal``, with alex winning. It is deliberately minimal;
# the subject of these tests is the boundary, not the game.
WORLD_TYPE = "test.world"
ACTION_TYPE = "test.action"
OUTCOME_TYPE = "test.outcome"
OBSERVATION_TYPE = "test.observation"


def world(identity):
    return nor.WorldState(WORLD_TYPE, identity)


def action(identity):
    return nor.Action(ACTION_TYPE, identity)


def outcome(identity):
    return nor.ChanceOutcome(OUTCOME_TYPE, identity)


def observation(identity):
    return nor.Observation(OBSERVATION_TYPE, identity)


class DeterministicProvider(nor.DynamicEnvironmentProvider):
    """A well formed, purely deterministic pure-Python game."""

    def __init__(self):
        super().__init__()
        self.calls = 0

    def _touch(self):
        self.calls += 1

    def max_player_count(self):
        return 2

    def player_count(self):
        return 2

    def stochasticity(self):
        return nor.Stochasticity.deterministic

    def initial_world_state(self):
        return world("root")

    def players(self, state):
        self._touch()
        return [nor.Player.alex, nor.Player.bob]

    def active_player(self, state):
        self._touch()
        return nor.Player.alex if state.identity == "root" else nor.Player.bob

    def is_terminal(self, state):
        self._touch()
        return state.identity == "terminal"

    def is_partaking(self, state, player):
        return True

    def actions(self, player, state):
        self._touch()
        if state.identity == "terminal":
            return []
        return [action("left"), action("right")]

    def reward(self, player, state):
        self._touch()
        if state.identity != "terminal":
            return 0.0
        return 1.0 if player == nor.Player.alex else -1.0

    def transition_action(self, state, action):
        self._touch()
        return world("after" if state.identity == "root" else "terminal")

    def private_observation_action(self, player, state, action, next_state):
        self._touch()
        return observation(action.identity)

    def public_observation_action(self, state, action, next_state):
        self._touch()
        return observation(action.identity)


class ChanceProvider(DeterministicProvider):
    """The same game behind one chance move, so the choice superset is genuinely exercised."""

    def stochasticity(self):
        return nor.Stochasticity.choice

    def initial_world_state(self):
        return world("chance_root")

    def active_player(self, state):
        self._touch()
        if state.identity == "chance_root":
            return nor.Player.chance
        return nor.Player.alex if state.identity.startswith("after") else nor.Player.bob

    def chance_actions(self, state):
        self._touch()
        if state.identity != "chance_root":
            return []
        return [outcome("heads"), outcome("tails")]

    def chance_probability(self, state, outcome):
        self._touch()
        return 0.5

    def transition_chance(self, state, outcome):
        self._touch()
        return world("after_" + outcome.identity)

    def transition_action(self, state, action):
        self._touch()
        if state.identity.startswith("after"):
            return world("terminal")
        return world("after_plain")

    def private_observation_chance(self, player, state, outcome, next_state):
        self._touch()
        # Only the public side of a chance move is shared; keeping the private side per player is
        # what makes the two branches distinguishable to their owner alone.
        return observation(outcome.identity if player == nor.Player.alex else "hidden")

    def public_observation_chance(self, state, outcome, next_state):
        self._touch()
        return observation("dealt")


def static_session(game_name, profile="vanilla_alternating", **kwargs):
    return nor.Game(game_name).make_session(profile, **kwargs)


class BindingTestCase(unittest.TestCase):
    def assertCapabilityError(self, code=None):
        """Context manager asserting a CapabilityError, optionally with a specific code."""
        outer = self

        class _Checker:
            def __enter__(self):
                return self

            def __exit__(self, exc_type, exc, traceback):
                outer.assertIsNotNone(exc, "expected a CapabilityError")
                outer.assertTrue(
                    issubclass(exc_type, nor.CapabilityError),
                    f"expected CapabilityError, got {exc_type}: {exc}",
                )
                if code is not None:
                    outer.assertEqual(exc.code, code, str(exc))
                self.exception = exc
                return True

        return _Checker()
