"""Pure-Python games run the same compiled solvers through the provider boundary."""

import gc
import unittest

import _noregret as nor

from support import (
    BindingTestCase,
    ChanceProvider,
    DeterministicProvider,
    action,
)


class Malformed(DeterministicProvider):
    """A provider that misbehaves at one named point, and only there."""

    def __init__(self, fault, late=False):
        super().__init__()
        self.fault = fault
        self.late = late

    def _faulty_here(self, state):
        if not self.late:
            return True
        return state.identity != "root"

    def max_player_count(self):
        return 0 if self.fault == "max_players" else 2

    def player_count(self):
        return 1 if self.fault == "player_count" else 2

    def stochasticity(self):
        if self.fault == "sample":
            return nor.Stochasticity.sample
        return super().stochasticity()

    def serialized(self):
        return self.fault != "serialized"

    def initial_world_state(self):
        if self.fault == "empty_world":
            return nor.WorldState("test.world", "")
        return super().initial_world_state()

    def players(self, state):
        if self.fault == "roster" and self._faulty_here(state):
            return [nor.Player.alex, nor.Player.alex]
        if self.fault == "foreign_roster" and self._faulty_here(state):
            return [nor.Player.alex, nor.Player.cedric]
        return super().players(state)

    def active_player(self, state):
        if self.fault == "active" and self._faulty_here(state):
            return nor.Player.unknown
        return super().active_player(state)

    def actions(self, player, state):
        if self.fault == "empty_actions" and self._faulty_here(state):
            return []
        if self.fault == "duplicate_actions" and self._faulty_here(state):
            return [action("left"), action("left")]
        if self.fault == "invalid_action" and self._faulty_here(state):
            return [action("left"), nor.Action("", "")]
        return super().actions(player, state)

    def reward(self, player, state):
        if self.fault == "reward" and state.identity == "terminal":
            return float("nan")
        return super().reward(player, state)

    def transition_action(self, state, action):
        if self.fault == "bad_world" and self._faulty_here(state):
            return nor.WorldState("test.world", "")
        return super().transition_action(state, action)

    def private_observation_action(self, player, state, action, next_state):
        if self.fault == "observation" and self._faulty_here(state):
            return nor.Observation("", "")
        return super().private_observation_action(player, state, action, next_state)


class BadChanceProvider(ChanceProvider):
    def __init__(self, fault):
        super().__init__()
        self.fault = fault

    def chance_probability(self, state, outcome):
        if self.fault == "sum":
            return 0.6
        if self.fault == "negative":
            return -0.5
        if self.fault == "nan":
            return float("nan")
        return super().chance_probability(state, outcome)

    def chance_actions(self, state):
        outcomes = super().chance_actions(state)
        if self.fault == "duplicate" and outcomes:
            return [outcomes[0], outcomes[0]]
        if self.fault == "empty":
            return []
        return outcomes


class DeterministicProviderTest(BindingTestCase):
    def test_a_pure_python_game_runs_a_compiled_solver(self):
        provider = DeterministicProvider()
        game = nor.Game(provider)
        self.assertTrue(game.is_dynamic)
        self.assertIs(game.id, nor.GameId.dynamic)
        self.assertEqual(game.min_players, 2)
        self.assertEqual(game.max_players, 2)

        session = game.make_session("vanilla_alternating")
        self.assertTrue(session.dynamic)
        before = provider.calls
        result = session.iterate()
        self.assertGreater(provider.calls, before)
        self.assertEqual(result.iteration, 0)

        stats = session.stats()
        self.assertIs(stats.game, nor.GameId.dynamic)
        self.assertEqual(stats.player_count, 2)
        self.assertGreater(stats.current_policy_entries, 0)

    def test_every_advertised_dynamic_capability_constructs(self):
        game = nor.Game(DeterministicProvider())
        for capability in nor.dynamic_capabilities():
            session = game.make_session(capability.solver, capability.profile)
            session.iterate()

    def test_dynamic_sessions_support_the_same_operations(self):
        session = nor.Game(DeterministicProvider()).make_session("vanilla_alternating")
        session.advance(2)
        self.assertEqual(session.stats().iteration, 2)
        last = session.advance_last(2)
        self.assertEqual(last.iteration, 3)
        trace = session.trace(4, 2)
        self.assertEqual([step.iteration for step in trace.iterations], [5, 7])

    def test_policy_rows_of_a_dynamic_game_carry_provider_values(self):
        session = nor.Game(DeterministicProvider()).make_session("vanilla_alternating")
        session.advance(2)
        rows = session.policy().to_entries()
        self.assertTrue(rows)
        player, info_state, actions = rows[0]
        self.assertIn(player, (nor.Player.alex, nor.Player.bob))
        self.assertTrue(actions)
        identities = {action.identity for action, _ in actions}
        self.assertEqual(identities, {"left", "right"})
        self.assertAlmostEqual(sum(probability for _, probability in actions), 1.0)


class ChanceProviderTest(BindingTestCase):
    def test_a_chance_provider_runs_through_the_choice_superset(self):
        provider = ChanceProvider()
        session = nor.Game(provider).make_session("vanilla_alternating")
        result = session.iterate()
        self.assertEqual(len(result.root_values), 2)
        self.assertGreater(provider.calls, 0)
        self.assertEqual(session.stats().player_count, 2)

    def test_chance_sampling_profiles_work_on_a_python_chance_game(self):
        session = nor.Game(ChanceProvider()).make_session(
            "mccfr_chance_sampling", seed=3
        )
        session.advance(4)
        self.assertEqual(session.stats().iteration, 4)


class MalformedProviderTest(BindingTestCase):
    def assertRejectedAtAdmission(self, provider, fragment):
        with self.assertCapabilityError(
            nor.CapabilityErrorCode.invalid_dynamic_provider
        ) as checker:
            nor.Game(provider)
        self.assertIn(fragment, str(checker.exception))

    def assertRejectedWhileRunning(self, provider, fragment):
        # Admissible at its initial state; only a state the solver reaches later is malformed.
        game = nor.Game(provider)
        session = game.make_session("vanilla_alternating")
        with self.assertCapabilityError(
            nor.CapabilityErrorCode.invalid_dynamic_provider
        ) as checker:
            session.iterate()
        self.assertIn(fragment, str(checker.exception))

    def test_declarations_are_checked_at_admission(self):
        self.assertRejectedAtAdmission(Malformed("max_players"), "max_player_count")
        self.assertRejectedAtAdmission(Malformed("player_count"), "player_count")
        self.assertRejectedAtAdmission(Malformed("sample"), "stochasticity::sample")
        self.assertRejectedAtAdmission(Malformed("serialized"), "serialized() == true")
        self.assertRejectedAtAdmission(Malformed("empty_world"), "initial_world_state()")

    def test_initial_state_faults_are_checked_at_admission(self):
        self.assertRejectedAtAdmission(Malformed("roster"), "duplicate players")
        self.assertRejectedAtAdmission(Malformed("active"), "neither chance nor an admitted")
        self.assertRejectedAtAdmission(Malformed("empty_actions"), "no legal actions")
        self.assertRejectedAtAdmission(
            Malformed("duplicate_actions"), "legal actions must be unique"
        )
        self.assertRejectedAtAdmission(
            Malformed("invalid_action"), "legal actions must have"
        )

    def test_later_reachable_state_faults_are_checked_too(self):
        self.assertRejectedWhileRunning(
            Malformed("empty_actions", late=True), "no legal actions"
        )
        self.assertRejectedWhileRunning(
            Malformed("duplicate_actions", late=True), "legal actions must be unique"
        )
        self.assertRejectedWhileRunning(
            Malformed("invalid_action", late=True), "legal actions must have"
        )
        self.assertRejectedWhileRunning(
            Malformed("active", late=True), "neither chance nor an admitted"
        )
        self.assertRejectedWhileRunning(
            Malformed("roster", late=True), "duplicate players"
        )
        self.assertRejectedWhileRunning(
            Malformed("foreign_roster", late=True), "outside the admitted roster"
        )
        self.assertRejectedWhileRunning(Malformed("reward"), "non-finite")
        self.assertRejectedWhileRunning(
            Malformed("bad_world", late=True), "world state without a type name"
        )
        self.assertRejectedWhileRunning(
            Malformed("observation", late=True), "observation without a type name"
        )

    def test_chance_distributions_are_checked(self):
        self.assertRejectedAtAdmission(
            BadChanceProvider("sum"), "sum approximately to one"
        )
        self.assertRejectedAtAdmission(
            BadChanceProvider("negative"), "finite and nonnegative"
        )
        self.assertRejectedAtAdmission(BadChanceProvider("nan"), "finite and nonnegative")
        self.assertRejectedAtAdmission(
            BadChanceProvider("duplicate"), "chance outcomes must be unique"
        )
        self.assertRejectedAtAdmission(
            BadChanceProvider("empty"), "no chance outcomes"
        )

    def test_a_provider_that_raises_is_reported_rather_than_escaping(self):
        class Raising(DeterministicProvider):
            def actions(self, player, state):
                raise RuntimeError("provider exploded")

        with self.assertRaises(Exception) as caught:
            nor.Game(Raising())
        self.assertIn("provider exploded", str(caught.exception))

    def test_a_provider_returning_the_wrong_type_is_rejected(self):
        class WrongType(DeterministicProvider):
            def actions(self, player, state):
                return ["left", "right"]

        with self.assertRaises(Exception):
            nor.Game(WrongType())


class ProviderLifetimeTest(BindingTestCase):
    def test_the_session_keeps_the_provider_alive(self):
        provider = DeterministicProvider()
        session = nor.Game(provider).make_session("vanilla_alternating")
        del provider
        gc.collect()
        # If the provider had been collected the traversal would call into freed storage.
        session.advance(2)
        self.assertEqual(session.stats().iteration, 2)

    def test_the_game_keeps_the_provider_alive(self):
        game = nor.Game(DeterministicProvider())
        gc.collect()
        session = game.make_session("vanilla_alternating")
        session.iterate()

    def test_a_session_outlives_the_game_it_came_from(self):
        session = nor.Game(DeterministicProvider()).make_session("vanilla_alternating")
        gc.collect()
        session.advance(2)
        self.assertEqual(session.stats().iteration, 2)


class ReentrancyTest(BindingTestCase):
    """A provider callback runs on the solver's own thread, inside a session operation.

    The callbacks used here hook ``active_player`` rather than ``actions``: the solver queries the
    legal actions only while it is building an information node, so ``actions`` is not called again
    on later iterations, whereas ``active_player`` is consulted at every visited node.

    Each test records only the *type* of what it observed and drops its reference to the session at
    the end. A provider that keeps a session -- or an exception object, whose traceback reaches the
    frame the session lives in -- forms a reference cycle Python's collector cannot break, because
    the other side of the link is a C++ ``shared_ptr`` holding the provider's Python object alive.
    """

    def test_a_provider_callback_may_not_reenter_its_own_session(self):
        class Reentrant(DeterministicProvider):
            session = None
            observed = None

            def active_player(self, state):
                if self.session is not None and self.observed is None:
                    try:
                        self.session.iterate()
                        self.observed = "no error"
                    except BaseException as error:  # noqa: BLE001 - classified, not propagated
                        self.observed = type(error)
                return super().active_player(state)

        provider = Reentrant()
        session = nor.Game(provider).make_session("vanilla_alternating")
        provider.session = session
        try:
            session.iterate()
            self.assertIs(provider.observed, nor.ReentrantSessionError)
        finally:
            provider.session = None

    def test_a_provider_callback_may_not_reenter_through_a_policy_handle(self):
        class PolicyReentrant(DeterministicProvider):
            policy = None
            observed = None

            def active_player(self, state):
                if self.policy is not None and self.observed is None:
                    try:
                        self.policy.to_entries()
                        self.observed = "no error"
                    except BaseException as error:  # noqa: BLE001
                        self.observed = type(error)
                return super().active_player(state)

        provider = PolicyReentrant()
        session = nor.Game(provider).make_session("vanilla_alternating")
        session.iterate()
        provider.policy = session.policy()
        try:
            session.iterate()
            self.assertIs(provider.observed, nor.ReentrantSessionError)
        finally:
            provider.policy = None

    def test_a_separate_session_is_not_blocked_by_reentrancy_detection(self):
        # Reentrancy is per session, not per thread: an unrelated session remains usable.
        class Nested(DeterministicProvider):
            other = None
            nested_iterations = 0

            def active_player(self, state):
                if self.other is not None and self.nested_iterations == 0:
                    self.other.iterate()
                    self.nested_iterations += 1
                return super().active_player(state)

        provider = Nested()
        session = nor.Game(provider).make_session("vanilla_alternating")
        other = nor.Game(DeterministicProvider()).make_session("vanilla_alternating")
        provider.other = other
        try:
            session.iterate()
            self.assertEqual(provider.nested_iterations, 1)
            self.assertEqual(other.stats().iteration, 1)
        finally:
            provider.other = None


if __name__ == "__main__":
    unittest.main()
