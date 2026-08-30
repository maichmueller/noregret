"""Session operation semantics on compiled games."""

import unittest

import _noregret as nor

from support import BindingTestCase, static_session


class StaticSessionTest(BindingTestCase):
    def test_rock_paper_scissors_iterates(self):
        session = static_session("rock_paper_scissors")
        result = session.iterate()
        self.assertEqual(result.iteration, 0)
        self.assertEqual(len(result.root_values), 2)
        self.assertEqual(
            {value.player for value in result.root_values},
            {nor.Player.alex, nor.Player.bob},
        )
        stats = session.stats()
        self.assertEqual(stats.iteration, 1)
        self.assertIs(stats.game, nor.GameId.rock_paper_scissors)
        self.assertIs(stats.profile, nor.ProfileId.vanilla_alternating)
        self.assertEqual(stats.player_count, 2)
        self.assertGreater(stats.current_policy_entries, 0)

    def test_kuhn_poker_iterates(self):
        session = static_session("kuhn_poker")
        session.iterate()
        stats = session.stats()
        self.assertIs(stats.game, nor.GameId.kuhn_poker)
        self.assertEqual(stats.player_count, 2)
        self.assertGreater(stats.current_policy_entries, 0)

    def test_advance_collects_nothing_and_counts_exactly(self):
        session = static_session("kuhn_poker")
        self.assertIsNone(session.advance(4))
        self.assertEqual(session.stats().iteration, 4)
        session.advance(0)
        self.assertEqual(session.stats().iteration, 4)

    def test_advance_last_returns_only_the_final_iteration(self):
        session = static_session("kuhn_poker")
        self.assertIsNone(session.advance_last(0))
        self.assertEqual(session.stats().iteration, 0)

        last = session.advance_last(3)
        self.assertIsNotNone(last)
        self.assertEqual(last.iteration, 2)
        self.assertEqual(session.stats().iteration, 3)

    def test_trace_collects_on_the_requested_cadence(self):
        session = static_session("kuhn_poker")
        session.advance(5)
        trace = session.trace(5, 2)
        self.assertEqual(trace.first_iteration, 5)
        self.assertEqual(trace.last_iteration, 10)
        self.assertEqual(len(trace), 2)
        self.assertEqual([step.iteration for step in trace.iterations], [6, 8])
        self.assertEqual(trace[0].iteration, 6)
        with self.assertRaises(IndexError):
            trace[2]

        # A final partial interval is intentionally not collected.
        self.assertEqual(len(session.trace(3, 4)), 0)

    def test_trace_rejects_a_zero_cadence(self):
        session = static_session("rock_paper_scissors")
        with self.assertCapabilityError(nor.CapabilityErrorCode.invalid_spec):
            session.trace(4, 0)

    def test_iterate_and_trace_agree_on_root_values(self):
        stepwise = static_session("kuhn_poker")
        values = [stepwise.iterate() for _ in range(4)]
        traced = static_session("kuhn_poker").trace(4, 1)
        self.assertEqual(len(traced), 4)
        for step, expected in zip(traced.iterations, values):
            self.assertEqual(step.iteration, expected.iteration)
            actual_by_player = {v.player: v.value for v in step.root_values}
            for value in expected.root_values:
                self.assertAlmostEqual(actual_by_player[value.player], value.value)

    def test_sampling_profiles_accept_their_knobs(self):
        session = static_session(
            "kuhn_poker", profile="mccfr_outcome_lazy", epsilon=0.4, seed=11
        )
        session.advance(5)
        self.assertEqual(session.stats().iteration, 5)

    def test_epsilon_outside_the_unit_interval_is_rejected(self):
        game = nor.Game("kuhn_poker")
        for epsilon in (-0.1, 1.5, float("nan"), float("inf")):
            with self.assertCapabilityError(nor.CapabilityErrorCode.invalid_spec):
                game.make_session("mccfr_outcome_lazy", epsilon=epsilon)

    def test_solver_and_profile_may_be_named_separately(self):
        game = nor.Game(nor.GameId.rock_paper_scissors)
        session = game.make_session(nor.SolverId.mccfr, nor.ProfileId.mccfr_pure_cfr)
        session.iterate()
        self.assertIs(session.stats().solver, nor.SolverId.mccfr)

        with self.assertCapabilityError(
            nor.CapabilityErrorCode.profile_solver_mismatch
        ):
            game.make_session(nor.SolverId.mccfr, nor.ProfileId.vanilla_alternating)

    def test_game_spec_fields_reach_the_environment(self):
        game = nor.Game(nor.GameId.goofspiel, deck_size=3)
        self.assertEqual(game.spec.find(nor.GameFieldId.deck_size), 3)
        game.make_session("vanilla_alternating").iterate()

        with self.assertCapabilityError(nor.CapabilityErrorCode.invalid_spec):
            # A field belonging to another game is not silently ignored.
            nor.Game(nor.GameId.goofspiel, n_faces=6)

    def test_unknown_and_reserved_identifiers_are_rejected(self):
        game = nor.Game("rock_paper_scissors")
        with self.assertRaises(ValueError):
            game.make_session("not_a_profile")
        with self.assertRaises(TypeError):
            game.make_session(object())

        # The dynamic identifier is reserved for provider-backed games and is not a static game.
        with self.assertCapabilityError(nor.CapabilityErrorCode.unknown_game):
            nor.Game(nor.GameId.dynamic)

    def test_stratego_default_is_playable(self):
        session = static_session("stratego")
        result = session.iterate()
        self.assertEqual(len(result.root_values), 2)
        self.assertGreater(session.stats().current_policy_entries, 0)


if __name__ == "__main__":
    unittest.main()
