"""High-value regression coverage for the public Nanobind boundary contract."""

import unittest

import _noregret as nor

from support import BindingTestCase, DeterministicProvider, static_session


class MutableDeclarations(DeterministicProvider):
    """A provider whose declarations can change after its Game has been admitted."""

    def __init__(self):
        super().__init__()
        self.declared_max_players = 2
        self.declared_players = 2
        self.declared_stochasticity = nor.Stochasticity.deterministic

    def max_player_count(self):
        return self.declared_max_players

    def player_count(self):
        return self.declared_players

    def stochasticity(self):
        return self.declared_stochasticity


class APIContractTest(BindingTestCase):
    def test_static_and_dynamic_games_share_the_same_game_and_session_surface(self):
        games = [nor.Game("rock_paper_scissors"), nor.Game(DeterministicProvider())]
        game_surface = (
            "id",
            "name",
            "is_dynamic",
            "min_players",
            "max_players",
            "stochasticity",
            "spec",
            "capabilities",
            "make_session",
        )
        for game in games:
            for attribute in game_surface:
                self.assertTrue(hasattr(game, attribute), attribute)

        sessions = [game.make_session("vanilla_alternating") for game in games]
        self.assertIs(type(sessions[0]), type(sessions[1]))
        session_surface = (
            "iterate",
            "advance",
            "advance_last",
            "trace",
            "stats",
            "policy",
            "dynamic",
            "closed",
        )
        for session in sessions:
            for attribute in session_surface:
                self.assertTrue(hasattr(session, attribute), attribute)
            self.assertEqual(session.iterate().iteration, 0)
            self.assertIsNone(session.advance(1))
            self.assertEqual(session.advance_last(1).iteration, 2)
            trace = session.trace(2, 2)
            self.assertEqual([step.iteration for step in trace.iterations], [4])

    def test_all_dynamic_domain_values_preserve_kind_and_missing_representations(self):
        values = (
            (nor.ValueKind.action, nor.Action("contract.action", "left")),
            (nor.ValueKind.chance_outcome, nor.ChanceOutcome("contract.outcome", "heads")),
            (nor.ValueKind.observation, nor.Observation("contract.observation", "seen")),
            (nor.ValueKind.info_state, nor.InfoState(nor.Player.alex)),
            (nor.ValueKind.public_state, nor.PublicState()),
            (nor.ValueKind.world_state, nor.WorldState("contract.world", "root")),
        )
        for kind, value in values:
            self.assertIs(value.kind, kind)
            self.assertTrue(value.valid)
            self.assertTrue(value.is_dynamic)
            self.assertIsNone(value.to_string())
            self.assertIsNone(value.to_tensor())
            self.assertFalse(value.capabilities.to_string)
            self.assertFalse(value.capabilities.to_tensor)

        tensor = nor.TensorData([1.0], [1])
        for value in (
            nor.Action("contract.action", "left", "left", tensor),
            nor.ChanceOutcome("contract.outcome", "heads", "heads", tensor),
            nor.Observation("contract.observation", "seen", "seen", tensor),
            nor.WorldState("contract.world", "root", "root", tensor),
        ):
            self.assertEqual(value.to_string(), value.identity)
            self.assertEqual(value.to_tensor(), tensor)

    def test_reflected_game_spec_accepts_the_full_uint64_range(self):
        spec = nor.GameSpec(nor.GameId.goofspiel, deck_size=2**63)
        self.assertIs(spec.game, nor.GameId.goofspiel)
        self.assertEqual(spec.find(nor.GameFieldId.deck_size), 2**63)

        spec.set(nor.GameFieldId.deck_size, 2**64 - 1)
        self.assertEqual(spec.find(nor.GameFieldId.deck_size), 2**64 - 1)

        with self.assertRaises(TypeError):
            nor.GameSpec(nor.GameId.goofspiel, deck_size=2**64)

    def test_capability_errors_keep_context_for_invalid_combinations(self):
        game = nor.Game(nor.GameId.rock_paper_scissors)

        with self.assertCapabilityError(nor.CapabilityErrorCode.invalid_spec) as failure:
            game.make_session("mccfr_outcome_lazy", epsilon=2.0)
        self.assertIs(failure.exception.game, nor.GameId.rock_paper_scissors)
        self.assertIs(failure.exception.solver, nor.SolverId.mccfr)
        self.assertIs(failure.exception.profile, nor.ProfileId.mccfr_outcome_lazy)

        with self.assertCapabilityError(
            nor.CapabilityErrorCode.profile_solver_mismatch
        ) as failure:
            game.make_session(nor.SolverId.mccfr, nor.ProfileId.vanilla_alternating)
        self.assertIs(failure.exception.game, nor.GameId.rock_paper_scissors)
        self.assertIs(failure.exception.solver, nor.SolverId.mccfr)
        self.assertIs(failure.exception.profile, nor.ProfileId.vanilla_alternating)

    def test_zero_iteration_operations_do_not_stale_borrowed_policy_views(self):
        session = static_session("rock_paper_scissors")
        session.iterate()
        policy = session.policy()
        row = policy.at(policy.to_entries()[0][1])

        session.advance(0)
        self.assertTrue(policy.valid)
        self.assertTrue(row.valid)

        self.assertIsNone(session.advance_last(0))
        self.assertTrue(policy.valid)
        self.assertTrue(row.valid)

        self.assertEqual(len(session.trace(0, 3)), 0)
        self.assertTrue(policy.valid)
        self.assertTrue(row.valid)
        self.assertTrue(policy.to_entries())

    def test_sessions_reuse_the_game_admission_snapshot(self):
        provider = MutableDeclarations()
        game = nor.Game(provider)

        # The provider is malformed relative to the already admitted Game. The session must use
        # the same immutable certificate that supplies the public metadata, rather than re-admit a
        # second, contradictory declaration set.
        provider.declared_max_players = 1
        provider.declared_players = 1
        provider.declared_stochasticity = nor.Stochasticity.choice

        self.assertEqual((game.min_players, game.max_players), (2, 2))
        self.assertIs(game.stochasticity, nor.Stochasticity.deterministic)
        session = game.make_session("vanilla_alternating")
        self.assertEqual(session.iterate().iteration, 0)


if __name__ == "__main__":
    unittest.main()
