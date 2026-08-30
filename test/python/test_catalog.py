"""The catalog is generated from reflection and from what the compiler admitted."""

import unittest

import _noregret as nor

from support import BindingTestCase

EXPECTED_GAMES = {
    "kuhn_poker",
    "leduc_poker",
    "rock_paper_scissors",
    "stratego",
    "texas_holdem_poker",
    "goofspiel",
    "three_player_goofspiel",
    "battleship",
    "battleship_gs",
    "dark_hex",
    "pursuit_evasion",
    "oshi_zumo",
    "shapley",
    "centipede",
    "colonel_blotto",
    "sheriff",
    "liars_dice",
}

EXPECTED_SOLVERS = {
    "vanilla_cfr",
    "cfr_plus",
    "lazy_cfr",
    "lazy_cfr_plus",
    "extragradient_cfr",
    "discounted_cfr",
    "linear_cfr",
    "exponential_cfr",
    "greedy_cfr",
    "mccfr",
    "mccfr_plus",
}


class CatalogTest(BindingTestCase):
    def test_game_roster_is_exactly_the_compiled_partitions(self):
        self.assertEqual({game.name for game in nor.games()}, EXPECTED_GAMES)
        self.assertEqual(len(nor.games()), len(EXPECTED_GAMES))

    def test_solver_roster_is_exact(self):
        self.assertEqual({solver.name for solver in nor.solvers()}, EXPECTED_SOLVERS)

    def test_names_come_from_reflection(self):
        # Every descriptor name is the reflected enumerator spelling, and the enumeration itself
        # carries that same name. Nothing here is a handwritten table.
        for game in nor.games():
            self.assertEqual(game.name, game.id.name)
            self.assertIs(getattr(nor.GameId, game.name), game.id)
        for solver in nor.solvers():
            self.assertEqual(solver.name, solver.id.name)
        for profile in nor.profiles():
            self.assertEqual(profile.name, profile.id.name)
        for field in nor.Game(nor.GameId.texas_holdem_poker).spec.fields:
            self.assertIsInstance(field.id, nor.GameFieldId)

    def test_every_listed_combination_is_constructible_and_no_other_is(self):
        listed = {(c.game, c.solver, c.profile) for c in nor.capabilities()}
        self.assertTrue(listed)
        for game in nor.games():
            per_game = {
                (c.game, c.solver, c.profile) for c in nor.capabilities_for(game.id)
            }
            self.assertTrue(per_game <= listed)
            self.assertEqual(
                per_game, {entry for entry in listed if entry[0] == game.id}
            )

        # A profile always belongs to the solver family it is listed under.
        by_profile = {profile.id: profile.solver for profile in nor.profiles()}
        for capability in nor.capabilities():
            self.assertEqual(by_profile[capability.profile], capability.solver)

        # Every combination the catalog advertises for one small game really constructs.
        game = nor.Game(nor.GameId.rock_paper_scissors)
        for capability in game.capabilities:
            session = game.make_session(capability.solver, capability.profile)
            self.assertFalse(session.closed)

    def test_profiles_for_partitions_the_profile_list(self):
        collected = []
        for solver in nor.solvers():
            for profile in nor.profiles_for(solver.id):
                self.assertEqual(profile.solver, solver.id)
                collected.append(profile)
        self.assertEqual(
            sorted(profile.name for profile in collected),
            sorted(profile.name for profile in nor.profiles()),
        )

    def test_dynamic_capabilities_are_reported_separately(self):
        dynamic = nor.dynamic_capabilities()
        self.assertTrue(dynamic)
        for capability in dynamic:
            self.assertIs(capability.game, nor.GameId.dynamic)
            self.assertEqual(capability.name, capability.profile.name)
        # The dynamic identifier is not a static game.
        self.assertNotIn(nor.GameId.dynamic, {game.id for game in nor.games()})

    def test_unknown_game_name_is_rejected(self):
        with self.assertRaises(ValueError):
            nor.Game("not_a_game")

    def test_descriptor_metadata_is_present(self):
        rps = next(game for game in nor.games() if game.name == "rock_paper_scissors")
        self.assertEqual(rps.min_players, 2)
        self.assertEqual(rps.max_players, 2)
        self.assertIs(rps.stochasticity, nor.Stochasticity.deterministic)
        self.assertEqual(rps.fields, [])

        holdem = next(
            game for game in nor.games() if game.name == "texas_holdem_poker"
        )
        names = {field.name for field in holdem.fields}
        self.assertIn("n_players", names)
        self.assertIn("small_blind", names)
        self.assertIn("deck_size", names)
        for field in holdem.fields:
            self.assertEqual(field.name, field.id.name)
            self.assertIsInstance(field.kind, nor.SpecKind)


if __name__ == "__main__":
    unittest.main()
