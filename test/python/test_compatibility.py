"""Exercise the Python binding against the compiler-admitted capability catalog."""

import unittest

import _noregret as nor

from support import BindingTestCase, ChanceProvider, DeterministicProvider


def capability_key(capability):
    return capability.game, capability.solver, capability.profile


def capability_label(capability):
    return (
        f"game={capability.game.name}, solver={capability.solver.name}, "
        f"profile={capability.profile.name}"
    )


class StaticCompatibilityTest(BindingTestCase):
    def test_game_views_match_the_global_capability_catalog(self):
        catalog = tuple(nor.capabilities())
        catalog_keys = {capability_key(capability) for capability in catalog}
        self.assertTrue(catalog_keys)
        self.assertEqual(
            len(catalog), len(catalog_keys), "global catalog contains duplicates"
        )

        expected_by_game = {}
        for capability in catalog:
            expected_by_game.setdefault(capability.game, set()).add(
                capability_key(capability)
            )

        seen_from_game_views = set()
        for descriptor in nor.games():
            game = nor.Game(descriptor.id)
            expected = expected_by_game.get(descriptor.id, set())
            catalog_view = tuple(nor.capabilities_for(descriptor.id))
            game_view = tuple(game.capabilities)
            catalog_view_keys = {
                capability_key(capability) for capability in catalog_view
            }
            game_view_keys = {capability_key(capability) for capability in game_view}
            game_label = f"game={descriptor.id.name}"

            self.assertEqual(
                catalog_view_keys,
                expected,
                f"{game_label}: capabilities_for disagrees with nor.capabilities()",
            )
            self.assertEqual(
                game_view_keys,
                expected,
                f"{game_label}: Game.capabilities disagrees with nor.capabilities()",
            )
            self.assertEqual(
                len(catalog_view),
                len(catalog_view_keys),
                f"{game_label}: capabilities_for contains duplicate tuples",
            )
            self.assertEqual(
                len(game_view),
                len(game_view_keys),
                f"{game_label}: Game.capabilities contains duplicate tuples",
            )
            seen_from_game_views.update(game_view_keys)

        self.assertEqual(seen_from_game_views, catalog_keys)
        for capability in catalog:
            self.assertIn(
                capability_key(capability),
                seen_from_game_views,
                f"{capability_label(capability)} is missing from its Game.capabilities view",
            )

    def test_every_advertised_static_capability_advances_once(self):
        for capability in nor.capabilities():
            label = capability_label(capability)
            with self.subTest(
                game=capability.game.name,
                solver=capability.solver.name,
                profile=capability.profile.name,
            ):
                try:
                    session = nor.Game(capability.game).make_session(
                        capability.solver, capability.profile
                    )
                    result = session.advance(1)
                except Exception as error:  # noqa: BLE001 - add the capability context
                    self.fail(f"{label}: {type(error).__name__}: {error}")
                self.assertFalse(session.closed, f"{label}: session closed")
                self.assertIsNone(result, f"{label}: advance(1) returned a result")


class DynamicCompatibilityTest(BindingTestCase):
    def test_every_dynamic_capability_runs_for_each_valid_provider_shape(self):
        catalog = tuple(nor.dynamic_capabilities())
        catalog_keys = {capability_key(capability) for capability in catalog}
        self.assertTrue(catalog_keys)
        self.assertEqual(
            len(catalog), len(catalog_keys), "dynamic catalog contains duplicates"
        )

        for provider_type in (DeterministicProvider, ChanceProvider):
            provider = provider_type()
            game = nor.Game(provider)
            game_view_keys = {
                capability_key(capability) for capability in game.capabilities
            }
            provider_label = provider_type.__name__
            self.assertEqual(
                game_view_keys,
                catalog_keys,
                f"{provider_label}: dynamic Game.capabilities disagrees with the catalog",
            )

            for capability in catalog:
                label = capability_label(capability)
                with self.subTest(
                    provider=provider_label,
                    game=capability.game.name,
                    solver=capability.solver.name,
                    profile=capability.profile.name,
                ):
                    try:
                        session = game.make_session(
                            capability.solver, capability.profile
                        )
                        result = session.advance(1)
                    except (
                        Exception
                    ) as error:  # noqa: BLE001 - add the capability context
                        self.fail(
                            f"{label} provider={provider_label}: "
                            f"{type(error).__name__}: {error}"
                        )
                    self.assertTrue(session.dynamic, f"{label}: session is not dynamic")
                    self.assertFalse(session.closed, f"{label}: session closed")
                    self.assertIsNone(result, f"{label}: advance(1) returned a result")


if __name__ == "__main__":
    unittest.main()
