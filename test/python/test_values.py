"""Generic domain values expose only representations that actually exist."""

import unittest

import _noregret as nor

from support import BindingTestCase, static_session


class ValueTest(BindingTestCase):
    def test_compiled_values_report_their_reflected_type(self):
        session = static_session("rock_paper_scissors")
        session.iterate()
        policy = session.policy()
        _, info_state, actions = policy.to_entries()[0]
        action = actions[0][0]

        self.assertIsInstance(action, nor.Action)
        self.assertFalse(action.is_dynamic)
        self.assertTrue(action.valid)
        self.assertIs(action.kind, nor.ValueKind.action)
        self.assertEqual(action.type_name, "Action")
        self.assertIsNone(action.identity)

        # rock/paper/scissors is printable, so a string representation exists and is used as is.
        self.assertTrue(action.capabilities.to_string)
        self.assertIn(action.to_string(), {"rock", "paper", "scissors"})
        # No tensor is invented for a value that does not provide one.
        self.assertFalse(action.capabilities.to_tensor)
        self.assertIsNone(action.to_tensor())

        self.assertIsInstance(info_state, nor.InfoState)
        self.assertIs(info_state.kind, nor.ValueKind.info_state)

    def test_equality_and_hashing_follow_the_underlying_value(self):
        session = static_session("rock_paper_scissors")
        session.iterate()
        policy = session.policy()
        rows = policy.to_entries()
        first_actions = [action for action, _ in rows[0][2]]

        self.assertEqual(first_actions[0], first_actions[0])
        self.assertNotEqual(first_actions[0], first_actions[1])
        self.assertEqual(hash(first_actions[0]), hash(first_actions[0]))
        self.assertEqual(len({first_actions[0], first_actions[0]}), 1)
        # Comparing against an unrelated object is not an error.
        self.assertNotEqual(first_actions[0], object())
        self.assertNotEqual(first_actions[0], 3)

    def test_provider_values_carry_the_representations_they_were_given(self):
        tensor = nor.TensorData([1.0, 2.0], [2])
        described = nor.Action("py.action", "left", "the left branch", tensor)
        plain = nor.Action("py.action", "left")
        other = nor.Action("py.action", "right")

        self.assertTrue(described.is_dynamic)
        self.assertTrue(described.valid)
        self.assertEqual(described.type_name, "py.action")
        self.assertEqual(described.identity, "left")
        self.assertEqual(described.to_string(), "the left branch")
        self.assertEqual(described.to_tensor().values, [1.0, 2.0])
        self.assertEqual(described.to_tensor().shape, [2])

        # Presentation is not identity: the same type and identity is the same solver key.
        self.assertEqual(described, plain)
        self.assertEqual(hash(described), hash(plain))
        self.assertNotEqual(described, other)

        self.assertIsNone(plain.to_string())
        self.assertIsNone(plain.to_tensor())
        self.assertFalse(plain.capabilities.to_string)
        self.assertFalse(plain.capabilities.to_tensor)

    def test_empty_provider_values_are_invalid(self):
        self.assertFalse(nor.Action("py.action", "").valid)
        self.assertFalse(nor.Action("", "left").valid)
        self.assertFalse(bool(nor.Action("", "")))
        self.assertTrue(bool(nor.Action("py.action", "left")))

    def test_provider_state_histories_accumulate_observations(self):
        info_state = nor.InfoState(nor.Player.alex)
        self.assertEqual(len(info_state), 0)
        self.assertIs(info_state.player, nor.Player.alex)
        info_state.update(
            nor.Observation("py.obs", "public"), nor.Observation("py.obs", "private")
        )
        self.assertEqual(len(info_state), 1)
        public, private = info_state[0]
        self.assertEqual(public.identity, "public")
        self.assertEqual(private.identity, "private")
        self.assertEqual(info_state.latest()[1].identity, "private")

        public_state = nor.PublicState()
        public_state.update(nor.Observation("py.obs", "deal"))
        self.assertEqual(len(public_state), 1)
        self.assertEqual(public_state.latest().identity, "deal")

    def test_world_states_are_replaceable_values(self):
        state = nor.WorldState("py.world", "root")
        self.assertEqual(state.identity, "root")
        state.set(nor.WorldState("py.world", "next"))
        self.assertEqual(state.identity, "next")


if __name__ == "__main__":
    unittest.main()
