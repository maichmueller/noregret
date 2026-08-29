"""Policy wrappers: focused lookup, explicit copying, staleness and synchronization."""

import gc
import unittest

import _noregret as nor

from support import BindingTestCase, static_session


class PolicyTest(BindingTestCase):
    def make_policy(self, kind=nor.PolicyKind.current):
        session = static_session("kuhn_poker")
        session.advance(3)
        return session, session.policy(kind)

    def test_lookup_is_focused_and_matches_the_bulk_copy(self):
        session, policy = self.make_policy()
        rows = policy.to_entries()
        self.assertTrue(rows)
        for player, info_state, actions in rows:
            row = policy.at(info_state)
            self.assertEqual(row.player, player)
            self.assertEqual(row.size, len(actions))
            self.assertEqual(len(row), len(actions))
            for index, (action, probability) in enumerate(actions):
                self.assertEqual(row.action_at(index), action)
                self.assertAlmostEqual(row.value_at(index), probability)
                self.assertAlmostEqual(row.at(action), probability)
                self.assertAlmostEqual(row.find(action), probability)
                self.assertIn(action, row)

    def test_bulk_row_order_is_unspecified_but_the_row_set_is_not(self):
        session, policy = self.make_policy()
        first = policy.to_entries()
        second = policy.to_entries()
        self.assertEqual(len(first), len(second))
        as_set = {info_state for _, info_state, _ in first}
        self.assertEqual(as_set, {info_state for _, info_state, _ in second})

    def test_missing_lookups_report_absence_rather_than_guessing(self):
        session, policy = self.make_policy()
        foreign = nor.InfoState(nor.Player.alex)
        self.assertIsNone(policy.find(foreign))
        with self.assertRaises(KeyError):
            policy.at(foreign)

        row = policy.at(policy.to_entries()[0][1])
        unknown_action = nor.Action("test.action", "not_in_this_game")
        self.assertIsNone(row.find(unknown_action))
        self.assertFalse(row.contains(unknown_action))
        with self.assertRaises(KeyError):
            row.at(unknown_action)

    def test_current_and_average_are_separate_views(self):
        session = static_session("kuhn_poker")
        session.advance(3)
        current = session.policy(nor.PolicyKind.current)
        average = session.policy(nor.PolicyKind.average)
        self.assertIs(current.kind, nor.PolicyKind.current)
        self.assertIs(average.kind, nor.PolicyKind.average)
        self.assertTrue(current.to_entries())
        self.assertTrue(average.to_entries())

    def test_row_tensor_is_the_probability_vector_only(self):
        session, policy = self.make_policy()
        row = policy.at(policy.to_entries()[0][1])
        tensor = row.to_tensor()
        self.assertEqual(tensor.shape, [row.size])
        self.assertEqual(len(tensor.values), row.size)
        for index, value in enumerate(tensor.values):
            self.assertAlmostEqual(value, row.value_at(index))

    def test_handles_go_stale_when_the_session_advances(self):
        session, policy = self.make_policy()
        info_state = policy.to_entries()[0][1]
        row = policy.at(info_state)
        self.assertTrue(policy.valid)
        self.assertTrue(row.valid)

        session.advance(1)
        self.assertFalse(policy.valid)
        self.assertFalse(row.valid)
        with self.assertRaises(nor.StaleViewError):
            policy.find(info_state)
        with self.assertRaises(nor.StaleViewError):
            row.size

        # A fresh handle works again.
        refreshed = session.policy()
        self.assertTrue(refreshed.valid)
        self.assertIsNotNone(refreshed.find(info_state))

    def test_handles_do_not_keep_a_destroyed_session_alive(self):
        session, policy = self.make_policy()
        info_state = policy.to_entries()[0][1]
        row = policy.at(info_state)

        del session
        gc.collect()

        self.assertFalse(policy.valid)
        self.assertFalse(row.valid)
        with self.assertRaises(nor.ClosedSessionError):
            policy.to_entries()
        with self.assertRaises(nor.ClosedSessionError):
            row.player

    def test_no_visitor_callback_is_exposed(self):
        # A Python callback running inside the solver's node traversal is exactly the hazard the
        # C++ side guards against; the binding must not offer one at all.
        session, policy = self.make_policy()
        for forbidden in ("visit", "for_each", "walk", "apply"):
            self.assertFalse(hasattr(policy, forbidden), forbidden)
        row = policy.at(policy.to_entries()[0][1])
        for forbidden in ("visit", "for_each"):
            self.assertFalse(hasattr(row, forbidden), forbidden)

    def test_copies_are_explicit(self):
        # Reading a row does not materialize a table; the only bulk copies are the two named
        # to_entries()/to_tensor() operations.
        session, policy = self.make_policy()
        row = policy.at(policy.to_entries()[0][1])
        entries = row.to_entries()
        self.assertIsInstance(entries, list)
        self.assertIsInstance(entries[0], tuple)
        self.assertIsInstance(entries[0][0], nor.Action)
        self.assertIsInstance(entries[0][1], float)


if __name__ == "__main__":
    unittest.main()
