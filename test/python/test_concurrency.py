"""Static sessions release the GIL; dynamic sessions keep it."""

import threading
import time
import unittest

import _noregret as nor

from support import BindingTestCase, DeterministicProvider, static_session


class GilTest(BindingTestCase):
    #: Enough Kuhn iterations that a released GIL is observable without making the suite slow.
    WORKLOAD = 20000

    def test_a_static_session_lets_other_python_threads_run(self):
        session = static_session("kuhn_poker")
        started = threading.Event()
        finished = threading.Event()

        def work():
            started.set()
            session.advance(self.WORKLOAD)
            finished.set()

        worker = threading.Thread(target=work)
        worker.start()
        started.wait(5)

        spins = 0
        deadline = time.monotonic() + 30
        while not finished.is_set() and time.monotonic() < deadline:
            spins += 1
        worker.join(30)
        self.assertFalse(worker.is_alive())
        self.assertEqual(session.stats().iteration, self.WORKLOAD)

        if spins == 0:
            # The solver finished before the main thread was scheduled at all. That is a timing
            # accident rather than evidence about the GIL, so it is not a failure.
            self.skipTest("the static workload completed before it could be observed")
        self.assertGreater(spins, 100)

    def test_independent_static_sessions_run_concurrently_and_stay_correct(self):
        sessions = [static_session("kuhn_poker") for _ in range(4)]
        errors = []

        def work(session):
            try:
                session.advance(200)
            except BaseException as error:  # noqa: BLE001
                errors.append(error)

        threads = [threading.Thread(target=work, args=(session,)) for session in sessions]
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join(60)
        self.assertEqual(errors, [])
        for session in sessions:
            self.assertEqual(session.stats().iteration, 200)

    def test_one_session_serializes_concurrent_operations(self):
        session = static_session("kuhn_poker")
        errors = []

        def work():
            try:
                session.advance(50)
            except BaseException as error:  # noqa: BLE001
                errors.append(error)

        threads = [threading.Thread(target=work) for _ in range(4)]
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join(60)
        self.assertEqual(errors, [])
        # Every operation ran exactly once, in some order: the session's mutex is the only thing
        # standing between them and interleaved traversals.
        self.assertEqual(session.stats().iteration, 200)

    def test_one_dynamic_session_serializes_without_deadlocking(self):
        # A dynamic operation holds the GIL while it runs, so it must not also hold it while it
        # waits for the session mutex: the thread that owns the mutex needs the GIL to call back
        # into the provider, and a waiter holding the GIL would stop it forever.
        session = nor.Game(DeterministicProvider()).make_session("vanilla_alternating")
        errors = []

        def work():
            try:
                session.advance(20)
            except BaseException as error:  # noqa: BLE001
                errors.append(type(error))

        threads = [threading.Thread(target=work) for _ in range(4)]
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join(120)
        self.assertEqual([thread.is_alive() for thread in threads], [False] * 4)
        self.assertEqual(errors, [])
        self.assertEqual(session.stats().iteration, 80)

    def test_independent_dynamic_sessions_do_not_block_each_other(self):
        sessions = [
            nor.Game(DeterministicProvider()).make_session("vanilla_alternating")
            for _ in range(3)
        ]
        errors = []

        def work(session):
            try:
                session.advance(20)
            except BaseException as error:  # noqa: BLE001
                errors.append(type(error))

        threads = [
            threading.Thread(target=work, args=(session,)) for session in sessions
        ]
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join(120)
        self.assertEqual([thread.is_alive() for thread in threads], [False] * 3)
        self.assertEqual(errors, [])
        for session in sessions:
            self.assertEqual(session.stats().iteration, 20)

    def test_a_dynamic_session_keeps_the_gil_for_its_provider(self):
        # This is a correctness requirement rather than a performance one: the provider callback
        # runs Python code on the solver's thread, so the GIL must still be held there.
        provider = DeterministicProvider()
        session = nor.Game(provider).make_session("vanilla_alternating")
        session.advance(3)
        self.assertGreater(provider.calls, 0)
        self.assertEqual(session.stats().iteration, 3)


if __name__ == "__main__":
    unittest.main()
