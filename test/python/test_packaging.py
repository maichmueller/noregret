"""The shipped package layout and the absence of the retired pybind11 surface."""

import importlib
import os
import pathlib
import shutil
import subprocess
import tempfile
import unittest

REPOSITORY = pathlib.Path(__file__).resolve().parents[2]


class NoPybind11Test(unittest.TestCase):
    def test_no_pybind11_reference_remains_in_the_tree(self):
        # Scan the checked-in sources when git is available, and the working tree otherwise, so
        # this holds for a mirrored source directory as much as for a checkout.
        listing = subprocess.run(
            ["git", "-C", str(REPOSITORY), "ls-files"],
            capture_output=True,
            text=True,
            check=False,
        )
        if listing.returncode == 0:
            candidates = [REPOSITORY / name for name in listing.stdout.splitlines()]
        else:
            candidates = [
                path
                for path in REPOSITORY.rglob("*")
                if not any(
                    part in {".git", ".conan-home"}
                    or part.startswith("build")
                    or part == "__pycache__"
                    for part in path.relative_to(REPOSITORY).parts
                )
            ]

        offenders = []
        for path in candidates:
            if not path.is_file() or path.resolve() == pathlib.Path(__file__).resolve():
                continue
            try:
                text = path.read_text(encoding="utf-8", errors="ignore")
            except OSError:
                continue
            if "pybind11" in text.lower():
                offenders.append(str(path.relative_to(REPOSITORY)))
        self.assertEqual(offenders, [])

    def test_no_trampoline_headers_remain(self):
        self.assertFalse((REPOSITORY / "src/nor/binding/trampolines").exists())

    def test_the_extension_module_is_named_as_the_package_expects(self):
        init = (REPOSITORY / "src/nor/__init__.py").read_text()
        self.assertIn("_noregret", init)
        module = importlib.import_module("_noregret")
        self.assertEqual(module.__name__, "_noregret")


@unittest.skipUnless(
    os.environ.get("NOR_PACKAGE_ROOT"),
    "set NOR_PACKAGE_ROOT to the directory containing the installed nor package",
)
class InstalledPackageTest(unittest.TestCase):
    def test_import_nor(self):
        root = os.environ["NOR_PACKAGE_ROOT"]
        script = (
            "import nor;"
            "assert nor.Game is not None;"
            "s = nor.Game('rock_paper_scissors').make_session('vanilla_alternating');"
            "r = s.iterate();"
            "assert r.iteration == 0;"
            "print('import nor ok', len(nor.games()))"
        )
        environment = dict(os.environ)
        environment["PYTHONPATH"] = root + os.pathsep + environment.get("PYTHONPATH", "")
        completed = subprocess.run(
            [os.environ.get("PYTHON_EXECUTABLE", "python3"), "-c", script],
            capture_output=True,
            text=True,
            env=environment,
            check=False,
        )
        self.assertEqual(completed.returncode, 0, completed.stderr)
        self.assertIn("import nor ok", completed.stdout)


@unittest.skipUnless(
    os.environ.get("NOR_INSTALLED_PACKAGE_ROOT"),
    "set NOR_INSTALLED_PACKAGE_ROOT to the installed package prefix",
)
class RelocatedInstalledPackageTest(unittest.TestCase):
    def test_import_nor_after_copying_the_installed_package(self):
        installed_root = pathlib.Path(os.environ["NOR_INSTALLED_PACKAGE_ROOT"])
        installed_package = installed_root / "nor"
        self.assertTrue(installed_package.is_dir())

        with tempfile.TemporaryDirectory(prefix="noregret-relocated-") as temporary:
            relocated_root = pathlib.Path(temporary)
            shutil.copytree(installed_package, relocated_root / "nor")
            environment = dict(os.environ)
            # Deliberately remove the build-tree module path: this child process may resolve only
            # the copied install, proving that the package's relative runtime layout is complete.
            environment["PYTHONPATH"] = str(relocated_root)
            script = (
                "import nor;"
                "assert nor.Game is not None;"
                "s = nor.Game('rock_paper_scissors').make_session('vanilla_alternating');"
                "assert s.iterate().iteration == 0;"
                "print('relocated import nor ok')"
            )
            completed = subprocess.run(
                [os.environ.get("PYTHON_EXECUTABLE", "python3"), "-c", script],
                capture_output=True,
                text=True,
                env=environment,
                check=False,
            )
            self.assertEqual(completed.returncode, 0, completed.stderr)
            self.assertIn("relocated import nor ok", completed.stdout)


if __name__ == "__main__":
    unittest.main()
