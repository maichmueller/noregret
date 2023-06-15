import os
import subprocess
import sys

import setuptools
from setuptools.command.build_ext import build_ext

import sys
import os
import subprocess
import shutil
import re
import select


# ---------------------- Configuration section ------------------------------
def find_executable(exe_name):
    executable = shutil.which(exe_name)
    if not os.path.isfile(executable):
        sys.exit(f'Could not find {exe_name}!')
    else:
        call_result = subprocess.run([executable, '--version'], capture_output=True, text=True)
        if call_result.returncode != 0:
            raise RuntimeError(call_result.stderr)
        print_res = str(call_result.stdout).replace('\n', ' ')
        print(f"{exe_name} found at: {executable}, version: {print_res}")
    return executable


CMAKE = find_executable("cmake")
NINJA = find_executable("ninja")
CONAN = find_executable("conan")

pymodule_cmake_target_name = "noregret"
compile_config = "Release"


# ---------------------- Building Extension section -------------------------

class CMakeExtension(setuptools.Extension):
    def __init__(self, name, sourcedir=""):
        super().__init__(name, sources=[])
        self.sourcedir = os.path.abspath(sourcedir)


def _patch_cached_executable_paths(build_folder):
    # hacky and fragile way to override executable filepaths in CMakeCache.txt.
    # This cache stores filepaths to executables of temp environments as created
    # by pip which are no longer valid upon a consecutive install run.
    # The encountered errors are cryptic 'No such file or directory' errors and
    # require deleting the CMakeCache.txt or, even better, deleting the entire
    # build folder altogether.
    cmake_cache_path = os.path.join(build_folder, "CMakeCache.txt")
    if os.path.exists(cmake_cache_path):
        pattern_generator = re.compile(r"(?<=CMAKE_MAKE_PROGRAM:FILEPATH=).*")
        pattern_conan = re.compile(r"(?<=CONAN:FILEPATH=).*")
        with open(cmake_cache_path, "r") as file:
            lines = file.read()
            lines = pattern_conan.sub(CONAN, lines)
            lines = pattern_generator.sub(NINJA, lines)

        with open(cmake_cache_path, "w") as file:
            file.write(lines)


class Build(build_ext):
    def run(self):
        for ext in self.extensions:
            self.build_extension(ext)

    def build_extension(self, ext):
        extension_dir = os.path.abspath(
            os.path.dirname(self.get_ext_fullpath(ext.name))
        )
        cwd = os.getcwd()
        source_folder = os.getcwd()
        build_folder = os.path.join(cwd, self.build_temp)
        env = os.environ.copy()

        _patch_cached_executable_paths(build_folder)

        if not os.path.exists(build_folder):
            os.makedirs(build_folder)

        # cmake configure
        subprocess.check_call(
            [
                CMAKE,
                "-S",
                source_folder,
                "-B",
                build_folder,
                "-DCMAKE_BUILD_TYPE=Release",
                "-DCMAKE_GENERATOR:INTERNAL=Ninja",
                "-DENABLE_BUILD_PYTHON_EXTENSION=ON",
                "-DENABLE_TESTING=OFF",
                "-DENABLE_CACHE=OFF",
                "-DINSTALL_PYMODULE=ON",
                "-DWARNINGS_AS_ERRORS=OFF",
            ],
            cwd=cwd,
            env=env,
        )
        # cmake build
        subprocess.check_call(
            [
                CMAKE,
                "--build",
                build_folder,
                "--config",
                compile_config,
                "--target",
                pymodule_cmake_target_name,
                "-j",
                str(os.cpu_count()),
            ],
            cwd=cwd,
            env=env,
        )
        # cmake install
        subprocess.check_call(
            [
                CMAKE,
                "--install",
                build_folder,
                "--prefix",
                extension_dir
            ],
            cwd=cwd,
            env=env,
        )


setuptools.setup(
    ext_modules=[CMakeExtension("noregret", sourcedir="src/nor")],
    cmdclass={"build_ext": Build},
)
