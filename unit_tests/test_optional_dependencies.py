# Call 'python -m unittest' on this folder
# coverage run -m unittest
# coverage report
# coverage html
"""Tests that panoptica works in an environment without its optional dependencies.

A plain `pip install panoptica` installs none of nibabel / SimpleITK / pynrrd / torch,
so importing any panoptica module must not touch them. The environment is simulated
by wrapping every finder on `sys.meta_path` so that the optional packages look like
they were never installed, and by re-importing panoptica from scratch inside that
simulated environment.
"""

import contextlib
import importlib
import subprocess
import sys
import textwrap
import unittest
from importlib.util import find_spec
from pathlib import Path

import numpy as np

from panoptica.utils.input_check_and_conversion.input_data_type_checker import (
    _MissingOptionalPackage,
)

OPTIONAL_PACKAGES = ("nibabel", "SimpleITK", "nrrd", "torch")

# Modules that must stay importable without any optional package installed.
IMPORTABLE_WITHOUT_OPTIONALS = (
    "panoptica",
    "panoptica.utils",
    "panoptica.utils.input_check_and_conversion.sanity_checker",
    "panoptica.utils.input_check_and_conversion.check_nibabel_image",
    "panoptica.utils.input_check_and_conversion.check_sitk_image",
    "panoptica.utils.input_check_and_conversion.check_nrrd_image",
    "panoptica.utils.input_check_and_conversion.check_torch_image",
    "panoptica.panoptica_evaluator",
    "panoptica.cli",
)


class _HidingFinder:
    """Wraps a real meta path finder and hides the given top level packages.

    Returning None for a hidden name is exactly what the import system sees for a
    package that is not installed: `find_spec()` yields None and `import` raises
    ModuleNotFoundError.
    """

    def __init__(self, inner, hidden_packages: frozenset):
        self._inner = inner
        self._hidden_packages = hidden_packages

    def find_spec(self, fullname, path=None, target=None):
        if fullname.split(".")[0] in self._hidden_packages:
            return None
        inner_find_spec = getattr(self._inner, "find_spec", None)
        if inner_find_spec is None:
            return None
        return inner_find_spec(fullname, path, target)

    def invalidate_caches(self):
        inner_invalidate = getattr(self._inner, "invalidate_caches", None)
        if inner_invalidate is not None:
            inner_invalidate()


@contextlib.contextmanager
def hide_packages(*package_names: str):
    """Pretend the given packages are not installed and re-import panoptica.

    Inside the context, panoptica (and the hidden packages) are purged from
    `sys.modules`, so any import performed there goes through the module code again
    under the simulated environment. The previous module set is restored on exit.
    """
    hidden = frozenset(package_names)
    saved_meta_path = list(sys.meta_path)
    saved_modules = dict(sys.modules)

    def _is_affected(module_name: str) -> bool:
        root = module_name.split(".")[0]
        return root in hidden or root == "panoptica"

    sys.meta_path = [_HidingFinder(finder, hidden) for finder in sys.meta_path]
    for module_name in list(sys.modules):
        if _is_affected(module_name):
            del sys.modules[module_name]
    importlib.invalidate_caches()
    try:
        yield
    finally:
        sys.meta_path = saved_meta_path
        for module_name in list(sys.modules):
            if _is_affected(module_name):
                del sys.modules[module_name]
        sys.modules.update(
            {
                module_name: module
                for module_name, module in saved_modules.items()
                if _is_affected(module_name)
            }
        )
        importlib.invalidate_caches()


class Test_Missing_Optional_Package_Shim(unittest.TestCase):
    def test_attribute_access_raises_import_error(self):
        missing = _MissingOptionalPackage("nibabel")
        with self.assertRaises(ImportError) as context:
            _ = missing.Nifti1Image
        self.assertIn("nibabel", str(context.exception))

    def test_error_mentions_install_name(self):
        missing = _MissingOptionalPackage("nrrd", install_name="pynrrd")
        with self.assertRaises(ImportError) as context:
            _ = missing.read
        self.assertIn("pip install pynrrd", str(context.exception))

    def test_is_falsy(self):
        self.assertFalse(_MissingOptionalPackage("torch"))


class Test_Hide_Packages_Helper(unittest.TestCase):
    """The simulation itself must be faithful, otherwise everything below is vacuous."""

    def test_hidden_packages_are_not_findable(self):
        with hide_packages(*OPTIONAL_PACKAGES):
            for package_name in OPTIONAL_PACKAGES:
                self.assertIsNone(find_spec(package_name), package_name)
                with self.assertRaises(ModuleNotFoundError):
                    importlib.import_module(package_name)

    def test_other_packages_still_importable(self):
        with hide_packages(*OPTIONAL_PACKAGES):
            self.assertIsNotNone(find_spec("numpy"))
            self.assertIsNotNone(importlib.import_module("scipy"))

    def test_environment_is_restored(self):
        with hide_packages(*OPTIONAL_PACKAGES):
            pass
        for package_name in OPTIONAL_PACKAGES:
            if package_name in ("torch", "nrrd") and find_spec(package_name) is None:
                continue  # genuinely not installed in this environment
            self.assertIsNotNone(find_spec(package_name), package_name)
        self.assertIn("panoptica", sys.modules)


class Test_Import_Without_Optional_Packages(unittest.TestCase):
    def test_all_modules_importable_without_any_optional_package(self):
        with hide_packages(*OPTIONAL_PACKAGES):
            for module_name in IMPORTABLE_WITHOUT_OPTIONALS:
                with self.subTest(module=module_name):
                    importlib.import_module(module_name)

    def test_all_modules_importable_with_each_package_missing_individually(self):
        for package_name in OPTIONAL_PACKAGES:
            with hide_packages(package_name):
                for module_name in IMPORTABLE_WITHOUT_OPTIONALS:
                    with self.subTest(missing=package_name, module=module_name):
                        importlib.import_module(module_name)

    def test_import_in_fresh_interpreter(self):
        """Same check in a subprocess, so no already imported module can mask it."""
        script = textwrap.dedent(
            f"""
            import sys

            hidden = {OPTIONAL_PACKAGES!r}

            class Finder:
                def __init__(self, inner):
                    self.inner = inner

                def find_spec(self, fullname, path=None, target=None):
                    if fullname.split(".")[0] in hidden:
                        return None
                    find_spec = getattr(self.inner, "find_spec", None)
                    return None if find_spec is None else find_spec(fullname, path, target)

            sys.meta_path = [Finder(f) for f in sys.meta_path]

            import panoptica
            import panoptica.utils
            from panoptica.utils.input_check_and_conversion.sanity_checker import (
                INPUTDTYPE,
            )

            assert not any(
                p in sys.modules for p in hidden
            ), "an optional package got imported anyway"
            print("ok")
            """
        )
        result = subprocess.run(
            [sys.executable, "-c", script],
            capture_output=True,
            text=True,
            cwd=str(Path(__file__).resolve().parent.parent),
        )
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("ok", result.stdout)


class Test_Behaviour_Without_Optional_Packages(unittest.TestCase):
    def test_checkers_report_missing_packages(self):
        with hide_packages(*OPTIONAL_PACKAGES):
            sanity_checker = importlib.import_module(
                "panoptica.utils.input_check_and_conversion.sanity_checker"
            )
            inputdtype = sanity_checker.INPUTDTYPE

            self.assertTrue(inputdtype.NUMPY.value.are_requirements_fulfilled())
            self.assertEqual(inputdtype.NUMPY.value.missing_packages, [])

            expected_missing = [
                (inputdtype.SITK, ["SimpleITK"]),
                (inputdtype.NIBABEL, ["nibabel"]),
                (inputdtype.TORCH, ["torch"]),
                (inputdtype.NRRD, ["nrrd"]),
            ]
            for dtype, missing in expected_missing:
                with self.subTest(dtype=dtype.name):
                    self.assertFalse(dtype.value.are_requirements_fulfilled())
                    self.assertEqual(dtype.value.missing_packages, missing)

            # must not raise, it only logs what is available
            sanity_checker.print_available_package_to_input_handlers()

    def test_numpy_input_still_works(self):
        with hide_packages(*OPTIONAL_PACKAGES):
            sanity_checker = importlib.import_module(
                "panoptica.utils.input_check_and_conversion.sanity_checker"
            )
            prediction = np.zeros((10, 10), dtype=np.uint8)
            prediction[2:5, 2:5] = 1
            reference = np.zeros((10, 10), dtype=np.uint8)
            reference[2:5, 2:5] = 1

            (arrays, metadata), inputdtype = (
                sanity_checker.sanity_check_and_convert_to_array(prediction, reference)
            )
            self.assertEqual(inputdtype, sanity_checker.INPUTDTYPE.NUMPY)
            self.assertTrue(np.array_equal(arrays[0], prediction))
            self.assertTrue(np.array_equal(arrays[1], reference))
            self.assertEqual(metadata, {})

    def test_full_evaluation_with_numpy_input_still_works(self):
        with hide_packages(*OPTIONAL_PACKAGES):
            panoptica = importlib.import_module("panoptica")
            importlib.import_module(
                "panoptica.utils.citation_reminder"
            ).disable_citation_reminder()

            prediction = np.zeros((20, 20), dtype=np.uint8)
            prediction[2:8, 2:8] = 1
            reference = np.zeros((20, 20), dtype=np.uint8)
            reference[2:8, 2:8] = 1

            evaluator = panoptica.Panoptica_Evaluator(
                expected_input=panoptica.InputType.MATCHED_INSTANCE,
            )
            result = evaluator.evaluate(prediction, reference)["ungrouped"]
            self.assertEqual(result.tp, 1)
            self.assertEqual(result.fp, 0)
            self.assertEqual(result.fn, 0)

    def test_calling_unavailable_checker_raises_import_error(self):
        with hide_packages(*OPTIONAL_PACKAGES):
            sanity_checker = importlib.import_module(
                "panoptica.utils.input_check_and_conversion.sanity_checker"
            )
            checker = sanity_checker.INPUTDTYPE.NIBABEL.value
            with self.assertRaises(ImportError) as context:
                checker("prediction.nii.gz", "reference.nii.gz")
            self.assertIn("nibabel", str(context.exception))

    def test_touching_missing_package_raises_import_error(self):
        """The module attribute is a shim, so use gives ImportError, not AttributeError."""
        with hide_packages(*OPTIONAL_PACKAGES):
            check_nibabel_image = importlib.import_module(
                "panoptica.utils.input_check_and_conversion.check_nibabel_image"
            )
            with self.assertRaises(ImportError):
                _ = check_nibabel_image.nib.Nifti1Image

            check_sitk_image = importlib.import_module(
                "panoptica.utils.input_check_and_conversion.check_sitk_image"
            )
            with self.assertRaises(ImportError):
                _ = check_sitk_image.sitk.Image

            check_nrrd_image = importlib.import_module(
                "panoptica.utils.input_check_and_conversion.check_nrrd_image"
            )
            with self.assertRaises(ImportError):
                check_nrrd_image.nrrd.read("some.nrrd")

            check_torch_image = importlib.import_module(
                "panoptica.utils.input_check_and_conversion.check_torch_image"
            )
            with self.assertRaises(ImportError):
                _ = check_torch_image.torch.Tensor

    def test_unsupported_file_ending_reports_missing_packages(self):
        with hide_packages(*OPTIONAL_PACKAGES):
            sanity_checker = importlib.import_module(
                "panoptica.utils.input_check_and_conversion.sanity_checker"
            )
            with self.assertRaises(ImportError) as context:
                sanity_checker.sanity_check_and_convert_to_array(
                    "prediction.nii.gz", "reference.nii.gz"
                )
            message = str(context.exception)
            self.assertIn("SimpleITK", message)
            self.assertIn("nibabel", message)


if __name__ == "__main__":
    unittest.main()
