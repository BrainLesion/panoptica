# Call 'python -m unittest' on this folder
# coverage run -m unittest
# coverage report
# coverage html
import os
import unittest
import numpy as np
from panoptica.utils.numpy_utils import (
    _unique_without_zeros,
    _count_unique_without_zeros,
    _get_smallest_fitting_uint,
)
from panoptica.utils.citation_reminder import (
    citation_reminder,
    disable_citation_reminder,
    enable_citation_reminder,
)


class Test_Citation_Reminder(unittest.TestCase):
    def setUp(self) -> None:
        os.environ["PANOPTICA_CITATION_REMINDER"] = "True"
        return super().setUp()

    def test_citation_code(self):
        @citation_reminder
        def foo():
            return "bar"

        foo()

    def test_disable_citation_reminder(self):
        os.environ["PANOPTICA_CITATION_REMINDER"] = "True"
        disable_citation_reminder()
        try:

            @citation_reminder
            def foo():
                return "bar"

            self.assertEqual(foo(), "bar")
            # Disabled -> the reminder block (and its "show once" env latch) is skipped,
            # so the env var is left untouched.
            self.assertEqual(os.environ["PANOPTICA_CITATION_REMINDER"], "True")
        finally:
            enable_citation_reminder()  # restore module state for other tests


class Test_Numpy_Utils(unittest.TestCase):
    def setUp(self) -> None:
        os.environ["PANOPTICA_CITATION_REMINDER"] = "False"
        return super().setUp()

    def test_np_unique(self):
        a = np.array([0, 1, 2, 3, 6])
        b = _unique_without_zeros(a)

        self.assertTrue(b[0] == 1)
        self.assertTrue(b[1] == 2)
        self.assertTrue(b[2] == 3)
        self.assertTrue(b[3] == 6)
        self.assertEqual(b.shape[0], 4)

        with self.assertWarns(UserWarning):
            a = np.array([0, 1, -2, 3, -6])
            b = _unique_without_zeros(a)

    def test_np_count_unique(self):
        a = np.array([0, 1, 2, 3, 6])
        b = _count_unique_without_zeros(a)
        self.assertEqual(b, 4)
        #
        with self.assertWarns(UserWarning):
            a = np.array([0, 1, -2, 3, -6])
            b = _count_unique_without_zeros(a)
