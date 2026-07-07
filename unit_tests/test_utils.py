# Call 'python -m unittest' on this folder
# coverage run -m unittest
# coverage report
# coverage html
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
)


class Test_Citation_Reminder(unittest.TestCase):
    def test_citation_code(self):
        @citation_reminder
        def foo():
            return "bar"

        self.assertEqual(foo(), "bar")

    def test_disable_citation_reminder(self):
        disable_citation_reminder()

        @citation_reminder
        def foo():
            return "bar"

        self.assertEqual(foo(), "bar")


class Test_Set_Log_Level(unittest.TestCase):
    def setUp(self) -> None:
        disable_citation_reminder()
        return super().setUp()

    def test_set_log_level(self):
        import logging
        from panoptica import set_log_level
        from panoptica.utils.logger import logger

        original = logger.level
        try:
            set_log_level("WARNING")
            self.assertEqual(logger.level, logging.WARNING)
            set_log_level(logging.DEBUG)
            self.assertEqual(logger.level, logging.DEBUG)
            set_log_level("debug")  # names are case-insensitive
            self.assertEqual(logger.level, logging.DEBUG)
        finally:
            logger.setLevel(original)


class Test_Numpy_Utils(unittest.TestCase):
    def setUp(self) -> None:
        disable_citation_reminder()
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
