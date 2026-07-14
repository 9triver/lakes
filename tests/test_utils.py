from __future__ import annotations

import unittest
from datetime import date

from lake_workbench.utils import add_months, clean_optional, truthy_flag


class UtilityTests(unittest.TestCase):
    def test_add_months_clamps_end_of_month(self) -> None:
        self.assertEqual(add_months(date(2024, 3, 31), -1), date(2024, 2, 29))

    def test_clean_optional_normalizes_csv_numbers(self) -> None:
        self.assertIsNone(clean_optional(float("nan")))
        self.assertEqual(clean_optional("123.0"), "123")

    def test_truthy_flag_parses_csv_values(self) -> None:
        self.assertTrue(truthy_flag("yes"))
        self.assertFalse(truthy_flag("0", default=True))

if __name__ == "__main__":
    unittest.main()
