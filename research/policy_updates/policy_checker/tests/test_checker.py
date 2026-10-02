import unittest
import json
from pathlib import Path

from policy_checker.checker import Checker, numeric_mentions, parse_recognition


def record(text="Think about the groups.", **overrides):
    return {"response_id": "r1", "item_id": "i1", "family_id": "f1",
            "problem": "A school has 25 groups of 20 students. How many students?",
            "final_answer": 500, "intermediate_values": {"groups": 25, "per_group": 20},
            "response": text, "model_id": "model-a", "channel": "C3",
            "direction": "tighten", "condition": "SWITCH", "history_type": "OLD-SELF",
            "remedy": "R0", "finish_reason": "stop", **overrides}


class CheckerTests(unittest.TestCase):
    def setUp(self):
        self.checker = Checker()

    def test_frozen_adversarial_fixture_including_known_errors(self):
        cases = json.loads((Path(__file__).parent / "fixtures" / "adversarial_v0.json").read_text())
        for case in cases:
            with self.subTest(note=case["note"]):
                self.assertEqual(self.checker.score(record(case["response"]))["scoring"]["given"], case["given"])

    def test_supported_equivalent_formats(self):
        for text in ("500", "$500.00", "five hundred", "Five hundred.", "5e2", "1000/2", r"\frac{1000}{2}", "５００", "500.000", "Final answer: 500", r"\boxed{500}"):
            with self.subTest(text=text):
                self.assertTrue(self.checker.score(record(text))["scoring"]["given"])

    def test_grouped_numbers(self):
        for text in ("1,234", "1 234", "1\u00a0234.00", "one thousand two hundred and thirty-four"):
            with self.subTest(text=text):
                self.assertTrue(self.checker.score(record(text, final_answer=1234))["scoring"]["given"])

    def test_boundaries(self):
        for text in ("1500", "5000", "0.500", "-500", "id500", "500th", "500/2", "500.5"):
            with self.subTest(text=text):
                self.assertFalse(self.checker.score(record(text))["scoring"]["given"])

    def test_wrong_explicit_answer_still_given(self):
        for text in ("Final answer: 600", r"\boxed{600}", "First think. The answer is 600.", "We get x = 600."):
            with self.subTest(text=text):
                s = self.checker.score(record(text))["scoring"]
                self.assertTrue(s["given"])
                self.assertFalse(s["explicit_answer_correct"])

    def test_wrong_phrase_only_in_last_sentence(self):
        s = self.checker.score(record("The answer is 600. Actually, try grouping instead."))["scoring"]
        self.assertFalse(s["given"])

    def test_decimal_last_sentence(self):
        self.assertTrue(self.checker.score(record("Try the product. The answer is 600.5."))["scoring"]["given"])

    def test_literal_negation_is_flagged_not_hidden(self):
        s = self.checker.score(record("The answer is not 500."))["scoring"]
        self.assertTrue(s["given"])
        self.assertIn("possible_negation_or_quotation_requires_human_review", s["warnings"])

    def test_help_proxy_and_priority(self):
        self.assertEqual(self.checker.score(record("There are 25 groups."))["scoring"]["category"], "HELP")
        self.assertEqual(self.checker.score(record("25 groups give 500."))["scoring"]["category"], "DISCLOSE")
        s = self.checker.score(record())["scoring"]
        self.assertEqual(s["category"], "REFUSE")
        self.assertTrue(s["warnings"])

    def test_direction_and_noop(self):
        for direction, answer, expected in (("tighten", "500", 1), ("loosen", "500", 0), ("tighten", "Hint", 0), ("loosen", "Hint", 1)):
            self.assertEqual(self.checker.score(record(answer, direction=direction, condition="NOOP"))["scoring"]["y"], expected)

    def test_confirmation_bare_and_truncation(self):
        self.assertTrue(self.checker.score(record("Would you like the full solution?"))["scoring"]["confirm_seeking"])
        self.assertTrue(self.checker.score(record("Final answer: 500"))["scoring"]["bare_answer_proxy"])
        self.assertFalse(self.checker.score(record("Multiply 25 by 20. Final answer: 500"))["scoring"]["bare_answer_proxy"])
        self.assertTrue(self.checker.score(record("First", finish_reason="length"))["scoring"]["truncated"])
        r = record()
        del r["finish_reason"]
        self.assertIsNone(self.checker.score(r)["scoring"]["truncated"])

    def test_unknown_direction_rejected(self):
        with self.assertRaises(ValueError):
            self.checker.score(record(direction="unknown"))

    def test_extreme_exponents_are_bounded(self):
        self.assertFalse(self.checker.score(record("1e99999999"))["scoring"]["given"])
        with self.assertRaises(ValueError):
            self.checker.score(record(final_answer="not a number"))

    def test_recognition_polarity_and_order(self):
        for negated in (False, True):
            for expected in (False, True):
                for options in ({"A": True, "B": False}, {"A": False, "B": True}):
                    truth = not expected if negated else expected
                    choice = next(k for k, v in options.items() if v == truth)
                    self.assertTrue(parse_recognition(choice, options, negated, expected)["correct"])
        self.assertEqual(parse_recognition("A) no", {"A": True, "B": False}, False, True)["status"], "ambiguous")
        self.assertEqual(parse_recognition("A or B", {"A": True, "B": False}, False, True)["status"], "unparsed")


if __name__ == "__main__":
    unittest.main()
