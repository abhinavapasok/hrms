# Copyright (c) 2026, Frappe Technologies Pvt. Ltd. and Contributors
# See license.txt

import frappe

from hrms.tests.utils import HRMSTestSuite
from hrms.utils.ocr import (
	HeuristicFieldExtractor,
	TesseractProvider,
	get_ocr_provider,
	match_expense_type,
)

SAMPLE_RECEIPT = """SuperMart Grocery
GSTIN: 29ABCDE1234F1Z5
123 Main Street

Date: 14/05/2026

Milk        ₹ 50.00
Bread       ₹ 40.00

Grand Total ₹ 90.00
"""


class TestOCRParser(HRMSTestSuite):
	def setUp(self):
		frappe.set_user("Administrator")
		self.settings = frappe.get_single("HR Settings")

	def tearDown(self):
		frappe.db.rollback()

	def test_parse_date_formats(self):
		extractor = HeuristicFieldExtractor()
		cases = {
			"Date: 2026-05-14": "2026-05-14",
			"Date: 14/05/2026": "2026-05-14",
			"Date: 14 May 2026": "2026-05-14",
			"Date: May 14, 2026": "2026-05-14",
		}
		for text, expected in cases.items():
			result = extractor.extract_fields(text)
			self.assertEqual(result.expense_date, expected, msg=f"failed for: {text}")

	def test_parse_amount(self):
		extractor = HeuristicFieldExtractor()
		result = extractor.extract_fields(SAMPLE_RECEIPT)
		self.assertEqual(result.amount, 90.0)

	def test_heuristic_does_not_guess_currency(self):
		# currency is ambiguous from symbols; the heuristic leaves it unset
		extractor = HeuristicFieldExtractor()
		self.assertIsNone(extractor.extract_fields("Total ₹ 90.00").currency)
		self.assertIsNone(extractor.extract_fields("Total $ 90.00").currency)

	def test_validate_currency(self):
		from hrms.utils.ocr import _validate_currency

		self.assertEqual(_validate_currency("usd"), "USD")  # normalized + exists
		self.assertEqual(_validate_currency("INR"), "INR")
		self.assertIsNone(_validate_currency("ZZZ"))  # not a real currency
		self.assertIsNone(_validate_currency(None))

	def test_parse_vendor(self):
		extractor = HeuristicFieldExtractor()
		result = extractor.extract_fields(SAMPLE_RECEIPT)
		self.assertEqual(result.vendor_name, "SuperMart Grocery")

	def test_amount_ignores_subtotal_and_cash(self):
		extractor = HeuristicFieldExtractor()
		text = "Subtotal 38.00\nGST 1.90\nCASH 50.00\nGrand Total 39.90\nChange 10.10"
		self.assertEqual(extractor._extract_amount(text.split("\n")), 39.90)

	def test_date_locale_day_first_vs_month_first(self):
		extractor = HeuristicFieldExtractor()
		self.assertEqual(extractor._extract_date("Date: 03/04/2026", day_first=True), "2026-04-03")
		self.assertEqual(extractor._extract_date("Date: 03/04/2026", day_first=False), "2026-03-04")

	def test_match_expense_type(self):
		claim_types = frappe.get_all("Expense Claim Type", pluck="name")
		if not claim_types:
			self.skipTest("No Expense Claim Type records available")

		# Exact name appearing in the text should match that type
		target = claim_types[0]
		self.assertEqual(match_expense_type(target, ""), target)

		# No match returns None (no arbitrary default)
		self.assertIsNone(match_expense_type("zzz-nonexistent-vendor", ""))
		self.assertIsNone(match_expense_type("", ""))

	def test_provider_factory(self):
		frappe.db.set_single_value("HR Settings", "ocr_provider", "Tesseract")
		provider = get_ocr_provider()
		self.assertIsInstance(provider, TesseractProvider)

	def test_provider_not_configured(self):
		frappe.db.set_single_value("HR Settings", "ocr_provider", "None")
		with self.assertRaises(frappe.ValidationError):
			get_ocr_provider()

	def test_unknown_provider(self):
		frappe.db.set_single_value("HR Settings", "ocr_provider", "Bogus")
		with self.assertRaises(frappe.ValidationError):
			get_ocr_provider()

	# ── LLM provider tests (LiteLLM) ────────────────────────

	def test_build_llm_model_string(self):
		from hrms.utils.ocr import build_llm_model_string

		# default model per provider
		frappe.db.set_single_value("HR Settings", "llm_provider", "Google Gemini")
		frappe.db.set_single_value("HR Settings", "llm_model", "")
		self.assertEqual(build_llm_model_string(), "gemini/gemini-2.5-flash")

		frappe.db.set_single_value("HR Settings", "llm_provider", "Anthropic")
		self.assertEqual(build_llm_model_string(), "anthropic/claude-haiku-4-5")

		frappe.db.set_single_value("HR Settings", "llm_provider", "OpenAI")
		self.assertEqual(build_llm_model_string(), "openai/gpt-4o-mini")

		# explicit model overrides the default
		frappe.db.set_single_value("HR Settings", "llm_model", "gpt-4o")
		self.assertEqual(build_llm_model_string(), "openai/gpt-4o")

	def test_build_llm_model_string_unknown_provider(self):
		from hrms.utils.ocr import build_llm_model_string

		frappe.db.set_single_value("HR Settings", "llm_provider", "Bogus")
		with self.assertRaises(frappe.ValidationError):
			build_llm_model_string()

	def test_compatible_requires_base_url(self):
		from hrms.utils.ocr import build_llm_model_string

		frappe.db.set_single_value("HR Settings", "llm_provider", "OpenAI-Compatible")
		frappe.db.set_single_value("HR Settings", "llm_model", "llama3")
		frappe.db.set_single_value("HR Settings", "llm_base_url", "")
		with self.assertRaises(frappe.ValidationError):
			build_llm_model_string()

		frappe.db.set_single_value("HR Settings", "llm_base_url", "http://localhost:11434/v1")
		self.assertEqual(build_llm_model_string(), "openai/llama3")

	def test_extract_json(self):
		from hrms.utils.ocr import _extract_json

		self.assertEqual(_extract_json('```json\n{"a": 1}\n```'), '{"a": 1}')
		self.assertEqual(_extract_json('prefix {"a": 1} suffix'), '{"a": 1}')
		self.assertEqual(_extract_json('{"a": 1}'), '{"a": 1}')

	def test_llm_extractor_with_mocked_litellm(self):
		"""End-to-end Stage 2 through LiteLLM, with the API call mocked (no network)."""
		try:
			import litellm  # noqa: F401
		except ImportError:
			self.skipTest("litellm is not installed")

		from unittest.mock import MagicMock, patch

		from hrms.utils.ocr import LLMFieldExtractor

		frappe.db.set_single_value("HR Settings", "ocr_provider", "Tesseract")
		frappe.db.set_single_value("HR Settings", "ocr_extraction_method", "LLM")
		frappe.db.set_single_value("HR Settings", "llm_provider", "Google Gemini")
		frappe.db.set_single_value("HR Settings", "llm_model", "")
		settings = frappe.get_single("HR Settings")
		settings.llm_api_key = "dummy-key"
		settings.save()

		payload = (
			'{"amount": 39.6, "currency": "GBP", "vendor_name": "The Copper Spoon", '
			'"expense_date": "2026-05-18", "description": "Dinner", "category": null, '
			'"confidence": 0.95}'
		)
		fake = MagicMock()
		fake.choices = [MagicMock(message=MagicMock(content=payload))]

		with patch("litellm.completion", return_value=fake) as mock_completion:
			result = LLMFieldExtractor().extract_fields("raw receipt text")
			# provider+model resolved into the LiteLLM model string
			self.assertEqual(mock_completion.call_args.kwargs["model"], "gemini/gemini-2.5-flash")

		self.assertEqual(result.amount, 39.6)
		self.assertEqual(result.currency, "GBP")
		self.assertEqual(result.vendor_name, "The Copper Spoon")
		self.assertEqual(result.confidence_score, 0.95)
