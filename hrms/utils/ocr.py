# Copyright (c) 2026, Frappe Technologies Pvt. Ltd. and Contributors
# License: GNU General Public License v3. See license.txt

import frappe
from frappe import _


class OCRResult:
	"""Structured result from OCR processing."""

	def __init__(self):
		self.expense_date = None
		self.amount = None
		self.currency = None
		self.vendor_name = None
		self.description = None
		self.expense_type_suggestion = None
		self.raw_text = ""
		self.confidence_score = 0.0

	def as_dict(self):
		return {
			"expense_date": self.expense_date,
			"amount": self.amount,
			"currency": self.currency,
			"vendor_name": self.vendor_name,
			"description": self.description,
			"expense_type_suggestion": self.expense_type_suggestion,
			"raw_text": self.raw_text,
			"confidence_score": self.confidence_score,
		}


# ──────────────────────────────────────────────────────
# STAGE 1: Text Extraction (OCR providers)
# ──────────────────────────────────────────────────────


class BaseOCRProvider:
	"""Abstract base. Subclasses handle text extraction only."""

	def extract(self, file_path: str) -> OCRResult:
		"""
		Default: extract raw text, then delegate to field extractor.
		Cloud providers override this to return structured data directly.
		"""
		raw_text = self.extract_text(file_path)
		extractor = get_field_extractor()
		result = extractor.extract_fields(raw_text)
		result.raw_text = raw_text
		return result

	def extract_text(self, file_path: str) -> str:
		raise NotImplementedError


class TesseractProvider(BaseOCRProvider):
	"""Free, self-hosted. Returns raw text → needs Stage 2."""

	def extract_text(self, file_path: str) -> str:
		try:
			import pytesseract
			from PIL import Image
		except ImportError:
			frappe.throw(_("pytesseract is not installed. Run: pip install pytesseract"))

		image = Image.open(file_path)
		return pytesseract.image_to_string(image)


class GoogleVisionProvider(BaseOCRProvider):
	"""Uses Document AI prebuilt-receipt model — returns structured data directly."""

	def extract(self, file_path: str) -> OCRResult:
		# Skips Stage 2 entirely — Google's receipt model returns structured fields
		try:
			from google.cloud import documentai_v1 as documentai
		except ImportError:
			frappe.throw(_("google-cloud-documentai is not installed"))

		processor_name = frappe.db.get_single_value("HR Settings", "ocr_api_endpoint")
		if not processor_name:
			frappe.throw(
				_("Set the Document AI processor resource name in the Custom OCR API Endpoint field")
			)

		with open(file_path, "rb") as f:
			content = f.read()

		client = documentai.DocumentProcessorServiceClient()
		mime_type = _guess_mime_type(file_path)
		raw_document = documentai.RawDocument(content=content, mime_type=mime_type)
		request = documentai.ProcessRequest(name=processor_name, raw_document=raw_document)
		response = client.process_document(request=request)
		document = response.document

		result = OCRResult()
		result.raw_text = document.text or ""

		# Map Document AI receipt entities → OCRResult
		field_map = {
			"supplier_name": "vendor_name",
			"receipt_date": "expense_date",
			"total_amount": "amount",
			"currency": "currency",
		}
		confidences = []
		for entity in document.entities:
			confidences.append(entity.confidence)
			target = field_map.get(entity.type_)
			if not target:
				continue
			value = entity.mention_text
			if target == "amount":
				result.amount = _to_float(value)
			elif target == "expense_date":
				result.expense_date = _normalize_date(value)
			elif target == "currency":
				result.currency = _validate_currency(value)
			else:
				setattr(result, target, value)

		if confidences:
			result.confidence_score = sum(confidences) / len(confidences)
		else:
			result.confidence_score = 0.9
		return result


class AzureDocIntelligenceProvider(BaseOCRProvider):
	"""Uses prebuilt-receipt model — returns structured data directly."""

	def extract(self, file_path: str) -> OCRResult:
		# Skips Stage 2 — Azure's receipt model returns structured fields
		try:
			from azure.ai.formrecognizer import DocumentAnalysisClient
			from azure.core.credentials import AzureKeyCredential
		except ImportError:
			frappe.throw(_("azure-ai-formrecognizer is not installed"))

		from frappe.utils.password import get_decrypted_password

		api_key = get_decrypted_password("HR Settings", "HR Settings", "ocr_api_key")
		endpoint = frappe.db.get_single_value("HR Settings", "ocr_api_endpoint")
		if not (api_key and endpoint):
			frappe.throw(_("Azure requires both an OCR API Key and a Custom OCR API Endpoint"))

		client = DocumentAnalysisClient(endpoint=endpoint, credential=AzureKeyCredential(api_key))
		with open(file_path, "rb") as f:
			poller = client.begin_analyze_document("prebuilt-receipt", document=f)
		receipts = poller.result()

		result = OCRResult()
		result.raw_text = receipts.content or ""
		if not receipts.documents:
			return result

		fields = receipts.documents[0].fields
		confidences = [receipts.documents[0].confidence] if receipts.documents[0].confidence else []

		merchant = fields.get("MerchantName")
		if merchant:
			result.vendor_name = merchant.value
			confidences.append(merchant.confidence)

		txn_date = fields.get("TransactionDate")
		if txn_date and txn_date.value:
			result.expense_date = str(txn_date.value)
			confidences.append(txn_date.confidence)

		total = fields.get("Total")
		if total and total.value is not None:
			amount = getattr(total.value, "amount", total.value)
			result.amount = _to_float(amount)
			result.currency = _validate_currency(getattr(total.value, "currency_code", None)) or result.currency
			confidences.append(total.confidence)

		if confidences:
			result.confidence_score = sum(c for c in confidences if c) / len(confidences)
		return result


# ──────────────────────────────────────────────────────
# STAGE 2: Field Extraction (from raw text)
# Only used when Stage 1 returns raw text (Tesseract)
# ──────────────────────────────────────────────────────


class BaseFieldExtractor:
	"""Abstract base for field extraction from raw OCR text."""

	def extract_fields(self, raw_text: str) -> OCRResult:
		raise NotImplementedError


class LLMFieldExtractor(BaseFieldExtractor):
	"""
	Recommended. Sends raw text to an LLM with a structured prompt.
	Handles all the ambiguity ("Grand Total" vs "Amt Payable" vs "You Pay")
	because the LLM was trained on millions of receipts.
	"""

	def extract_fields(self, raw_text: str) -> OCRResult:
		expense_types = frappe.get_all("Expense Claim Type", pluck="name")

		prompt = f"""Extract expense data from this receipt text.
Return ONLY valid JSON with these exact keys:
- expense_date: date in YYYY-MM-DD format
- amount: the final total amount the customer paid (number, no currency symbol)
- currency: 3-letter ISO code (e.g. INR, USD, EUR)
- vendor_name: business or store name
- description: brief summary of what was purchased
- category: best match from this list: {expense_types}
- confidence: your confidence in the extraction from 0.0 to 1.0

Receipt text:
\"\"\"
{raw_text}
\"\"\"

If a field cannot be determined, set it to null.
Return ONLY the JSON object."""

		response = self._call_llm(prompt)
		return self._parse_llm_response(response)

	def _call_llm(self, prompt: str) -> str:
		"""
		Call the configured provider via LiteLLM (one unified API for OpenAI,
		Gemini, Anthropic, and any OpenAI-compatible endpoint).
		"""
		try:
			import litellm
		except ImportError:
			frappe.throw(_("litellm is not installed. Run: pip install litellm"))

		from frappe.utils.password import get_decrypted_password

		api_key = get_decrypted_password("HR Settings", "HR Settings", "llm_api_key")
		if not api_key:
			frappe.throw(_("LLM API Key is not configured in HR Settings"))

		model = build_llm_model_string()
		base_url = frappe.db.get_single_value("HR Settings", "llm_base_url") or None

		# Let LiteLLM silently drop params a given model doesn't support
		# (e.g. response_format on providers without a native JSON mode).
		litellm.drop_params = True
		response = litellm.completion(
			model=model,
			messages=[{"role": "user", "content": prompt}],
			temperature=0,
			response_format={"type": "json_object"},
			api_key=api_key,
			api_base=base_url,
		)
		return response.choices[0].message.content

	def _parse_llm_response(self, response: str) -> OCRResult:
		import json

		data = json.loads(_extract_json(response))
		result = OCRResult()
		result.expense_date = data.get("expense_date")
		result.amount = data.get("amount")
		result.currency = _validate_currency(data.get("currency"))
		result.vendor_name = data.get("vendor_name")
		result.description = data.get("description")
		result.expense_type_suggestion = data.get("category")
		result.confidence_score = data.get("confidence", 0.0)
		return result


class HeuristicFieldExtractor(BaseFieldExtractor):
	"""
	Free fallback. Uses regex + keyword matching.
	Lower accuracy — works for simple, well-formatted receipts.
	"""

	# Keywords that mark the final payable amount, strongest first.
	# "subtotal" deliberately excluded (it contains "total").
	TOTAL_KEYWORDS = [
		"grand total",
		"amount payable",
		"amount due",
		"total payable",
		"total amount",
		"net total",
		"net payable",
		"net amount",
		"balance due",
		"total inclusive",
		"total incl",
		"total inc",
		"rounded total",
		"bill total",
		"you pay",
		"total",
	]

	# Lines containing these are never the final total (skip them).
	EXCLUDE_AMOUNT_KEYWORDS = [
		"subtotal",
		"sub total",
		"sub-total",
		"change",
		"cash",
		"tender",
		"rounding",
		"round adj",
		"discount",
		"qty",
		"quantity",
		"unit price",
		"item count",
		"no. of",
	]

	# Money value: requires exactly 2 decimals (rejects quantities like "3.0").
	MONEY_RE = r"(?<!\d)(?:\d{1,3}(?:,\d{3})+|\d+)\.\d{2}(?!\d)"

	# Date substrings, ISO first so it isn't partially matched by the d/m/y rule.
	DATE_PATTERNS = [
		r"(?<!\d)\d{4}[/.\-]\d{1,2}[/.\-]\d{1,2}(?!\d)",  # 2018-12-25
		r"(?<!\d)\d{1,2}[/.\-]\d{1,2}[/.\-]\d{2,4}(?!\d)",  # 25/12/2018, 12-01-19, 23.03.18
		r"(?<!\d)\d{1,2}\s+[A-Za-z]{3,9}\.?\s+\d{2,4}(?!\d)",  # 22 MAR 18, 14 May 2026
		r"[A-Za-z]{3,9}\.?\s+\d{1,2},?\s+\d{2,4}(?!\d)",  # May 14, 2026
	]

	def _extract_vendor(self, lines: list[str]) -> str | None:
		"""First meaningful line near the top — usually the store name."""
		skip = ("receipt", "tax invoice", "invoice", "gst", "tel", "date", "bill")
		for line in lines[:6]:
			cleaned = line.strip()
			low = cleaned.lower()
			if not cleaned or any(s in low for s in skip):
				continue
			alpha = sum(c.isalpha() for c in cleaned)
			if alpha >= 3 and alpha >= sum(c.isdigit() for c in cleaned):
				return cleaned
		return None

	def _extract_amount(self, lines: list[str]) -> float | None:
		import re

		def money(s: str) -> list[float]:
			return [float(v.replace(",", "")) for v in re.findall(self.MONEY_RE, s)]

		candidates = []
		for line in lines:
			low = line.lower()
			# subtotal/cash/tender/change/etc. are never the final payable amount
			if any(x in low for x in self.EXCLUDE_AMOUNT_KEYWORDS):
				continue
			vals = money(line)
			if not vals:
				continue
			rank = next((i for i, kw in enumerate(self.TOTAL_KEYWORDS) if kw in low), None)
			if rank is not None:
				candidates.append((rank, vals[-1]))

		if candidates:
			# strongest keyword wins; tie-break on the larger value (grand total)
			candidates.sort(key=lambda c: (c[0], -c[1]))
			return candidates[0][1]

		# fallback: the largest 2-decimal value in the receipt
		allv = [
			v for line in lines if not any(x in line.lower() for x in self.EXCLUDE_AMOUNT_KEYWORDS) for v in money(line)
		]
		return max(allv) if allv else None

	def _extract_date(self, text: str, day_first: bool = True) -> str | None:
		import re

		from dateutil import parser as dateparser

		def first_date(segment: str) -> str | None:
			for pattern in self.DATE_PATTERNS:
				for match in re.findall(pattern, segment):
					try:
						parsed = dateparser.parse(match, dayfirst=day_first)
					except (ValueError, OverflowError, TypeError):
						continue
					if 2000 <= parsed.year <= 2099:
						return parsed.strftime("%Y-%m-%d")
			return None

		# prefer a line that mentions "date"
		for line in text.split("\n"):
			if "date" in line.lower():
				found = first_date(line)
				if found:
					return found
		return first_date(text)

	def extract_fields(self, raw_text: str) -> OCRResult:
		result = OCRResult()
		lines = raw_text.strip().split("\n")

		result.vendor_name = self._extract_vendor(lines)
		result.expense_date = self._extract_date(raw_text, day_first=_prefers_day_first())
		result.amount = self._extract_amount(lines)

		# Currency is not inferred heuristically (symbol->code is ambiguous and the
		# Expense Claim currency defaults to the company currency). Cloud/LLM
		# providers set it from a validated ISO code instead.
		if result.vendor_name:
			result.description = result.vendor_name

		result.expense_type_suggestion = match_expense_type(
			result.vendor_name or "", result.description or ""
		)

		result.confidence_score = 0.4  # Low confidence for heuristic
		return result


# ──────────────────────────────────────────────────────
# LLM provider config (resolved into a LiteLLM model string)
# Add a provider = add one row to each map. LiteLLM handles the rest.
# ──────────────────────────────────────────────────────

# HR Settings "LLM Provider" → LiteLLM provider prefix
LLM_PROVIDER_PREFIX = {
	"OpenAI": "openai",
	"Google Gemini": "gemini",
	"Anthropic": "anthropic",
	"OpenAI-Compatible": "openai",  # paired with an api_base (LLM Base URL)
}

# Default model used when "LLM Model" is left blank
LLM_DEFAULT_MODEL = {
	"OpenAI": "gpt-4o-mini",
	"Google Gemini": "gemini-2.5-flash",
	"Anthropic": "claude-haiku-4-5",
	"OpenAI-Compatible": "",  # must be supplied by the user
}


def build_llm_model_string() -> str:
	"""Resolve HR Settings (provider + model) into a LiteLLM model string, e.g. 'gemini/gemini-2.0-flash'."""
	provider = frappe.db.get_single_value("HR Settings", "llm_provider") or "OpenAI"
	prefix = LLM_PROVIDER_PREFIX.get(provider)
	if prefix is None:
		frappe.throw(_("Unknown LLM provider: {0}").format(provider))

	model = (frappe.db.get_single_value("HR Settings", "llm_model") or "").strip()
	model = model or LLM_DEFAULT_MODEL.get(provider, "")
	if not model:
		frappe.throw(_("LLM Model is required for provider {0}").format(provider))

	if provider == "OpenAI-Compatible" and not frappe.db.get_single_value("HR Settings", "llm_base_url"):
		frappe.throw(_("OpenAI-Compatible provider requires an LLM Base URL"))

	return f"{prefix}/{model}"


# ──────────────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────────────


def _extract_json(text: str) -> str:
	"""Pull the first JSON object out of an LLM response (handles ``` fences / prose)."""
	import re

	if not text:
		return text
	fenced = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", text, re.DOTALL)
	if fenced:
		return fenced.group(1)
	start, end = text.find("{"), text.rfind("}")
	if start != -1 and end != -1 and end > start:
		return text[start : end + 1]
	return text


def _to_float(value) -> float | None:
	import re

	if value is None:
		return None
	if isinstance(value, int | float):
		return float(value)
	cleaned = re.sub(r"[^\d.]", "", str(value).replace(",", ""))
	try:
		return float(cleaned) if cleaned else None
	except ValueError:
		return None


def _normalize_date(value: str) -> str | None:
	from frappe.utils import getdate

	if not value:
		return None  # getdate(None) would return today
	try:
		return str(getdate(value))
	except Exception:
		return None


def _validate_currency(code) -> str | None:
	"""Accept a currency only if it's a real ISO code in the Currency doctype."""
	if not code:
		return None
	code = str(code).strip().upper()
	return code if frappe.db.exists("Currency", code) else None


def _prefers_day_first() -> bool:
	"""Most of the world writes dates day-first; the US is month-first."""
	country = frappe.db.get_single_value("System Settings", "country")
	return (country or "").strip() != "United States"


def _guess_mime_type(file_path: str) -> str:
	import mimetypes

	mime_type, _unused = mimetypes.guess_type(file_path)
	return mime_type or "application/octet-stream"


# ──────────────────────────────────────────────────────
# Factory & Entry Point
# ──────────────────────────────────────────────────────


def get_ocr_provider() -> BaseOCRProvider:
	"""Factory: returns the configured OCR provider from HR Settings."""
	provider_name = frappe.db.get_single_value("HR Settings", "ocr_provider")
	if not provider_name or provider_name == "None":
		frappe.throw(_("Receipt scanning is not configured. Set OCR Provider in HR Settings."))

	providers = {
		"Tesseract": TesseractProvider,
		"Google Cloud Vision": GoogleVisionProvider,
		"Azure Document Intelligence": AzureDocIntelligenceProvider,
	}
	if provider_name not in providers:
		frappe.throw(_("Unknown OCR provider: {0}").format(provider_name))

	return providers[provider_name]()


def get_field_extractor() -> BaseFieldExtractor:
	"""Factory: returns the configured field extractor.
	Only used for providers that return raw text (Tesseract)."""
	method = frappe.db.get_single_value("HR Settings", "ocr_extraction_method")
	if method == "LLM":
		return LLMFieldExtractor()
	return HeuristicFieldExtractor()


def match_expense_type(vendor_name: str, description: str) -> str | None:
	"""
	Suggest an Expense Claim Type whose name appears in the vendor/description.
	Returns None when there's no confident match so the user picks the category
	(an arbitrary default is worse than a blank field).
	"""
	text = f"{vendor_name} {description}".lower()
	if not text.strip():
		return None
	for claim_type in frappe.get_all("Expense Claim Type", pluck="name"):
		if claim_type.lower() in text:
			return claim_type
	return None


def scan_receipt_file(file_url: str) -> dict:
	"""Main entry point — processes a file and returns structured data."""
	if not file_url:
		frappe.throw(_("Please upload a receipt file"))

	file_name = frappe.db.get_value("File", {"file_url": file_url})
	if not file_name:
		frappe.throw(_("Uploaded file could not be found"))

	file_doc = frappe.get_doc("File", file_name)
	file_doc.check_permission("read")  # ensure the user may access this file
	file_path = file_doc.get_full_path()

	provider = get_ocr_provider()
	try:
		result = provider.extract(file_path)
	except frappe.ValidationError:
		raise
	except Exception:
		frappe.log_error(title="Receipt Scanning failed", message=frappe.get_traceback())
		frappe.throw(_("Could not process the receipt. Please try again or enter the expense manually."))

	# Post-process: match expense type if not already set
	if not result.expense_type_suggestion:
		result.expense_type_suggestion = match_expense_type(
			result.vendor_name or "", result.description or ""
		)

	return result.as_dict()
