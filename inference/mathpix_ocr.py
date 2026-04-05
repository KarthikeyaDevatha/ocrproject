"""
Mathpix OCR API Client for Adaptive OCR Agent.

Handles communication with Mathpix v3/text endpoint.
Features:
- Image file upload via multipart
- Base64 encoding support
- Image hash caching to avoid duplicate API calls
- MATHPIX_MOCK mode for demo safety
"""

import os
import json
import hashlib
import base64
from typing import Optional, Dict, Any
from dataclasses import dataclass


# Mock response for demo/quota safety
MOCK_RESPONSE = {
    "text": "\\int_0^1 x^2 dx = \\frac{1}{3}",
    "latex_styled": "\\int_0^1 x^2 \\, dx = \\frac{1}{3}",
    "confidence": 0.91,
    "confidence_rate": 0.91,
    "is_mock": True
}


@dataclass
class MathpixResult:
    """Structured result from Mathpix API."""
    text: str = ""
    latex_styled: str = ""
    confidence: float = 0.0
    confidence_rate: float = 0.0
    html: str = ""
    line_data: list = None
    word_data: list = None
    is_mock: bool = False
    raw_response: dict = None
    error: str = ""

    def __post_init__(self):
        if self.line_data is None:
            self.line_data = []
        if self.word_data is None:
            self.word_data = []
        if self.raw_response is None:
            self.raw_response = {}


class MathpixOCR:
    """
    Mathpix OCR API client.
    
    Auth: MATHPIX_APP_ID + MATHPIX_APP_KEY environment variables.
    Mock: Set MATHPIX_MOCK=true for demo safety.
    Cache: SHA256 hash of image bytes → cached response.
    """

    API_URL = "https://api.mathpix.com/v3/text"

    def __init__(
        self,
        app_id: Optional[str] = None,
        app_key: Optional[str] = None
    ):
        self.app_id = app_id or os.getenv("MATHPIX_APP_ID", "")
        self.app_key = app_key or os.getenv("MATHPIX_APP_KEY", "")
        self._cache: Dict[str, MathpixResult] = {}

    @property
    def is_mock_mode(self) -> bool:
        return os.getenv("MATHPIX_MOCK", "false").lower() == "true"

    @property
    def is_configured(self) -> bool:
        return bool(self.app_id and self.app_key)

    @property
    def is_available(self) -> bool:
        return self.is_configured or self.is_mock_mode

    def _hash_image(self, image_bytes: bytes) -> str:
        """SHA256 hash of image bytes for caching."""
        return hashlib.sha256(image_bytes).hexdigest()

    def recognize_image(
        self,
        image_path: str,
        include_line_data: bool = True,
        include_word_data: bool = False,
        math_inline_delimiters: list = None,
        formats: list = None
    ) -> MathpixResult:
        """
        Send image to Mathpix OCR API.
        
        Args:
            image_path: Path to the image file
            include_line_data: Request per-line results with contours
            include_word_data: Request per-word results
            math_inline_delimiters: Math delimiters, default ["$", "$"]
            formats: Output formats, default ["text", "data", "html"]
        
        Returns:
            MathpixResult with text, LaTeX, confidence, and line data
        """
        if math_inline_delimiters is None:
            math_inline_delimiters = ["$", "$"]
        if formats is None:
            formats = ["text", "data", "html"]

        # Read image
        try:
            with open(image_path, "rb") as f:
                image_bytes = f.read()
        except (FileNotFoundError, IOError) as e:
            return MathpixResult(error=f"Cannot read image: {e}")

        # Check cache
        img_hash = self._hash_image(image_bytes)
        if img_hash in self._cache:
            result = self._cache[img_hash]
            return result

        # Mock mode
        if self.is_mock_mode:
            result = MathpixResult(
                text=MOCK_RESPONSE["text"],
                latex_styled=MOCK_RESPONSE["latex_styled"],
                confidence=MOCK_RESPONSE["confidence"],
                confidence_rate=MOCK_RESPONSE["confidence_rate"],
                is_mock=True,
                raw_response=MOCK_RESPONSE
            )
            self._cache[img_hash] = result
            return result

        # Check credentials
        if not self.is_configured:
            return MathpixResult(
                error="Mathpix API credentials not configured. "
                      "Set MATHPIX_APP_ID and MATHPIX_APP_KEY environment variables."
            )

        # Build options
        options = {
            "math_inline_delimiters": math_inline_delimiters,
            "rm_spaces": True,
            "formats": formats,
            "data_options": {
                "include_asciimath": True,
                "include_latex": True
            }
        }
        if include_line_data:
            options["include_line_data"] = True
        if include_word_data:
            options["include_word_data"] = True

        # Make API call
        try:
            import requests

            headers = {
                "app_id": self.app_id,
                "app_key": self.app_key,
            }

            response = requests.post(
                self.API_URL,
                files={"file": ("image.jpg", image_bytes)},
                data={"options_json": json.dumps(options)},
                headers=headers,
                timeout=30
            )

            if response.status_code != 200:
                return MathpixResult(
                    error=f"Mathpix API error {response.status_code}: {response.text}"
                )

            data = response.json()

        except ImportError:
            return MathpixResult(
                error="requests library not installed. pip install requests"
            )
        except Exception as e:
            return MathpixResult(error=f"Mathpix API call failed: {e}")

        # Parse response
        result = MathpixResult(
            text=data.get("text", ""),
            latex_styled=data.get("latex_styled", ""),
            confidence=data.get("confidence", 0.0),
            confidence_rate=data.get("confidence_rate", 0.0),
            html=data.get("html", ""),
            line_data=data.get("line_data", []),
            word_data=data.get("word_data", []),
            is_mock=False,
            raw_response=data
        )

        # Cache
        self._cache[img_hash] = result
        return result

    def recognize_base64(
        self,
        image_bytes: bytes,
        mime_type: str = "image/png",
        **kwargs
    ) -> MathpixResult:
        """
        Send base64-encoded image to Mathpix API.
        
        Args:
            image_bytes: Raw image bytes
            mime_type: MIME type of the image
        
        Returns:
            MathpixResult
        """
        # Check cache
        img_hash = self._hash_image(image_bytes)
        if img_hash in self._cache:
            return self._cache[img_hash]

        if self.is_mock_mode:
            result = MathpixResult(
                text=MOCK_RESPONSE["text"],
                latex_styled=MOCK_RESPONSE["latex_styled"],
                confidence=MOCK_RESPONSE["confidence"],
                confidence_rate=MOCK_RESPONSE["confidence_rate"],
                is_mock=True,
                raw_response=MOCK_RESPONSE
            )
            self._cache[img_hash] = result
            return result

        if not self.is_configured:
            return MathpixResult(
                error="Mathpix API credentials not configured."
            )

        b64_data = base64.b64encode(image_bytes).decode("utf-8")
        src = f"data:{mime_type};base64,{b64_data}"

        options = {
            "src": src,
            "math_inline_delimiters": kwargs.get("math_inline_delimiters", ["$", "$"]),
            "rm_spaces": True,
            "formats": kwargs.get("formats", ["text", "data", "html"]),
            "data_options": {
                "include_asciimath": True,
                "include_latex": True
            }
        }
        if kwargs.get("include_line_data", True):
            options["include_line_data"] = True

        try:
            import requests
            response = requests.post(
                self.API_URL,
                json=options,
                headers={
                    "app_id": self.app_id,
                    "app_key": self.app_key,
                    "Content-type": "application/json"
                },
                timeout=30
            )

            if response.status_code != 200:
                return MathpixResult(
                    error=f"Mathpix API error {response.status_code}: {response.text}"
                )

            data = response.json()

        except Exception as e:
            return MathpixResult(error=f"Mathpix API call failed: {e}")

        result = MathpixResult(
            text=data.get("text", ""),
            latex_styled=data.get("latex_styled", ""),
            confidence=data.get("confidence", 0.0),
            confidence_rate=data.get("confidence_rate", 0.0),
            html=data.get("html", ""),
            line_data=data.get("line_data", []),
            word_data=data.get("word_data", []),
            is_mock=False,
            raw_response=data
        )
        self._cache[img_hash] = result
        return result

    def clear_cache(self):
        """Clear the response cache."""
        self._cache.clear()
