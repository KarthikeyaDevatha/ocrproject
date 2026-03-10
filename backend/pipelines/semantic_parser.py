"""Semantic reasoning layer to build assignment structure from OCR blocks."""

from __future__ import annotations

import json
import os
import re
from typing import List

from backend.pipelines.ocr_router import OCRResult


class SemanticParser:
    def __init__(self) -> None:
        self.api_key = os.getenv("OPENAI_API_KEY", "")

    def parse_document(self, ocr_results: List[OCRResult]) -> dict:
        payload = [r.__dict__ for r in ocr_results]
        if self.api_key:
            llm = self._parse_with_llm(payload)
            if llm:
                return llm

        blocks = []
        for idx, item in enumerate(ocr_results, start=1):
            text = item.text or item.latex
            problem = self.classify_problem(text)
            steps = self.extract_steps(text)
            blocks.append(
                {
                    "block_id": f"b{idx:03d}",
                    "type": "question" if idx == 1 else "answer",
                    "raw_text": text,
                    "latex": item.latex,
                    "problem_type": problem,
                    "subject": "mathematics",
                    "solution_steps": steps,
                    "confidence": item.confidence,
                    "bounding_box": [0, 0, 0, 0],
                }
            )
        return {
            "pages": [{"page_number": 1, "blocks": blocks}],
            "metadata": {
                "subject": "mathematics",
                "grade_level": "grade_10",
                "total_questions": len(blocks),
            },
        }

    def classify_problem(self, latex: str) -> str:
        text = latex.lower()
        if "\\int" in text:
            return "calculus_integral"
        if "x^2" in text or "quadratic" in text:
            return "quadratic_equation"
        if any(tok in text for tok in ["h2o", "co2", "na", "cl"]):
            return "chemistry_formula"
        return "general_problem"

    def extract_steps(self, text: str) -> List[str]:
        parts = re.split(r"(?:\n|\r|\t|\d+\.)", text)
        cleaned = [p.strip() for p in parts if p.strip()]
        return cleaned[:8]

    def _parse_with_llm(self, payload: list[dict]) -> dict | None:
        try:
            from openai import OpenAI

            client = OpenAI(api_key=self.api_key)
            prompt = (
                "You are a scientific document parser. Given OCR output from a student assignment, "
                "extract problem type, question boundaries, solution steps, subject, grade level. "
                "Return ONLY valid JSON.\n"
                f"OCR={json.dumps(payload)}"
            )
            resp = client.responses.create(model="gpt-4o-mini", input=prompt)
            return json.loads(resp.output_text)
        except Exception:
            return None
