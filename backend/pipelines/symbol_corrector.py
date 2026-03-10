"""Rule-based + optional LLM symbol correction for OCR outputs."""

from __future__ import annotations

import os


class SymbolCorrector:
    MAP = {"Z": "2", "l": "1", "O": "0", "S": "5", "q": "9", "I": "1"}

    def __init__(self) -> None:
        self.api_key = os.getenv("OPENAI_API_KEY", "")

    def correct_rules(self, text: str) -> str:
        out = text
        for src, dst in self.MAP.items():
            out = out.replace(src, dst)
        return out

    def correct_with_llm(self, text: str, context: str) -> str:
        if not self.api_key:
            return text
        try:
            from openai import OpenAI

            client = OpenAI(api_key=self.api_key)
            prompt = (
                "Fix OCR errors in this LaTeX. Only fix clear character recognition errors. "
                "Return corrected LaTeX only.\n"
                f"context={context}\n"
                f"input={text}"
            )
            resp = client.responses.create(model="gpt-4o-mini", input=prompt)
            return (resp.output_text or text).strip()
        except Exception:
            return text

    def correction_pipeline(self, text: str, context: str = "math") -> str:
        rule = self.correct_rules(text)
        return self.correct_with_llm(rule, context)
