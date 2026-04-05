import json
import logging
import os
from typing import List, Dict, Any

try:
    import openai
except ImportError:
    openai = None

class SemanticParser:
    def __init__(self, api_key: str = None):
        self.api_key = api_key or os.environ.get("LLM_API_KEY", "")
        if self.api_key and openai is not None:
            openai.api_key = self.api_key
            
        self.system_prompt = """
        You are a scientific document parser.
        Given OCR output from a student assignment, extract:
        1. Problem type (algebra / calculus / chemistry / geometry / statistics / unknown)
        2. Question boundaries (where each question starts and ends)
        3. Solution steps (numbered reasoning steps if present)
        4. Subject classification
        5. Estimated grade level
        Return ONLY valid JSON matching DocumentStructure schema.
        No code blocks or markdown, just raw JSON string.
        """

    def _call_llm(self, user_prompt: str) -> str:
        """Helper to invoke LLM with robust error handling."""
        if not self.api_key or openai is None:
            logging.error("LLM API key missing or openai not installed.")
            return "{}"

        try:
            # Assuming openai < 1.0. For > 1.0 client=OpenAI() is used. Using generic dict structure.
            response = openai.ChatCompletion.create(
                model="gpt-4o",
                messages=[
                    {"role": "system", "content": self.system_prompt},
                    {"role": "user", "content": user_prompt}
                ],
                temperature=0.0
            )
            content = response.choices[0].message.content.strip()
            # Strip markdown json blocks if returned
            if content.startswith("```json"):
                content = content[7:-3].strip()
            elif content.startswith("```"):
                content = content[3:-3].strip()
            return content
        except Exception as e:
            logging.error(f"Semantic parsing failed: {e}")
            return "{}"

    def parse_document(self, ocr_results: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Takes raw OCR blocks and groups/parses them into structured document JSON.
        ocr_results format: [{"region_id": str, "text": str, "latex": str, "type": str}, ...]
        """
        # Linearize OCR results for LLM
        linear_text = ""
        for i, block in enumerate(ocr_results):
            txt = block.get('latex') or block.get('text', '')
            b_type = block.get('type', 'text')
            linear_text += f"Block {i} [{b_type}]: {txt}\n"
            
        prompt = f"Parse the following OCR blocks:\n\n{linear_text}"
        
        json_resp = self._call_llm(prompt)
        try:
            return json.loads(json_resp)
        except json.JSONDecodeError:
            logging.error("Semantic Parser returned invalid JSON")
            return {
                "error": "Failed to parse JSON from LLM",
                "raw_response": json_resp
            }

    def classify_problem(self, latex: str) -> str:
        """
        Quickly classify a single LaTeX snippet.
        """
        prompt = f"Classify this problem type. Return ONLY one word: algebra, calculus, chemistry, geometry, statistics. \nInput: {latex}"
        resp = self._call_llm(prompt)
        # Fallback to dictionary extraction to be safe
        valid_classes = ["algebra", "calculus", "chemistry", "geometry", "statistics"]
        for vc in valid_classes:
            if vc in resp.lower():
                return vc
        return "unknown"

    def extract_steps(self, text: str) -> List[str]:
        """
        Extract numerical reasoning or solution steps from combined text block.
        """
        prompt = f"Extract solution steps from this text. Return a JSON array of strings for each step. \nText: {text}"
        resp = self._call_llm(prompt)
        try:
            data = json.loads(resp)
            if isinstance(data, list):
                return data
            if isinstance(data, dict) and "steps" in data:
                return data["steps"]
        except Exception:
            # Fallback to newline splitting if JSON fails
            pass
            
        return [step.strip() for step in text.split('\n') if step.strip()]
