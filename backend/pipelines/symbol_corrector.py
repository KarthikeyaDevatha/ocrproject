import logging
import os
import json
try:
    import openai
except ImportError:
    openai = None

class SymbolCorrector:
    def __init__(self, api_key: str = None):
        """
        Initializes the symbol corrector.
        Requires LLM capabilities for context-aware correction.
        """
        self.api_key = api_key or os.environ.get("LLM_API_KEY", "")
        if self.api_key and openai is not None:
            openai.api_key = self.api_key
        else:
            logging.warning("No LLM_API_KEY provided or openai not installed. "
                            "Context-aware correction will be a pass-through.")
                            
        self.hard_rules = {
            "math": {
                # Common confusions that are almost always wrong in standard math contexts
                " Z ": " 2 ",
                " l ": " 1 ",
                " O ": " 0 ",
                " S ": " 5 ",
                " q ": " 9 ",
                " I ": " 1 ",
                # LaTeX typical confusions
                "\\sin(x)": "\\sin(x)",
                "\\ast": "\\ast"
            },
            "general": {}
        }

    def correct_rules(self, text: str, context: str = "math") -> str:
        """
        Applies hardcoded, reliable find-replace rules based on context.
        """
        if not text:
            return text
            
        corrected = text
        rules = self.hard_rules.get(context, {})
        
        # We might need regex for whole words or specific spacing. 
        # Using exact boundaries for single characters.
        import re
        for wrong, right in rules.items():
            if len(wrong.strip()) == 1 and len(right.strip()) == 1:
                # E.g. 'Z' -> '2' standalone or surrounded by operators
                pattern = r'\b' + re.escape(wrong.strip()) + r'\b'
                corrected = re.sub(pattern, right.strip(), corrected)
            else:
                corrected = corrected.replace(wrong, right)
                
        return corrected

    def correct_with_llm(self, text: str, context: str) -> str:
        """
        Sends ambiguous OCR output to an LLM.
        Prompt: 'Fix OCR errors in this {context}. Only fix clear character recognition errors. 
                 Return corrected output only.'
        """
        if not text or not self.api_key or openai is None:
            return text

        prompt = (f"Fix OCR errors in this {context}. "
                  f"Only fix clear character recognition errors. "
                  f"Do not solve or evaluate. "
                  f"Return corrected text only, without quotes or markdown blocks. \n\n"
                  f"Input:\n{text}")

        try:
            # Placeholder for actual OpenAI v1 / Claude invocation
            # Using generic syntax assuming openai<1.0 or similar generic wrapper
            response = openai.ChatCompletion.create(
                model="gpt-4o",
                messages=[{"role": "user", "content": prompt}],
                temperature=0.0
            )
            corrected = response.choices[0].message.content.strip()
            return corrected
        except Exception as e:
            logging.error(f"LLM correction failed: {e}. Returning original.")
            return text

    def correction_pipeline(self, text: str, context: str = "math") -> str:
        """
        Runs the full correction pipeline: rules -> LLM.
        """
        if not text:
            return ""
            
        # 1. Apply fast rule-based heuritsics
        text_rules = self.correct_rules(text, context)
        
        # 2. Apply rigorous LLM check if string is sufficiently complex
        # Simple heuristic to avoid LLM limits on trivial stuff (e.g., "x=2")
        if len(text_rules) > 5 and "\\" in text_rules: 
            text_llm = self.correct_with_llm(text_rules, context)
            return text_llm
            
        # For non-math or simple text, returning rule-corrected text
        return text_rules
