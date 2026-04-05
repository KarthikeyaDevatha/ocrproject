import pytest
from backend.pipelines.semantic_parser import SemanticParser

def test_semantic_parser_init():
    parser = SemanticParser(api_key="dummy")
    assert parser is not None

def test_classify_problem():
    parser = SemanticParser(api_key="dummy")
    # Will fail if openai is not imported or dummy key rejects, 
    # but the logic runs if we mock _call_llm.
    # For CI safety, we just patch _call_llm manually
    parser._call_llm = lambda x: "algebra"
    res = parser.classify_problem("x = 5")
    assert res == "algebra"

def test_extract_steps():
    parser = SemanticParser(api_key="dummy")
    parser._call_llm = lambda x: '["Step 1", "Step 2"]'
    res = parser.extract_steps("Step 1\nStep 2")
    assert isinstance(res, list)
    assert len(res) == 2
