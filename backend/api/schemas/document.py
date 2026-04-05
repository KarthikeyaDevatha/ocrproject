from __future__ import annotations

from typing import List

from pydantic import BaseModel


class Block(BaseModel):
    block_id: str
    type: str
    raw_text: str
    latex: str = ""
    problem_type: str
    subject: str
    solution_steps: List[str]
    confidence: float
    bounding_box: List[int]


class Page(BaseModel):
    page_number: int
    blocks: List[Block]


class Metadata(BaseModel):
    subject: str
    grade_level: str
    total_questions: int
    processing_time_ms: int


class DocumentOutput(BaseModel):
    document_id: str
    pages: List[Page]
    metadata: Metadata
