"""
Classifies each incoming question into one of four intents and, for
cross-paper comparisons, decomposes it into focused sub-questions.

Query types:
  * FACT        -> a specific, narrow fact/number/definition/experiment
                   result, usually scoped to one paper.
  * SUMMARY      -> "summarize X", "what does paper Y argue", broad
                   whole-paper or whole-concept synthesis.
  * COMPARISON   -> requires combining/contrasting information from
                   multiple papers.
  * GENERAL      -> anything else / small talk / doesn't need retrieval.

Routine facts and obvious summaries are classified locally. Gemini is used
only when a comparison needs query decomposition.
"""
from __future__ import annotations

import logging
import re
from typing import List, Literal, Optional

from pydantic import BaseModel, Field
from langchain_google_genai import ChatGoogleGenerativeAI

import config

logger = logging.getLogger(__name__)

_COMPARISON_PATTERN = re.compile(
    r"\b(compare|contrast|difference|differ|versus|vs\.?|compared with|compared to)\b",
    re.IGNORECASE,
)
_SUMMARY_PATTERN = re.compile(
    r"\b(summari[sz]e|summary|overview|main contribution|contributions?|limitations?)\b",
    re.IGNORECASE,
)


class RouteDecision(BaseModel):
    query_type: Literal["FACT", "SUMMARY", "COMPARISON", "GENERAL"] = Field(
        description="The category of the user's question."
    )
    paper_hint: Optional[str] = Field(
        default=None,
        description="If the user named or clearly implied a specific paper/title/filename, extract it. Else null.",
    )
    sub_questions: List[str] = Field(
        default_factory=list,
        description=(
            "Only for COMPARISON queries: 2-4 focused, self-contained sub-questions "
            "that together cover what needs to be retrieved from each relevant paper."
        ),
    )


_llm = None


def get_router_llm():
    global _llm
    if _llm is None:
        _llm = ChatGoogleGenerativeAI(
            model=config.GEMINI_MODEL,
            temperature=0,
            google_api_key=config.GOOGLE_API_KEY,
            retries=0,
        )
    return _llm


ROUTER_SYSTEM_PROMPT = """You are a query router for an academic literature RAG system \
over a folder of research papers. Classify the user's question and, if it is a \
COMPARISON question (needs combining info from 2+ papers), break it into 2-4 \
focused sub-questions, one per aspect/paper needed. Respond ONLY with the \
structured output requested, no extra commentary."""


def route(
    question: str,
    paper_titles: List[str] | None = None,
) -> RouteDecision:
    question_folded = question.casefold()
    paper_hint = max(
        (
            title
            for title in (paper_titles or [])
            if title and title.casefold() in question_folded
        ),
        key=len,
        default=None,
    )

    if not _COMPARISON_PATTERN.search(question):
        query_type = "SUMMARY" if _SUMMARY_PATTERN.search(question) else "FACT"
        decision = RouteDecision(query_type=query_type, paper_hint=paper_hint)
        logger.info("Locally routed question -> %s (paper_hint=%s)", query_type, paper_hint)
        return decision

    llm = get_router_llm().with_structured_output(RouteDecision)
    decision = llm.invoke(
        [
            ("system", ROUTER_SYSTEM_PROMPT),
            ("human", question),
        ]
    )
    if isinstance(decision, RouteDecision):
        return decision
    if isinstance(decision, dict):
        return RouteDecision.model_validate(decision)
    raise TypeError("The query router returned an unexpected response type.")
