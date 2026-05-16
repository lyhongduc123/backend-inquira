from dataclasses import dataclass
from enum import StrEnum
from pathlib import Path
from typing import Dict

def load_prompt(path: str) -> str:
    base_dir = Path(__file__).resolve().parent
    return (base_dir / path).read_text(encoding="utf-8")

@dataclass(frozen=True)
class PromptDefinition:
    name: str
    version: int
    system_template: str
    
    from enum import StrEnum


class PromptKey(StrEnum):
    GENERATE_ANSWER = "generate_answer"
    GENERATE_NO_RESULTS_GUIDANCE = "generate_no_results_guidance"
    GENERATE_ANSWER_SCOPED = "generate_answer_scoped"
    GENERATE_SUB_AGENT_SUMMARY = "generate_sub_agent_summary"
    LITERATURE_REVIEW_BRIEF = "generate_literature_review_brief"

    DECOMPOSE_QUERY = "decompose_query"
    DECOMPOSE_QUERY_V2 = "decompose_query_v2"
    DECOMPOSE_QUERY_V3 = "decompose_query_v3"

    CONVERSATION_SUMMARIZATION = "conversation_summarization"


PROMPT_REGISTRY: Dict[PromptKey, PromptDefinition] = {
    PromptKey.GENERATE_ANSWER: PromptDefinition(
        name="generate_answer",
        version=1,
        system_template=load_prompt("system/generate_answer.md"),
    ),
    PromptKey.GENERATE_NO_RESULTS_GUIDANCE: PromptDefinition(
        name="generate_no_results_guidance",
        version=1,
        system_template=load_prompt("system/generate_no_results_guidance.md"),
    ),
    PromptKey.GENERATE_ANSWER_SCOPED: PromptDefinition(
        name="generate_answer_scoped",
        version=1,
        system_template=load_prompt("system/generate_answer_scoped.md"),
    ),
    PromptKey.GENERATE_SUB_AGENT_SUMMARY: PromptDefinition(
        name="generate_sub_agent_summary",
        version=1,
        system_template=load_prompt("system/generate_sub_agent_summary.md"),
    ),
    PromptKey.LITERATURE_REVIEW_BRIEF: PromptDefinition(
        name="generate_literature_review_brief",
        version=1,
        system_template=load_prompt("system/generate_literature_review_brief.md"),
    ),
    PromptKey.DECOMPOSE_QUERY: PromptDefinition(
        name="decompose_query",
        version=1,
        system_template=load_prompt("system/decompose_query.md"),
    ),
    PromptKey.DECOMPOSE_QUERY_V2: PromptDefinition(
        name="decompose_query_v2",
        version=2,
        system_template=load_prompt("system/decompose_query_v2.md"),
    ),
    PromptKey.DECOMPOSE_QUERY_V3: PromptDefinition(
        name="decompose_query_v3",
        version=3,
        system_template=load_prompt("system/decompose_query_v3.md"),
    ),
    PromptKey.CONVERSATION_SUMMARIZATION: PromptDefinition(
        name="conversation_summarization",
        version=1,
        system_template=load_prompt("system/summarize_conversation.md"),
    ),
}