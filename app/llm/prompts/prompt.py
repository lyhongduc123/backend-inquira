from dataclasses import dataclass
from pathlib import Path

def load_prompt(path: str) -> str:
    base_dir = Path(__file__).resolve().parent
    return (base_dir / path).read_text(encoding="utf-8")

@dataclass(frozen=True)
class PromptDefinition:
    name: str
    version: int
    system_template: str


PROMPT_REGISTRY = {
    "generate_answer": PromptDefinition(
        name="generate_answer",
        version=1,
        system_template=load_prompt("system/generate_answer.md"),
    ),
    "generate_no_results_guidance": PromptDefinition(
        name="generate_no_results_guidance",
        version=1,
        system_template=load_prompt("system/generate_no_results_guidance.md"),
    ),
    "generate_answer_scoped": PromptDefinition(
        name="generate_answer_scoped",
        version=1,
        system_template=load_prompt("system/generate_answer_scoped.md"),
    ),
    "generate_sub_agent_summary": PromptDefinition(
        name="generate_sub_agent_summary",
        version=1,
        system_template=load_prompt("system/generate_sub_agent_summary.md"),
    ),
    "generate_literature_review_brief": PromptDefinition(
        name="generate_literature_review_brief",
        version=1,
        system_template=load_prompt("system/generate_literature_review_brief.md"),
    ),
    "decompose_query": PromptDefinition(
        name="decompose_query",
        version=1,
        system_template=load_prompt("system/decompose_query.md"),
    ),
    "decompose_query_v2": PromptDefinition(
        name="decompose_query_v2",
        version=2,
        system_template=load_prompt("system/decompose_query_v2.md"),
    ),
    "decompose_query_v3": PromptDefinition(
        name="decompose_query_v3",
        version=3,
        system_template=load_prompt("system/decompose_query_v3.md"),
    ),
    "conversation_summarization": PromptDefinition(
        name="conversation_summarization",
        version=1,
        system_template=load_prompt("system/summarize_conversation.md"),
    ),
}

class PromptBuilder:

    @staticmethod
    def build(
        prompt_name: str,
        user_input: str,
        additional_content: str | None = None,
        dynamic_instruction: str | None = None,
    ):
        prompt_def = PROMPT_REGISTRY[prompt_name]
        system_content = prompt_def.system_template
        if dynamic_instruction:
            system_content += (
                "\n\n----------------------------------------\n"
                "ADDITIONAL USER INSTRUCTIONS:\n"
                f"{dynamic_instruction.strip()}"
            )

        messages = [
            {"role": "system", "content": system_content},
        ]

        if additional_content:
            messages.append({
                "role": "system",
                "content": additional_content
            })

        messages.append({
            "role": "user",
            "content": user_input
        })

        return messages, prompt_def.version