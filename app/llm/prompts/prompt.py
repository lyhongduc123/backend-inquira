from dataclasses import dataclass

def load_prompt(path: str) -> str:
    with open(f"app/llm/prompts/{path}", "r") as f:
        return f.read()

@dataclass(frozen=True)
class PromptDefinition:
    name: str
    version: int
    system_template: str


PROMPT_REGISTRY = {
    "generate_answer": PromptDefinition(
        name="generate_answer",
        version=1,
        system_template=load_prompt("system/generate_answer.txt"),
    ),
    "decompose_query": PromptDefinition(
        name="decompose_query",
        version=1,
        system_template=load_prompt("system/decompose_query.txt"),
    ),
    "decompose_query_v2": PromptDefinition(
        name="decompose_query_v2",
        version=2,
        system_template=load_prompt("system/decompose_query_v2.txt"),
    ),
    "conversation_summarization": PromptDefinition(
        name="conversation_summarization",
        version=1,
        system_template=load_prompt("system/summarize_conversation.txt"),
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