from app.llm.prompts.prompt_registry import PROMPT_REGISTRY, PromptKey


class PromptBuilder:
    @staticmethod
    def build(
        key: PromptKey,
        user_input: str,
        past_context: str | None = None,
        history: list[dict] | None = None,
    ) -> list[dict]:
        prompt_definition = PROMPT_REGISTRY.get(key)
        if prompt_definition is None:
            raise KeyError(f"Prompt key not found: {key}")

        messages = [
            {"role": "system", "content": prompt_definition.system_template},
        ]

        if past_context:
            messages.append({"role": "system", "content": f"[Past context]\n{past_context}"})
            
        if history:
            messages.extend(history)

        messages.append({"role": "user", "content": user_input})
            
        return messages