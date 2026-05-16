"""Reusable conversation setup helpers for chat entrypoints."""

from dataclasses import dataclass
from typing import Any, Optional


@dataclass(frozen=True)
class ConversationSetupResult:
    """Resolved conversation target for a chat operation."""

    conversation: Any
    is_new: bool
    was_cloned: bool

    @property
    def conversation_id(self) -> str:
        return str(self.conversation.conversation_id)


async def resolve_conversation_for_user(
    conversation_service,
    user_id: int,
    conversation_id: Optional[str],
    new_conversation_title: Optional[str] = None,
) -> Optional[ConversationSetupResult]:
    """Resolve an owned conversation for user, cloning if needed, or create new.

    Returns None only when a requested conversation_id does not exist.
    """

    if conversation_id:
        source_or_clone, was_cloned = (
            await conversation_service.get_or_clone_conversation_for_user(
                conversation_id=conversation_id,
                user_id=user_id,
            )
        )
        if not source_or_clone:
            return None

        return ConversationSetupResult(
            conversation=source_or_clone,
            is_new=False,
            was_cloned=was_cloned,
        )

    conversation = await conversation_service.create_conversation(
        user_id=user_id,
        title=new_conversation_title,
    )
    return ConversationSetupResult(
        conversation=conversation,
        is_new=True,
        was_cloned=False,
    )
