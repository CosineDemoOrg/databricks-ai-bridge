from typing import Any, Iterable, List


def _coerce_content_empty(message: Any) -> Any:
    # Prefer pydantic's model_copy if available (LangChain BaseMessage types)
    try:
        if hasattr(message, "model_copy"):
            return message.model_copy(update={"content": ""})
    except Exception:
        pass
    # Fallback: set attribute in place
    try:
        if getattr(message, "content", None) is None:
            setattr(message, "content", "")
            return message
    except Exception:
        pass
    # Dict-style message
    if isinstance(message, dict):
        new_msg = dict(message)
        if new_msg.get("content", None) is None:
            new_msg["content"] = ""
        return new_msg
    return message


def sanitize_messages_for_databricks(messages: Iterable[Any]) -> List[Any]:
    """
    Sanitize a sequence of messages before sending to Databricks chat endpoints with input guardrails.

    - Drops assistant messages that contain tool_calls (these serialize with content=null).
    - Coerces any message with content=None to content="".
    """
    sanitized: List[Any] = []
    for m in messages:
        # Drop assistant tool-call messages (identified by presence of 'tool_calls' attribute)
        tool_calls = getattr(m, "tool_calls", None)
        if tool_calls:
            # Keep ToolMessages; they do not have 'tool_calls'
            continue

        content = getattr(m, "content", None)
        if isinstance(m, dict):
            content = m.get("content", content)

        if content is None:
            m = _coerce_content_empty(m)

        sanitized.append(m)
    return sanitized


def sanitize_input_for_databricks(input_obj: Any) -> Any:
    """
    Accepts ChatPromptValue, list of messages, or {'messages': [...]}.
    Returns a list of sanitized messages suitable for model invocation.
    """
    # ChatPromptValue-like
    if hasattr(input_obj, "to_messages"):
        messages = input_obj.to_messages()
        return sanitize_messages_for_databricks(messages)

    # Dict-style state with messages
    if isinstance(input_obj, dict) and "messages" in input_obj:
        return sanitize_messages_for_databricks(input_obj["messages"])

    # Already a list of messages
    if isinstance(input_obj, list):
        return sanitize_messages_for_databricks(input_obj)

    # Unknown input type; return as-is
    return input_obj


def make_sanitizer_runnable():
    """
    Returns a RunnableLambda that can be inserted between a prompt and the model:
      date_assistant_runnable = date_assistant_prompt | make_sanitizer_runnable() | llm.bind_tools([...])

    This requires langchain-core to be installed.
    """
    try:
        from langchain_core.runnables import RunnableLambda
    except Exception as e:
        raise ImportError(
            "langchain-core is required to create a Runnable sanitizer. "
            "Install langchain-core or import and use sanitize_input_for_databricks directly."
        ) from e

    return RunnableLambda(sanitize_input_for_databricks)