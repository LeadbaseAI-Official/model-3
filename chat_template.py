def format_chat_prompt(prompt: str, system_prompt: str = "You are a helpful assistant.") -> str:
    if system_prompt:
        return f"<start_of_turn>user\n{system_prompt}\n\n{prompt}<end_of_turn>\n<start_of_turn>model\n"
    return f"<start_of_turn>user\n{prompt}<end_of_turn>\n<start_of_turn>model\n"
