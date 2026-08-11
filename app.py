import logging

import streamlit as st

from config import get_config
from llm.groq_client import GroqClientError, stream_chat_completion
from memory.chat_memory import ChatMemory
from prompts.system_prompt import SYSTEM_PROMPT


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s - %(message)s",
)

def main() -> None:
    config = get_config()
    memory = ChatMemory(
        history_path=config.history_path,
        recent_message_limit=config.recent_message_limit,
    )

    st.set_page_config(page_title="Stage 2 Memory Assistant", page_icon="AI", layout="centered")
    st.title("Stage 2 Memory Assistant")

    if "messages" not in st.session_state:
        load_result = memory.load_history()
        st.session_state.messages = load_result.messages
        if load_result.warning:
            st.warning(load_result.warning)

    with st.sidebar:
        st.header("Model Settings")
        model = st.text_input("Model", value=config.default_model)
        temperature = st.slider(
            "Temperature",
            min_value=0.0,
            max_value=2.0,
            value=config.default_temperature,
            step=0.1,
        )
        max_tokens = st.slider(
            "Max tokens",
            min_value=64,
            max_value=4096,
            value=config.default_max_tokens,
            step=64,
        )

        st.divider()
        if config.groq_api_key:
            st.success("GROQ_API_KEY loaded")
        else:
            st.error("GROQ_API_KEY is missing")

        st.header("Memory")
        st.caption(f"Persistent store: {config.history_path}")
        st.caption(f"Recent context limit: {config.recent_message_limit} messages")

        if st.button("Clear memory", type="secondary"):
            st.session_state.messages = memory.clear_history()
            st.rerun()

        with st.expander("Inspect memory"):
            st.json(st.session_state.messages)

    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    user_question = st.chat_input("Ask a question")
    if not user_question:
        return

    memory.add_message(st.session_state.messages, "user", user_question)
    memory.save_history(st.session_state.messages)

    with st.chat_message("user"):
        st.markdown(user_question)

    api_messages = memory.build_context(
        system_prompt=SYSTEM_PROMPT,
        messages=st.session_state.messages,
    )

    with st.chat_message("assistant"):
        response_placeholder = st.empty()
        full_response = ""

        try:
            for token in stream_chat_completion(
                config=config,
                messages=api_messages,
                model=model,
                temperature=temperature,
                max_tokens=max_tokens,
            ):
                full_response += token
                response_placeholder.markdown(full_response)

        except GroqClientError as exc:
            st.error(str(exc))
            return

    memory.add_message(st.session_state.messages, "assistant", full_response)
    memory.save_history(st.session_state.messages)


if __name__ == "__main__":
    main()
