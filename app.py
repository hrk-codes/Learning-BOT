import logging

import streamlit as st

from agent.agent import run_agent
from config import get_config
from llm.groq_client import GroqClientError
from memory.chat_memory import ChatMemory


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

    st.set_page_config(page_title="Stage 3 AI Agent", page_icon="AI", layout="centered")
    st.title("Stage 3 AI Agent")

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

        st.header("Agent Runtime")
        st.caption(f"Max iterations: {config.max_agent_iterations}")

        if st.button("Clear memory", type="secondary"):
            st.session_state.messages = memory.clear_history()
            st.rerun()

        with st.expander("Inspect memory"):
            st.json(st.session_state.messages)

    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    user_goal = st.chat_input("Give the agent a goal")
    if not user_goal:
        return

    memory.add_message(st.session_state.messages, "user", user_goal)
    memory.save_history(st.session_state.messages)

    with st.chat_message("user"):
        st.markdown(user_goal)

    conversation_context = memory.get_recent_history(st.session_state.messages)

    with st.chat_message("assistant"):
        status_placeholder = st.empty()

        try:
            status_placeholder.info("Agent is observing the goal and deciding what to do next...")
            agent_state = run_agent(
                config=config,
                goal=user_goal,
                conversation_context=conversation_context,
                model=model,
                temperature=temperature,
                max_tokens=max_tokens,
            )

        except GroqClientError as exc:
            st.error(str(exc))
            return

        status_placeholder.empty()
        st.markdown(agent_state.final_answer)

        with st.expander("Agent Execution"):
            st.write(
                {
                    "goal": agent_state.goal,
                    "status": agent_state.status,
                    "iterations": agent_state.iteration_count,
                    "max_iterations": agent_state.max_iterations,
                }
            )
            for step in agent_state.trace:
                st.write(
                    {
                        "iteration": step.iteration,
                        "action": step.action,
                        "status": step.status,
                        "observation": step.observation,
                    }
                )

    memory.add_message(st.session_state.messages, "assistant", agent_state.final_answer)
    memory.save_history(st.session_state.messages)


if __name__ == "__main__":
    main()
