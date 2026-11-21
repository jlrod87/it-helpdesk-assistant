# AI-Powered IT Helpdesk Assistant

A Streamlit chatbot that answers employees' IT questions with **RAG**, **tool calling** and an **MCP server**. Claude decides which tool to use for each question.

## Architecture

```
streamlit.py (chat UI)
   └── app_implementation.py (agent: Claude Messages API + tool-use loop)
         ├── search_it_knowledge_base  RAG: it_sector.txt → Voyage AI embeddings → in-memory vector index
         ├── get_current_datetime      client tool (runs in this app)
         ├── web_search                Anthropic server tool (runs on Anthropic's side)
         └── create_support_ticket / check_ticket_status
                                       MCP tools discovered at runtime from mcp_server.py
mcp_server.py  "IT Helpdesk MCP Server" (streamable HTTP, tickets in an in-memory dict)
```

| File | Purpose |
|---|---|
| `mcp_server.py` | MCP server with `create_support_ticket` and `check_ticket_status`. Seeds tickets IT-101..IT-103 from `mcp_tool_data.json`. |
| `app_implementation.py` | RAG pipeline, client/server tools, MCP client and the agent loop. |
| `streamlit.py` | Chat interface. Shows answers and which tools were used (inputs and outputs). |
| `it_sector.txt` | IT support FAQ used as the RAG knowledge base. |

## Setup

Requires Python 3.11 or later, an Anthropic API key and a Voyage AI API key.

```bash
python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install -r requirements.txt
cp .env.example .env               # then put your real keys in .env
```

## Run

Use two terminals, both with the virtual environment active.

```bash
# Terminal 1: MCP server (http://127.0.0.1:3000/mcp)
python mcp_server.py

# Terminal 2: chatbot (http://localhost:8501)
streamlit run streamlit.py
```

Tickets are stored in memory, so new tickets are lost when the MCP server restarts.

## Sample prompts

| Category | Prompt | Expected tool |
|---|---|---|
| RAG | How do I fix network connectivity issues? | `search_it_knowledge_base` |
| RAG | My computer is running slow. What should I do? | `search_it_knowledge_base` |
| RAG | How do I configure my email? | `search_it_knowledge_base` |
| RAG | How can I protect my laptop from malware? | `search_it_knowledge_base` |
| Web search | Search for the latest Windows 11 Wi-Fi troubleshooting steps. | `web_search` |
| Web search | Find solutions for Outlook not syncing emails. | `web_search` |
| MCP | Create a ticket for my laptop running slowly. | `create_support_ticket` |
| MCP | Check the status of ticket IT-101. | `check_ticket_status` |
| Tool calling | What is the current date and time? | `get_current_datetime` |

Enter your name in the sidebar so tickets are created under it.

## Known limitations

- **No authentication.** Anyone using the app can check any ticket by its id, as the assignment requires. In production, the MCP server would take the caller's identity from an authenticated session and return only that employee's tickets.
- **Markdown images are disabled** (the `!` of every `![` becomes an HTML entity) in replies before rendering, so a prompt-injected web result cannot leak data through an auto-loading image URL.
