"""IT Helpdesk MCP Server.

Exposes two tools over MCP (streamable HTTP transport):
  - create_support_ticket: create a new IT support ticket
  - check_ticket_status:   retrieve the status of an existing ticket

Tickets live in an in-memory Python dictionary, seeded from mcp_tool_data.json.
The server is a long-running process, so tickets survive across client calls
(they reset only when the server restarts).

Run with: python mcp_server.py   ->  http://localhost:3000/mcp
"""

import json
import os
from datetime import datetime
from pathlib import Path
from typing import Literal

import uvicorn
from mcp.server import MCPServer
from mcp.server.mcpserver.exceptions import ToolError

SEED_FILE = Path(__file__).parent / "mcp_tool_data.json"
HOST = os.getenv("MCP_HOST", "127.0.0.1")
PORT = int(os.getenv("MCP_PORT", "3000"))
TICKET_PREFIX = "IT-"
PLACEHOLDER_NAMES = {"unknown", "n/a", "na", "none", "employee", "user", "anonymous",
                     "no registrado", "desconocido", "usuario", "anónimo", "anonimo"}

Priority = Literal["Low", "Medium", "High"]

mcp_server = MCPServer(name="IT Helpdesk MCP Server", version="1.0.0")


def load_seed_tickets() -> dict[str, dict]:
    """Load the sample tickets so IT-101..IT-103 exist from the start."""
    tickets = json.loads(SEED_FILE.read_text(encoding="utf-8"))
    return {ticket["ticket_id"]: ticket for ticket in tickets}


# In-memory "database": ticket_id -> ticket data
TICKETS: dict[str, dict] = load_seed_tickets()


def next_ticket_id() -> str:
    """Generate the next sequential id, e.g. IT-104."""
    numbers = [int(ticket_id.removeprefix(TICKET_PREFIX)) for ticket_id in TICKETS]
    return f"{TICKET_PREFIX}{max(numbers, default=100) + 1}"


@mcp_server.tool(
    name="create_support_ticket",
    description=(
        "Create a new IT support ticket for an employee's issue. "
        "Use when the user asks to open, raise, log or create a ticket. "
        "Returns the created ticket, including its new ticket_id."
    ),
)
def create_support_ticket(
    employee: str,
    issue: str,
    description: str,
    priority: Priority = "Medium",
) -> dict:
    # Business rules live in the server: the prompt guides the model, the server enforces.
    employee, issue = employee.strip(), issue.strip()
    if not employee or employee.lower() in PLACEHOLDER_NAMES:
        raise ToolError("The employee's real name is required. Ask the user for it.")
    if not issue:
        raise ToolError("A short description of the issue is required.")
    ticket = {
        "ticket_id": next_ticket_id(),
        "employee": employee,
        "issue": issue,
        "description": description.strip(),
        "priority": priority,
        "status": "Open",
        "created_at": datetime.now().isoformat(timespec="seconds"),
    }
    TICKETS[ticket["ticket_id"]] = ticket
    return ticket


@mcp_server.tool(
    name="check_ticket_status",
    description=(
        "Retrieve the status and details of an existing IT support ticket "
        "by its id (format IT-<number>, e.g. IT-101)."
    ),
)
def check_ticket_status(ticket_id: str) -> dict:
    ticket = TICKETS.get(ticket_id.strip().upper())
    if ticket is None:
        # ToolError = an anticipated failure: its message reaches the model
        # (is_error=True). Any other exception is treated as a crash and its
        # details are hidden from the client.
        raise ToolError(f"Ticket {ticket_id} not found.")
    return ticket


if __name__ == "__main__":
    print(f"IT Helpdesk MCP Server running at http://{HOST}:{PORT}/mcp")
    uvicorn.run(mcp_server.streamable_http_app(host=HOST), host=HOST, port=PORT)
