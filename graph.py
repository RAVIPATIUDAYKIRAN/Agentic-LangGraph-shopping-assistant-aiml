"""
graph.py — LangGraph + Mistral AI Shopping Assistant

Graph topology (pure state-driven, zero intent routing):

  ┌─────────────┐
  │  auth_gate  │  ← Pure Python node. Zero LLM cost.
  └──────┬──────┘    Reads is_logged_in → sets routing_decision.
         │
    ┌────┴─────┐
    ▼          ▼
login_llm  shopping_llm  ← Two LLM nodes, different tool bindings.
    │          │
    └────┬─────┘
         ▼
       tools    ← Executes tool calls. Syncs auth. Updates last_products.
         │
    ┌────┴─────┐
    ▼          ▼
login_llm  shopping_llm  ← route_from_tools picks based on is_logged_in.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
ROOT CAUSE OF PREVIOUS RECURSION BUG (now fixed):

1. _normalize_tool_calls patched response.tool_calls but left
   additional_kwargs["tool_calls"] intact. LangGraph's ToolNode reads
   from tool_calls attribute; the router read from the same attribute.
   After tools ran, the LLM message still showed tool_calls in
   additional_kwargs — causing the router to think another tool call
   was pending even after tools had already run.

   FIX: after normalising, CLEAR additional_kwargs["tool_calls"] so
   the two sources never disagree.

2. route_from_tools had NO terminal condition. After tools ran it
   always routed back to an LLM node — even when the LLM had just
   produced a final text response. The LLM then saw the full message
   history including prior tool results and called the same tool again.

   FIX: route_from_tools checks the LAST message in state. If that
   last message is a ToolMessage (tools just ran, no LLM response yet)
   → route to LLM. If the last message is an AIMessage with no
   tool_calls (LLM already gave final answer) → END immediately.
   This means after tools execute the LLM gets exactly ONE more turn,
   and if that turn produces no new tool calls → END.

3. recursion_limit raised to 25 — complex multi-tool flows
   (login → search → add to cart → place order) legitimately need
   more steps than 12. 25 is a safe ceiling for any real user journey
   while still catching true infinite loops.
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
"""

import os
import json
import re
from typing import Annotated, Sequence
from typing_extensions import TypedDict

from dotenv import load_dotenv
from langchain_core.messages import (
    AIMessage,
    BaseMessage,
    HumanMessage,
    SystemMessage,
    ToolMessage,
)
from langchain_mistralai import ChatMistralAI
from langgraph.graph import StateGraph, END
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode

from tools.auth_tools import login_user
from tools.product_tools import (
    search_products,
    filter_products,
    get_reviews,
    get_product_details,
    clarify_product_query,
)
from tools.cart_tools import (
    add_to_cart,
    remove_from_cart,
    view_cart,
    place_order,
    buy_now,
)

load_dotenv()

# ── Tool sets ─────────────────────────────────────────────────────────────────
LOGIN_TOOLS = [login_user]

SHOPPING_TOOLS = [
    search_products,
    filter_products,
    get_reviews,
    get_product_details,
    clarify_product_query,
    add_to_cart,
    remove_from_cart,
    view_cart,
    place_order,
    buy_now,
]

ALL_TOOLS = LOGIN_TOOLS + SHOPPING_TOOLS

# Singleton ToolNode — created once at module load
_tool_node = ToolNode(ALL_TOOLS)

# ── LLM instances ─────────────────────────────────────────────────────────────
_base_llm = ChatMistralAI(
    model="mistral-large-latest",
    api_key=os.getenv("MISTRAL_API_KEY"),
    temperature=0.3,
)
login_llm    = _base_llm.bind_tools(LOGIN_TOOLS)
shopping_llm = _base_llm.bind_tools(SHOPPING_TOOLS)


# ── State ─────────────────────────────────────────────────────────────────────
class ShoppingState(TypedDict):
    messages:          Annotated[Sequence[BaseMessage], add_messages]
    user_id:           int | None
    is_logged_in:      bool
    user_email:        str | None
    routing_decision:  str    # "needs_login" | "go_shopping"
    last_products:     list   # context memory — products shown this session


# ── Helpers ───────────────────────────────────────────────────────────────────

def _is_login_attempt(text: str) -> bool:
    """
    Returns True when the message contains an email address plus something
    that looks like a password — regardless of exact phrasing.
    """
    has_email = bool(
        re.search(r"[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}", text)
    )
    if not has_email:
        return False
    t = text.lower()
    if any(kw in t for kw in ["password", "pass ", "pass:", "pwd", " and "]):
        return True
    leftover = re.sub(
        r"[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}", "", text
    ).strip()
    return len(leftover) >= 4


def _normalize_tool_calls(response: AIMessage) -> AIMessage:
    """
    Normalize Mistral's tool_calls from additional_kwargs into the
    response.tool_calls list that LangGraph ToolNode reads.

    CRITICAL FIX: After writing normalized tool_calls to response.tool_calls,
    we CLEAR additional_kwargs["tool_calls"]. Without this, both attributes
    exist simultaneously. The router reads response.tool_calls (correct),
    but ToolNode internally also checks additional_kwargs — causing it to
    see "pending tool calls" even after they've already executed, which
    loops the graph back into the tools node indefinitely.
    """
    raw = getattr(response, "additional_kwargs", {}).get("tool_calls")
    if not raw:
        return response

    normalized = []
    for tc in raw:
        args = tc["function"]["arguments"]
        if isinstance(args, str):
            try:
                args = json.loads(args)
            except Exception:
                args = {}
        normalized.append({
            "id":   tc["id"],
            "name": tc["function"]["name"],
            "args": args,
        })

    response.tool_calls = normalized

    # Clear the raw source so it never conflicts with the normalized version
    if hasattr(response, "additional_kwargs") and "tool_calls" in response.additional_kwargs:
        response.additional_kwargs.pop("tool_calls", None)

    return response


def _has_pending_tool_calls(state: ShoppingState) -> bool:
    """
    Returns True only if the LAST message in the state is an AIMessage
    with at least one tool_call that has not yet been executed.

    This is the single authoritative check used by ALL routers.
    It prevents routing to tools when the last message is already a
    ToolMessage (tools just finished) or a final AIMessage with no calls.
    """
    messages = list(state["messages"])
    if not messages:
        return False
    last = messages[-1]
    # ToolMessage means tools just ran — no more tool calls pending
    if isinstance(last, ToolMessage):
        return False
    # AIMessage with tool_calls list that is non-empty
    if isinstance(last, AIMessage):
        tc = getattr(last, "tool_calls", None)
        return bool(tc)
    return False


def _extract_products_from_messages(messages: list) -> list:
    """
    Scan ToolMessages for product data.
    Returns the LAST non-empty product list found.
    """
    product_tools = {
        "search_products", "filter_products", "get_reviews",
        "get_product_details", "add_to_cart", "buy_now",
    }
    last_products = []
    for msg in messages:
        if not isinstance(msg, ToolMessage) or msg.name not in product_tools:
            continue
        try:
            data = json.loads(msg.content) if isinstance(msg.content, str) else msg.content
            if not isinstance(data, dict):
                continue
            if "products" in data and isinstance(data["products"], list) and data["products"]:
                last_products = data["products"]
            elif data.get("found") and "product_id" in data:
                last_products = [data]
        except Exception:
            pass
    return last_products


def _format_last_products_context(last_products: list) -> str:
    if not last_products:
        return "  (none — no products shown yet)"
    lines = []
    for i, p in enumerate(last_products, 1):
        lines.append(
            f"  [{i}] product_id={p.get('product_id')}  |  "
            f"{p.get('title', 'Unknown')}  |  "
            f"₹{p.get('price', '-')}  |  "
            f"⭐{p.get('rating', '-')}  |  "
            f"brand: {p.get('brand', '-')}  |  "
            f"stock: {p.get('stock', '-')}"
        )
    return "\n".join(lines)


# ── System prompts ────────────────────────────────────────────────────────────

def _login_system_prompt() -> str:
    return """You are the authentication assistant for an Indian shopping platform.

YOUR ONLY JOB: Help the user log in.

RULES:
1. If the user message contains an email address AND a password (ANY format) →
   call login_user(email=..., password=...) immediately. Extract from any format:
     "x@y.com and Uday@!421"   → email="x@y.com"  password="Uday@!421"
     "email x@y.com pass abc"  → email="x@y.com"  password="abc"
     "x@y.com myPassword"      → email="x@y.com"  password="myPassword"

2. AFTER calling login_user — DO NOT call any other tool. STOP.
   Just respond with a warm welcome message. The system will handle the rest.

3. If credentials are missing → respond warmly and ask for them.
   Acknowledge what the user wanted, explain login is needed, show an example.

4. Do NOT discuss products, cart, or orders.

Warm response format when no credentials:
  "Of course, I'd love to help! 😊
   Please log in first so I can keep your cart and orders safe.
   👉  your@email.com  YourPassword
   Once you're in, I'll take care of everything!"
"""


def _shopping_system_prompt(state: ShoppingState) -> str:
    products_block = _format_last_products_context(state.get("last_products") or [])
    uid = state["user_id"]
    return f"""You are a warm, knowledgeable AI shopping assistant for an Indian e-commerce platform.

AUTHENTICATED SESSION:
  user_id = {uid}
  email   = {state['user_email']}

PRODUCTS SHOWN IN THIS CONVERSATION (context memory):
{products_block}

Always pass user_id={uid} to ALL cart and order tools.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
RULES
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

RULE 1 — CONTEXT MEMORY:
  When user says "this", "that", "it", "the first one", "add it",
  "tell me more", "details", "specs":
  • Use product [1] from PRODUCTS SHOWN above (or the one named/numbered).
  • NEVER search again for a product already in the context list.
  • "add this"   → add_to_cart(product_name="<title from list>", user_id={uid})
  • "tell me more" → get_product_details(product_id=<id from list>)
  • "buy this"   → buy_now(product_name="<title from list>", user_id={uid})
  • NEVER say a product is unavailable if it is in the context list above.

RULE 2 — REAL-WORLD NEED TRANSLATION:
  Translate user descriptions into clean product keywords before searching.
  • "I'm thirsty"           → search_products(query="water bottle")
  • "carry my laptop"       → search_products(query="laptop bag")
  • "sunny beach day"       → search_products(query="sunscreen")
  • "rainy day"             → search_products(query="raincoat")
  • "gift for fitness lover"→ search_products(query="fitness")

RULE 3 — VAGUE QUERIES:
  If too vague ("something nice", "a gift", "blue thing"):
  • Call clarify_product_query. Ask ONE specific follow-up.
  • Do NOT search until user gives a clear keyword.

RULE 4 — INSTANT PURCHASE:
  "buy X now", "order X right now" → buy_now(product_name=..., user_id={uid}).

RULE 5 — ONE TOOL PER TURN:
  Call only the tools needed for THIS specific request.
  Do NOT call multiple search tools in sequence unless explicitly asked.
  After getting tool results, STOP calling tools and format the response.

RULE 6 — STRICT PRODUCT DATA:
  ONLY show products from tool results — NEVER fabricate names, prices, ratings.
  If fallback=True → say no exact match found, show alternatives.

RULE 7 — CURRENCY:
  ALL prices as ₹<amount>. NEVER use $ or USD.

RULE 8 — CART & ORDERS:
  Pass user_id={uid} to all cart/order tools.
  After order → show Order ID and ₹ total.

RULE 9 — RESPONSE FORMAT:
  Warm, concise. Product list: name · ₹price · ⭐rating · one-line highlight.
"""


# ── Nodes ─────────────────────────────────────────────────────────────────────

def auth_gate_node(state: ShoppingState) -> dict:
    """Pure Python gate — zero LLM cost. Sets routing_decision only."""
    if state.get("is_logged_in") and state.get("user_id"):
        return {"routing_decision": "go_shopping"}
    last_human = next(
        (m for m in reversed(list(state["messages"])) if isinstance(m, HumanMessage)),
        None,
    )
    if last_human and _is_login_attempt(last_human.content):
        return {"routing_decision": "needs_login"}
    return {"routing_decision": "needs_login"}


def login_llm_node(state: ShoppingState) -> dict:
    """LLM with ONLY login_user bound."""
    system_msg = SystemMessage(content=_login_system_prompt())
    messages   = [system_msg] + list(state["messages"])
    response   = login_llm.invoke(messages)
    return {"messages": [_normalize_tool_calls(response)]}


def shopping_llm_node(state: ShoppingState) -> dict:
    """LLM with all shopping tools. login_user excluded."""
    system_msg = SystemMessage(content=_shopping_system_prompt(state))
    messages   = [system_msg] + list(state["messages"])
    response   = shopping_llm.invoke(messages)
    return {"messages": [_normalize_tool_calls(response)]}


def tool_node_handler(state: ShoppingState) -> dict:
    """Execute tool calls. Preserve all state. Sync auth + last_products."""
    result = _tool_node.invoke(state)

    # Start from a full copy of existing state — nothing gets silently reset
    updated_state = {
        "user_id":          state.get("user_id"),
        "is_logged_in":     state.get("is_logged_in", False),
        "user_email":       state.get("user_email"),
        "routing_decision": state.get("routing_decision", ""),
        "last_products":    state.get("last_products") or [],
        "messages":         result.get("messages", []),
    }

    new_messages = result.get("messages", [])

    # ── Sync login state ──────────────────────────────────────────────────
    for msg in new_messages:
        if not isinstance(msg, ToolMessage) or msg.name != "login_user":
            continue
        try:
            data = json.loads(msg.content) if isinstance(msg.content, str) else msg.content
            if isinstance(data, dict) and data.get("success"):
                updated_state["user_id"]      = data["user_id"]
                updated_state["is_logged_in"] = True
                updated_state["user_email"]   = data["email"]
        except (json.JSONDecodeError, TypeError):
            try:
                from db.db_client import execute_query
                last_ai = next(
                    (m for m in reversed(list(state["messages"]))
                     if not isinstance(m, (HumanMessage, ToolMessage))),
                    None,
                )
                if last_ai:
                    for tc in getattr(last_ai, "tool_calls", []):
                        if tc["name"] == "login_user":
                            email = tc["args"].get("email", "").strip().lower()
                            rows  = execute_query(
                                "SELECT user_id, email FROM users WHERE email = %s",
                                (email,),
                            )
                            if rows:
                                updated_state["user_id"]      = rows[0]["user_id"]
                                updated_state["is_logged_in"] = True
                                updated_state["user_email"]   = rows[0]["email"]
            except Exception:
                pass

    # ── Update last_products ──────────────────────────────────────────────
    fresh = _extract_products_from_messages(new_messages)
    if fresh:
        updated_state["last_products"] = fresh

    return updated_state


# ── Routers ───────────────────────────────────────────────────────────────────

def route_from_gate(state: ShoppingState) -> str:
    return state.get("routing_decision", "needs_login")


def route_from_login_llm(state: ShoppingState) -> str:
    """After login_llm: go to tools only if there are pending tool calls."""
    if _has_pending_tool_calls(state):
        return "tools"
    return END


def route_from_shopping_llm(state: ShoppingState) -> str:
    """After shopping_llm: go to tools only if there are pending tool calls."""
    if _has_pending_tool_calls(state):
        return "tools"
    return END


def route_from_tools(state: ShoppingState) -> str:
    """
    After tools execute:
    - Route to the appropriate LLM so it can format the tool results.
    - The LLM will then either call more tools (→ tools again) or
      produce a final text response (→ END via route_from_*_llm).

    This router ONLY decides WHICH LLM to send to.
    The terminal END decision is always made by route_from_login_llm
    or route_from_shopping_llm AFTER the LLM responds.
    """
    if state.get("is_logged_in") and state.get("user_id"):
        return "shopping_llm"
    return "login_llm"


# ── Graph ─────────────────────────────────────────────────────────────────────

def build_graph():
    graph = StateGraph(ShoppingState)

    graph.add_node("auth_gate",    auth_gate_node)
    graph.add_node("login_llm",    login_llm_node)
    graph.add_node("shopping_llm", shopping_llm_node)
    graph.add_node("tools",        tool_node_handler)

    graph.set_entry_point("auth_gate")

    graph.add_conditional_edges(
        "auth_gate", route_from_gate,
        {"needs_login": "login_llm", "go_shopping": "shopping_llm"},
    )
    graph.add_conditional_edges(
        "login_llm", route_from_login_llm,
        {"tools": "tools", END: END},
    )
    graph.add_conditional_edges(
        "shopping_llm", route_from_shopping_llm,
        {"tools": "tools", END: END},
    )
    graph.add_conditional_edges(
        "tools", route_from_tools,
        {"shopping_llm": "shopping_llm", "login_llm": "login_llm"},
    )

    return graph.compile()


_graph = None


def get_graph():
    global _graph
    if _graph is None:
        _graph = build_graph()
    return _graph


# ── Public runner ─────────────────────────────────────────────────────────────

def run_graph(user_message: str, session_state: dict):
    """
    Entry point called from Streamlit on every user message.
    Returns (response_text, updated_session_dict).
    """
    graph   = get_graph()
    history = session_state.get("history") or []

    initial_state = ShoppingState(
        messages         = history + [HumanMessage(content=user_message)],
        user_id          = session_state.get("user_id"),
        is_logged_in     = session_state.get("is_logged_in", False),
        user_email       = session_state.get("user_email"),
        routing_decision = "",
        last_products    = session_state.get("last_products") or [],
    )

    final_state = graph.invoke(
        initial_state,
        config={"recursion_limit": 25},
    )

    # ── Final text response ───────────────────────────────────────────────
    ai_messages = [
        m for m in final_state["messages"]
        if not isinstance(m, (HumanMessage, ToolMessage))
        and not getattr(m, "tool_calls", None)
    ]
    response = (
        ai_messages[-1].content
        if ai_messages
        else "Something went wrong. Please try again."
    )

    # ── Persist last_products for next turn ───────────────────────────────
    tool_msgs      = [m for m in final_state["messages"] if isinstance(m, ToolMessage)]
    fresh_products = _extract_products_from_messages(tool_msgs)
    last_products  = fresh_products if fresh_products else (
        final_state.get("last_products") or []
    )

    # ── Clean history ─────────────────────────────────────────────────────
    clean_history = [
        m for m in final_state["messages"]
        if not isinstance(m, ToolMessage)
        and not getattr(m, "tool_calls", None)
    ]

    updated_session = {
        "user_id":       final_state.get("user_id"),
        "is_logged_in":  final_state.get("is_logged_in", False),
        "user_email":    final_state.get("user_email"),
        "history":       clean_history[-20:],
        "last_products": last_products,
    }

    return response, updated_session