"""
app.py — Streamlit UI for the AI Shopping Assistant

Pure LangGraph + Mistral. No manual routing. No intent parsing.
Every user message → graph → LLM reasons → tools execute → LLM formats → user sees result.

Session fields:
  user_id, is_logged_in, user_email  — auth state
  history                            — clean conversation history (Human + AI only)
  last_products                      — context memory for "this", "add it", etc.

All updated_session reads use .get() with safe defaults so a missing
key never causes a KeyError.
"""

import streamlit as st
from graph import run_graph
from memory.redis_memory import get_cart

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="🛒 AI Shopping Assistant",
    page_icon="🛒",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown("""
<style>
    .main { padding-top: 1rem; }
    .stChatMessage { border-radius: 12px; margin-bottom: 8px; }
    .cart-item {
        padding: 6px 0;
        border-bottom: 1px solid #f0f0f0;
        font-size: 0.88rem;
        line-height: 1.5;
    }
    .cart-total {
        font-weight: 700;
        color: #2e7d32;
        font-size: 1rem;
        margin-top: 10px;
    }
</style>
""", unsafe_allow_html=True)


# ── Session state ─────────────────────────────────────────────────────────────
_DEFAULTS = {
    "messages":      [],
    "user_id":       None,
    "is_logged_in":  False,
    "user_email":    None,
    "history":       [],
    "last_products": [],
}

for _k, _v in _DEFAULTS.items():
    if _k not in st.session_state:
        st.session_state[_k] = _v


# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## 🛒 Shopping Assistant")
    st.markdown("---")

    if st.session_state["is_logged_in"]:
        st.success(f"👤 {st.session_state['user_email']}")
        if st.button("🚪 Logout", use_container_width=True):
            for _k, _v in _DEFAULTS.items():
                st.session_state[_k] = _v
            st.rerun()
    else:
        st.warning("🔒 Not logged in")
        st.caption("Share your email and password to get started.")

    st.markdown("---")
    st.markdown("### 🛍️ Your Cart")

    if st.session_state["is_logged_in"] and st.session_state["user_id"]:
        cart = get_cart(st.session_state["user_id"])
        if cart:
            for item in cart:
                title = item["title"][:38] + "…" if len(item["title"]) > 38 else item["title"]
                st.markdown(
                    f"<div class='cart-item'>📦 {title}<br>"
                    f"<strong>₹{item['price']}</strong>"
                    f" &nbsp; ⭐ {item.get('rating', '-')}</div>",
                    unsafe_allow_html=True,
                )
            total = round(sum(i["price"] for i in cart), 2)
            count = len(cart)
            st.markdown(
                f"<div class='cart-total'>"
                f"💰 Total: ₹{total} "
                f"({count} item{'s' if count > 1 else ''})"
                f"</div>",
                unsafe_allow_html=True,
            )
        else:
            st.caption("Cart is empty")
    else:
        st.caption("Login to see your cart")

    st.markdown("---")
    with st.expander("💡 Try asking..."):
        st.markdown("""
**Real-world needs:**
- *I'm thirsty*
- *I need to carry my laptop*
- *Something for a rainy day*
- *Gift idea for a fitness lover*

**Vague → Clarification:**
- *I need something nice under ₹500*
- *Buy me a gift*

**Search & Filter:**
- *Show wireless earbuds*
- *Laptops under ₹40,000 with 4+ stars*

**Details & Reviews:**
- *Tell me more about this*
- *How are the reviews for running shoes?*

**Cart:**
- *Add this to my cart*
- *Remove water bottle from cart*
- *Show my cart*

**Orders:**
- *Place order for the laptop bag*
- *Order all items in my cart*
- *Buy yoga mat right now* ⚡

**Login:**
- *your@email.com  YourPassword*
        """)


# ── Main chat ─────────────────────────────────────────────────────────────────
st.markdown("# 🛒 AI Shopping Assistant")
st.caption(
    "Powered by Mistral AI · LangGraph · Redis · MySQL · "
    "Pure agentic tool-calling · 🇮🇳 Built for India"
)

for msg in st.session_state.messages:
    avatar = "🧑" if msg["role"] == "user" else "🤖"
    with st.chat_message(msg["role"], avatar=avatar):
        st.markdown(msg["content"])

user_input = st.chat_input("Ask me anything…")

if user_input:
    st.session_state.messages.append({"role": "user", "content": user_input})
    with st.chat_message("user", avatar="🧑"):
        st.markdown(user_input)

    with st.chat_message("assistant", avatar="🤖"):
        with st.spinner("🧠 Thinking…"):
            try:
                session = {
                    "user_id":       st.session_state["user_id"],
                    "is_logged_in":  st.session_state["is_logged_in"],
                    "user_email":    st.session_state["user_email"],
                    "history":       st.session_state["history"],
                    "last_products": st.session_state["last_products"],
                }

                response, updated_session = run_graph(user_input, session)

                # Safe .get() with defaults — never throws KeyError
                st.session_state["user_id"]       = updated_session.get("user_id")
                st.session_state["is_logged_in"]  = updated_session.get("is_logged_in", False)
                st.session_state["user_email"]    = updated_session.get("user_email")
                st.session_state["history"]       = updated_session.get("history", [])
                st.session_state["last_products"] = updated_session.get("last_products", [])

            except Exception as exc:
                response = (
                    f"⚠️ Something went wrong: `{str(exc)}`\n\n"
                    "Please check your API keys, Redis, and MySQL connections."
                )

            st.markdown(response)

    st.session_state.messages.append({"role": "assistant", "content": response})
    st.rerun()