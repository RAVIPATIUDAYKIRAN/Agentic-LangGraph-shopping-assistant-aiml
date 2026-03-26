"""
tools/auth_tools.py

login_user — Authenticates via bcrypt + MySQL. Stores session in Redis.

Security note:
  On authentication failure, a single generic message is returned regardless
  of whether the email was not found or the password was wrong.
  This prevents user-enumeration attacks where an attacker probes which
  email addresses have accounts by reading different error messages.
"""

import bcrypt
from langchain_core.tools import tool
from db.db_client import execute_query
from memory.redis_memory import set_session

_AUTH_FAIL_MSG = "Invalid email or password. Please check your credentials and try again."


@tool
def login_user(email: str, password: str) -> dict:
    """
    Authenticate a user with email and password.

    Call ONLY when the user explicitly provides both email AND password.
    On success, user_id is written into LangGraph ShoppingState by tool_node_handler.

    Returns:
        {
          "success"  : bool,
          "user_id"  : int   (only on success),
          "email"    : str   (only on success),
          "full_name": str   (only on success),
          "message"  : str
        }
    """
    if not email or not password:
        return {"success": False, "message": _AUTH_FAIL_MSG}

    rows = execute_query(
        "SELECT user_id, email, full_name, password_hash "
        "FROM users WHERE email = %s",
        (email.strip().lower(),),
    )

    # Generic failure — no hint whether email or password was wrong
    if not rows:
        return {"success": False, "message": _AUTH_FAIL_MSG}

    user = rows[0]

    try:
        valid = bcrypt.checkpw(
            password.encode("utf-8"),
            user["password_hash"].encode("utf-8"),
        )
    except Exception:
        return {"success": False, "message": _AUTH_FAIL_MSG}

    if not valid:
        return {"success": False, "message": _AUTH_FAIL_MSG}

    set_session(user["user_id"], user["email"], user.get("full_name", ""))

    return {
        "success":   True,
        "user_id":   user["user_id"],
        "email":     user["email"],
        "full_name": user.get("full_name") or user["email"],
        "message":   "Login successful. Welcome back!",
    }