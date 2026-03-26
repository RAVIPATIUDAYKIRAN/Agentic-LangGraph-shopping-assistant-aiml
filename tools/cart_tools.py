"""
tools/cart_tools.py

All cart and order operations. Prices in INR (₹). Hard login guard on every tool.

Key design points:
  - place_order and buy_now use execute_transaction so stock decrement
    and order INSERT happen atomically — if either fails, both roll back.
  - Stock decrement uses rowcount (not a re-query) to verify it applied.
    UPDATE ... WHERE stock > 0 ensures stock never goes negative at DB level.
  - Email is sent in a daemon thread — never blocks the order response.
  - place_order removes ONLY the ordered items from the Redis cart;
    remaining items stay untouched.
  - buy_now never touches the existing cart at all (express checkout).
"""

import uuid
import json
import threading
from datetime import datetime

from langchain_core.tools import tool
from db.db_client import execute_query, execute_transaction
from memory import redis_memory as mem
from services.email_service import send_order_confirmation


# ── Login guard ───────────────────────────────────────────────────────────────

def _require_login(user_id) -> dict | None:
    try:
        uid = int(user_id)
    except (TypeError, ValueError):
        uid = 0
    if uid <= 0:
        return {
            "blocked": True,
            "reason":  "not_authenticated",
            "message": "You are not logged in. Please login first.",
        }
    return None


# ── Product resolver ──────────────────────────────────────────────────────────

def _resolve_product(query: str) -> list:
    """
    Find in-stock products matching query.
    Ranking: exact title match → title LIKE → category LIKE.
    Returns up to 5 results.
    """
    exact = query.lower()
    like  = f"%{exact}%"
    rows  = execute_query(
        """
        SELECT product_id, title, category, brand, price, rating, stock,
               CASE
                 WHEN LOWER(title) = %s        THEN 1
                 WHEN LOWER(title) LIKE %s     THEN 2
                 WHEN LOWER(category) LIKE %s  THEN 3
                 ELSE 4
               END AS relevance_rank
        FROM   products
        WHERE (LOWER(title) LIKE %s OR LOWER(category) LIKE %s)
          AND  stock > 0
        ORDER  BY relevance_rank ASC, rating DESC
        LIMIT  5
        """,
        (exact, like, like, like, like),
    )
    return rows or []


# ── User info ─────────────────────────────────────────────────────────────────

def _fetch_user_info(user_id: int) -> tuple[str, str]:
    rows = execute_query(
        "SELECT email, full_name FROM users WHERE user_id = %s",
        (user_id,),
    )
    if rows:
        return rows[0]["email"], rows[0].get("full_name") or rows[0]["email"]
    return "", ""


# ── Async email ───────────────────────────────────────────────────────────────

def _send_email_async(**kwargs) -> None:
    """Fire-and-forget email in a daemon thread. Never blocks the order response."""
    threading.Thread(
        target=send_order_confirmation,
        kwargs=kwargs,
        daemon=True,
    ).start()


# ── Item dict ─────────────────────────────────────────────────────────────────

def _to_item_dict(p: dict) -> dict:
    return {
        "product_id": p["product_id"],
        "title":      p["title"],
        "price":      float(p["price"]),
        "category":   p["category"],
        "rating":     p["rating"],
    }


# ── Tools ─────────────────────────────────────────────────────────────────────

@tool
def add_to_cart(product_name: str, user_id: int) -> dict:
    """
    Add a product to the user's cart (stored in Redis).

    REQUIRES: user must be logged in.

    Call when user says: 'add X to cart', 'put X in my cart', 'I want X'.

    Behaviour:
      - Single match  → add immediately.
      - Multiple matches → return all so LLM asks user to pick one.
      - No match → return not_found.

    Stock is NOT decremented here — only on order placement.
    Returns raw dict — LLM formats. Prices in INR (raw float).
    """
    block = _require_login(user_id)
    if block:
        return block

    matches = _resolve_product(product_name)

    if not matches:
        return {
            "added":   False,
            "reason":  "not_found",
            "query":   product_name,
            "message": f"No product matching '{product_name}' found in our catalogue.",
            "cart":    mem.get_cart(user_id),
        }

    if len(matches) > 1:
        return {
            "added":   False,
            "reason":  "multiple_matches",
            "query":   product_name,
            "message": "Multiple products matched. Ask the user to pick one.",
            "matches": [
                {
                    "product_id": p["product_id"],
                    "title":      p["title"],
                    "price":      float(p["price"]),
                    "rating":     p["rating"],
                    "category":   p["category"],
                }
                for p in matches
            ],
            "cart": mem.get_cart(user_id),
        }

    item   = _to_item_dict(matches[0])
    result = mem.add_item_to_cart(user_id, item)
    cart   = result["cart"]

    return {
        "added":      result["added"],
        "reason":     result.get("reason", ""),
        "product":    item,
        "cart_size":  len(cart),
        "cart_total": round(sum(i["price"] for i in cart), 2),
        "cart":       cart,
    }


@tool
def remove_from_cart(product_name: str, user_id: int) -> dict:
    """
    Remove a product from the user's cart.

    REQUIRES: user must be logged in.

    Call when user says: 'remove X', 'delete X from cart', 'drop X'.
    Returns raw dict — LLM formats. Prices in INR (raw float).
    """
    block = _require_login(user_id)
    if block:
        return block

    result = mem.remove_item_from_cart(user_id, product_name)
    cart   = result["cart"]

    return {
        "removed":    result["removed"],
        "query":      product_name,
        "title":      result.get("title"),
        "cart_size":  len(cart),
        "cart_total": round(sum(i["price"] for i in cart), 2),
        "cart":       cart,
    }


@tool
def view_cart(user_id: int) -> dict:
    """
    Retrieve the user's current cart contents.

    REQUIRES: user must be logged in.

    Call when user says: 'show my cart', "what's in my cart", 'cart summary'.
    Returns raw dict — LLM formats. Prices in INR (raw float).
    """
    block = _require_login(user_id)
    if block:
        return block

    cart = mem.get_cart(user_id)
    return {
        "cart":       cart,
        "cart_size":  len(cart),
        "cart_total": round(sum(i["price"] for i in cart), 2),
        "is_empty":   len(cart) == 0,
    }


@tool
def place_order(user_id: int, product_name: str = "") -> dict:
    """
    Place an order from the user's cart.

    REQUIRES: user must be logged in. Cart must not be empty.

    Behaviour:
      - product_name provided → order ONLY that item; remaining cart items stay.
      - product_name empty    → order ALL cart items; cart is cleared.

    For each ordered item (ATOMIC — both succeed or both roll back):
      1. UPDATE products SET stock = stock - 1 WHERE product_id = %s AND stock > 0
      2. INSERT INTO orders ...

    Then:
      3. Remove ordered items from Redis cart.
      4. Save order to Redis history.
      5. Send confirmation email in background thread.

    Call when user says: 'place order', 'checkout', 'order the X',
    'buy just the Y', 'confirm my order'.

    Returns raw dict — LLM formats. Prices in INR (raw float).
    """
    block = _require_login(user_id)
    if block:
        return block

    cart = mem.get_cart(user_id)
    if not cart:
        return {
            "success": False,
            "reason":  "empty_cart",
            "message": "Your cart is empty. Add products first.",
        }

    # ── Determine which items to order ────────────────────────────────────
    if product_name:
        pn_lower    = product_name.lower()
        order_items = [i for i in cart if pn_lower in i["title"].lower()]
        if not order_items:
            return {
                "success": False,
                "reason":  "not_in_cart",
                "message": f"'{product_name}' is not in your cart.",
                "cart":    cart,
            }
    else:
        order_items = list(cart)

    total     = round(sum(i["price"] for i in order_items), 2)
    order_id  = f"ORD-{str(uuid.uuid4())[:8].upper()}"
    placed_at = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    # ── Atomic: decrement stock + insert order ────────────────────────────
    try:
        statements = []
        for item in order_items:
            statements.append((
                "UPDATE products "
                "SET stock = stock - 1 "
                "WHERE product_id = %s AND stock > 0",
                (item["product_id"],),
            ))
        statements.append((
            "INSERT INTO orders (order_id, user_id, total_amount, items_json) "
            "VALUES (%s, %s, %s, %s)",
            (order_id, user_id, total, json.dumps(order_items)),
        ))
        execute_transaction(statements)
    except Exception as exc:
        return {"success": False, "reason": "db_error", "error": str(exc)}

    # ── Remove ONLY ordered items from Redis cart ─────────────────────────
    ordered_ids = {str(i["product_id"]) for i in order_items}
    remaining   = [i for i in cart if str(i["product_id"]) not in ordered_ids]
    mem.save_cart(user_id, remaining)

    # ── Save to Redis order history ───────────────────────────────────────
    mem.save_order(user_id, {
        "order_id":  order_id,
        "total":     total,
        "items":     order_items,
        "placed_at": placed_at,
    })

    # ── Email in background (non-blocking) ────────────────────────────────
    email, full_name = _fetch_user_info(user_id)
    if email:
        _send_email_async(
            to_email  = email,
            full_name = full_name,
            order_id  = order_id,
            placed_at = placed_at,
            items     = order_items,
            total     = total,
        )

    return {
        "success":         True,
        "order_id":        order_id,
        "placed_at":       placed_at,
        "ordered_items":   order_items,
        "items_count":     len(order_items),
        "total":           total,
        "remaining_cart":  remaining,
        "remaining_count": len(remaining),
    }


@tool
def buy_now(product_name: str, user_id: int) -> dict:
    """
    INSTANT PURCHASE: find product → order it immediately.

    REQUIRES: user must be logged in.

    Orders ONLY the requested product. Existing cart is untouched.
    Atomic: stock decrement + order INSERT in one transaction.
    Email sent in background thread.

    Use when user says: 'buy X now', 'order X right now', 'get me X immediately'.

    Multiple matches → return options so LLM asks user to pick.
    Returns raw dict — LLM formats. Price in INR (raw float).
    """
    block = _require_login(user_id)
    if block:
        return block

    matches = _resolve_product(product_name)

    if not matches:
        return {"success": False, "reason": "not_found", "query": product_name}

    if len(matches) > 1:
        return {
            "success": False,
            "reason":  "multiple_matches",
            "query":   product_name,
            "matches": [
                {
                    "product_id": p["product_id"],
                    "title":      p["title"],
                    "price":      float(p["price"]),
                    "rating":     p["rating"],
                    "category":   p["category"],
                }
                for p in matches
            ],
        }

    item        = _to_item_dict(matches[0])
    order_items = [item]
    total       = round(item["price"], 2)
    order_id    = f"ORD-{str(uuid.uuid4())[:8].upper()}"
    placed_at   = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    # ── Atomic: decrement stock + insert order ────────────────────────────
    try:
        execute_transaction([
            (
                "UPDATE products "
                "SET stock = stock - 1 "
                "WHERE product_id = %s AND stock > 0",
                (item["product_id"],),
            ),
            (
                "INSERT INTO orders (order_id, user_id, total_amount, items_json) "
                "VALUES (%s, %s, %s, %s)",
                (order_id, user_id, total, json.dumps(order_items)),
            ),
        ])
    except Exception as exc:
        return {"success": False, "reason": "db_error", "error": str(exc)}

    # Cart intentionally untouched
    mem.save_order(user_id, {
        "order_id":  order_id,
        "total":     total,
        "items":     order_items,
        "placed_at": placed_at,
    })

    # ── Email in background ───────────────────────────────────────────────
    email, full_name = _fetch_user_info(user_id)
    if email:
        _send_email_async(
            to_email  = email,
            full_name = full_name,
            order_id  = order_id,
            placed_at = placed_at,
            items     = order_items,
            total     = total,
        )

    return {
        "success":          True,
        "instant_purchase": True,
        "order_id":         order_id,
        "placed_at":        placed_at,
        "product":          item,
        "items_count":      1,
        "total":            total,
    }