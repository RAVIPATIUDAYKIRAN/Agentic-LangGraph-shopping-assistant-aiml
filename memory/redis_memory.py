"""
memory/redis_memory.py

Redis-backed storage — pure key/value operations, no business logic.

Design points:
  - Singleton Redis client with auto-reconnect on ping failure.
  - add_item_to_cart uses WATCH + pipeline (optimistic locking) to prevent
    the get→modify→set race condition under concurrent requests.
  - save_order uses the same atomic pipeline pattern.
  - TTL is refreshed on every cart write (sliding 24h window).
  - All values stored as JSON strings; prices are raw floats (INR).

Key schema:
  session:{user_id}  → JSON {user_id, email, full_name, logged_in}
  cart:{user_id}     → JSON list of cart item dicts
  orders:{user_id}   → JSON list of order record dicts
"""

import os
import json
import redis
from dotenv import load_dotenv

load_dotenv()

_CART_TTL    = 86_400           # 24 hours
_SESSION_TTL = 86_400           # 24 hours
_ORDER_TTL   = 86_400 * 30      # 30 days
_MAX_RETRIES = 3                # optimistic lock retries

_client: redis.Redis | None = None


# ── Client ────────────────────────────────────────────────────────────────────

def _get_client() -> redis.Redis:
    global _client
    if _client is None or not _ping(_client):
        _client = redis.Redis(
            host                = os.getenv("REDIS_HOST", "localhost"),
            port                = int(os.getenv("REDIS_PORT", 6379)),
            db                  = int(os.getenv("REDIS_DB", 0)),
            decode_responses    = True,
            socket_connect_timeout = 5,
            socket_timeout      = 5,
            retry_on_timeout    = True,
        )
    return _client


def _ping(client: redis.Redis) -> bool:
    try:
        return client.ping()
    except Exception:
        return False


# ── Session ───────────────────────────────────────────────────────────────────

def set_session(user_id: int, email: str, full_name: str) -> None:
    _get_client().set(
        f"session:{user_id}",
        json.dumps({
            "user_id":   user_id,
            "email":     email,
            "full_name": full_name,
            "logged_in": True,
        }),
        ex=_SESSION_TTL,
    )


def get_session(user_id: int) -> dict | None:
    raw = _get_client().get(f"session:{user_id}")
    return json.loads(raw) if raw else None


def clear_session(user_id: int) -> None:
    _get_client().delete(f"session:{user_id}")


# ── Cart ──────────────────────────────────────────────────────────────────────

def get_cart(user_id: int) -> list:
    raw = _get_client().get(f"cart:{user_id}")
    return json.loads(raw) if raw else []


def save_cart(user_id: int, cart: list) -> None:
    """
    Persist cart and refresh the sliding TTL.
    """
    _get_client().set(f"cart:{user_id}", json.dumps(cart), ex=_CART_TTL)


def add_item_to_cart(user_id: int, item: dict) -> dict:
    """
    Add item to cart using WATCH + pipeline (optimistic locking).
    Retries up to _MAX_RETRIES times if a concurrent write invalidates the watch.
    Prevents duplicates by product_id.

    Returns:
        { added: bool, reason: str, title: str, cart: list }
    """
    r   = _get_client()
    key = f"cart:{user_id}"

    for _ in range(_MAX_RETRIES):
        try:
            with r.pipeline() as pipe:
                pipe.watch(key)                     # watch for concurrent changes

                raw  = pipe.get(key)
                cart = json.loads(raw) if raw else []

                already = any(
                    str(p.get("product_id")) == str(item.get("product_id"))
                    for p in cart
                )
                if already:
                    pipe.unwatch()
                    return {
                        "added":  False,
                        "reason": "duplicate",
                        "title":  item["title"],
                        "cart":   cart,
                    }

                cart.append(item)

                pipe.multi()                        # start atomic block
                pipe.set(key, json.dumps(cart), ex=_CART_TTL)
                pipe.execute()                      # commit

                return {"added": True, "title": item["title"], "cart": cart}

        except redis.WatchError:
            continue   # another writer modified key — retry

    # All retries exhausted (extremely rare)
    return {
        "added":  False,
        "reason": "concurrent_write",
        "title":  item.get("title", ""),
        "cart":   get_cart(user_id),
    }


def remove_item_from_cart(user_id: int, product_name: str) -> dict:
    """
    Remove an item by fuzzy title match.
    Uses WATCH + pipeline for safety.

    Returns:
        { removed: bool, title: str | None, query: str, cart: list }
    """
    r   = _get_client()
    key = f"cart:{user_id}"

    for _ in range(_MAX_RETRIES):
        try:
            with r.pipeline() as pipe:
                pipe.watch(key)

                raw  = pipe.get(key)
                cart = json.loads(raw) if raw else []

                lower = product_name.lower()
                match = next(
                    (p for p in cart if lower in p["title"].lower()), None
                )
                if not match:
                    pipe.unwatch()
                    return {"removed": False, "query": product_name, "cart": cart}

                cart.remove(match)

                pipe.multi()
                pipe.set(key, json.dumps(cart), ex=_CART_TTL)
                pipe.execute()

                return {"removed": True, "title": match["title"], "cart": cart}

        except redis.WatchError:
            continue

    return {"removed": False, "query": product_name, "cart": get_cart(user_id)}


def clear_cart(user_id: int) -> None:
    _get_client().delete(f"cart:{user_id}")


# ── Orders ────────────────────────────────────────────────────────────────────

def get_orders(user_id: int) -> list:
    raw = _get_client().get(f"orders:{user_id}")
    return json.loads(raw) if raw else []


def save_order(user_id: int, order: dict) -> None:
    """
    Append a new order to the user's order history using WATCH + pipeline.
    """
    r   = _get_client()
    key = f"orders:{user_id}"

    for _ in range(_MAX_RETRIES):
        try:
            with r.pipeline() as pipe:
                pipe.watch(key)

                raw    = pipe.get(key)
                orders = json.loads(raw) if raw else []
                orders.append(order)

                pipe.multi()
                pipe.set(key, json.dumps(orders), ex=_ORDER_TTL)
                pipe.execute()
                return

        except redis.WatchError:
            continue