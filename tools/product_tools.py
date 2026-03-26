"""
tools/product_tools.py

All product-facing tools. Returns only DB data — LLM never fabricates.

Design philosophy (pure LangGraph + LLM):
  The LLM translates user intent into a clean search keyword via the system
  prompt before calling any tool. There is NO pre-processing of the query
  in Python — no stopword removal, no keyword filtering, no text manipulation.
  The query arrives from the LLM already clean. These tools simply execute
  SQL against the DB and return structured data.

  The only Python helper here is _row_to_dict (data shaping) and
  _popular_fallback (DB query for when nothing matches).

Currency: ALL prices returned as raw float (INR). LLM displays as ₹<price>.
"""

from langchain_core.tools import tool
from db.db_client import execute_query


# ── Shared helpers ────────────────────────────────────────────────────────────

def _row_to_dict(r: dict) -> dict:
    """Shape a DB row into the standard product dict returned by all tools."""
    return {
        "product_id":   r["product_id"],
        "title":        r["title"],
        "category":     r["category"],
        "brand":        r.get("brand", ""),
        "price":        float(r["price"]),   # INR — raw float
        "rating":       r["rating"],
        "rating_count": r["rating_count"],
        "stock":        r["stock"],
    }


def _popular_fallback(limit: int = 5) -> list:
    """Return top-rated in-stock products as fallback when a search has 0 results."""
    rows = execute_query(
        """
        SELECT product_id, title, category, brand,
               price, rating, rating_count, stock
        FROM   products
        WHERE  stock > 0
        ORDER  BY rating DESC, rating_count DESC
        LIMIT  %s
        """,
        (limit,),
    )
    return [_row_to_dict(r) for r in rows]


# ── Tools ─────────────────────────────────────────────────────────────────────

@tool
def search_products(query: str, top_n: int = 8) -> dict:
    """
    Search for products by keyword, category, or brand.

    The LLM must translate the user's real-world request into a product keyword
    before calling this tool. Examples from the system prompt:
      "I'm thirsty"          → query="water bottle"
      "carry my laptop"      → query="laptop bag"
      "gift for fitness"     → query="fitness"
      "rainy day outdoors"   → query="raincoat"

    SQL relevance ranking (CASE in ORDER BY):
      1 = exact title match
      2 = title LIKE %query%
      3 = category LIKE %query%
      4 = brand LIKE %query%
      5 = description LIKE %query%

    Falls back to popular products automatically when found=0.

    Returns:
        {
          "query"          : str,
          "found"          : int,
          "fallback"       : bool,
          "fallback_reason": str | None,
          "products"       : list[dict]   — prices in INR (raw float)
        }
    """
    like  = f"%{query.lower()}%"
    exact = query.lower()

    rows = execute_query(
        """
        SELECT product_id, title, category, brand,
               price, rating, rating_count, stock,
               CASE
                 WHEN LOWER(title) = %s           THEN 1
                 WHEN LOWER(title)    LIKE %s      THEN 2
                 WHEN LOWER(category) LIKE %s      THEN 3
                 WHEN LOWER(brand)    LIKE %s      THEN 4
                 ELSE 5
               END AS relevance_rank
        FROM   products
        WHERE (
                LOWER(title)       LIKE %s OR
                LOWER(category)    LIKE %s OR
                LOWER(brand)       LIKE %s OR
                LOWER(description) LIKE %s
              )
          AND  stock > 0
        ORDER  BY relevance_rank ASC, rating DESC, rating_count DESC
        LIMIT  %s
        """,
        (exact, like, like, like, like, like, like, like, top_n),
    )

    if rows:
        return {
            "query":           query,
            "found":           len(rows),
            "fallback":        False,
            "fallback_reason": None,
            "products":        [_row_to_dict(r) for r in rows],
        }

    fallback = _popular_fallback(5)
    return {
        "query":           query,
        "found":           0,
        "fallback":        True,
        "fallback_reason": (
            f"No products matched '{query}'. "
            "Showing our most popular items instead."
        ),
        "products": fallback,
    }


@tool
def filter_products(
    query: str,
    max_price: float = None,
    min_rating: float = None,
    category: str = None,
    top_n: int = 8,
) -> dict:
    """
    Search products with optional filters.

    Use when the user adds constraints:
      'under ₹500'         → max_price=500
      '4+ stars'           → min_rating=4.0
      'only electronics'   → category='electronics'

    Combines keyword search with filter conditions.
    Falls back to popular products when found=0.

    Returns:
        {
          "query"          : str,
          "filters"        : dict,
          "found"          : int,
          "fallback"       : bool,
          "fallback_reason": str | None,
          "products"       : list[dict]   — prices in INR (raw float)
        }
    """
    like  = f"%{query.lower()}%"
    exact = query.lower()

    # ── Build WHERE conditions ────────────────────────────────────────────
    conditions = ["stock > 0"]
    filter_params: list = []

    if query:
        conditions.append(
            "(LOWER(title) LIKE %s OR LOWER(category) LIKE %s OR LOWER(brand) LIKE %s)"
        )
        filter_params.extend([like, like, like])

    if max_price is not None:
        conditions.append("price <= %s")
        filter_params.append(max_price)

    if min_rating is not None:
        conditions.append("rating >= %s")
        filter_params.append(min_rating)

    if category:
        conditions.append("LOWER(category) LIKE %s")
        filter_params.append(f"%{category.lower()}%")

    # Ranking params come FIRST (they appear first in SELECT CASE)
    # then filter params, then LIMIT — order matches SQL placeholder order exactly
    all_params = tuple([exact, like, like] + filter_params + [top_n])

    sql = (
        "SELECT product_id, title, category, brand, "
        "       price, rating, rating_count, stock, "
        "       CASE "
        "         WHEN LOWER(title) = %s       THEN 1 "
        "         WHEN LOWER(title) LIKE %s    THEN 2 "
        "         WHEN LOWER(category) LIKE %s THEN 3 "
        "         ELSE 4 "
        "       END AS relevance_rank "
        f"FROM products WHERE {' AND '.join(conditions)} "
        "ORDER BY relevance_rank ASC, rating DESC "
        "LIMIT %s"
    )

    rows = execute_query(sql, all_params)

    if rows:
        return {
            "query":           query,
            "filters":         {"max_price": max_price, "min_rating": min_rating, "category": category},
            "found":           len(rows),
            "fallback":        False,
            "fallback_reason": None,
            "products":        [_row_to_dict(r) for r in rows],
        }

    fallback = _popular_fallback(5)
    return {
        "query":           query,
        "filters":         {"max_price": max_price, "min_rating": min_rating, "category": category},
        "found":           0,
        "fallback":        True,
        "fallback_reason": (
            "No products matched those filters. "
            "Try relaxing the price or rating. Here are our top picks:"
        ),
        "products": fallback,
    }


@tool
def get_product_details(product_id: int) -> dict:
    """
    Get full details for a specific product by its product_id.

    Use when user asks: 'tell me more', 'what are the specs',
    'describe this product', 'more info about it'.

    Always use the product_id from the context list in the system prompt.
    Never search again for a product already shown.

    Returns:
        {
          "found"       : bool,
          "product_id"  : int,
          "title"       : str,
          "description" : str,
          "category"    : str,
          "brand"       : str,
          "price"       : float   — INR (raw float),
          "rating"      : float,
          "rating_count": int,
          "stock"       : int
        }
    """
    rows = execute_query(
        "SELECT * FROM products WHERE product_id = %s",
        (product_id,),
    )
    if not rows:
        return {"found": False, "product_id": product_id}

    p = rows[0]
    return {
        "found":        True,
        "product_id":   p["product_id"],
        "title":        p["title"],
        "description":  p.get("description", ""),
        "category":     p["category"],
        "brand":        p.get("brand", ""),
        "price":        float(p["price"]),
        "rating":       p["rating"],
        "rating_count": p["rating_count"],
        "stock":        p["stock"],
    }


@tool
def get_reviews(query: str) -> dict:
    """
    Retrieve rating and review data for products matching a name or category.

    Use when user asks: 'how are the reviews', 'is it good quality',
    'what do people say about X', 'ratings for X'.

    Returns raw data — LLM reasons over the numbers.
    Do NOT use fixed sentiment labels in the LLM response.

    Returns:
        {
          "query"   : str,
          "found"   : int,
          "products": list[dict]   — includes rating, rating_count, description
        }
    """
    like = f"%{query.lower()}%"

    rows = execute_query(
        """
        SELECT product_id, title, category, brand,
               price, rating, rating_count, description
        FROM   products
        WHERE  LOWER(title) LIKE %s OR LOWER(category) LIKE %s
        ORDER  BY rating_count DESC, rating DESC
        LIMIT  4
        """,
        (like, like),
    )

    return {
        "query": query,
        "found": len(rows),
        "products": [
            {
                "product_id":   r["product_id"],
                "title":        r["title"],
                "category":     r["category"],
                "brand":        r.get("brand", ""),
                "price":        float(r["price"]),
                "rating":       r["rating"],
                "rating_count": r["rating_count"],
                "description":  r.get("description", ""),
            }
            for r in rows
        ],
    }


@tool
def clarify_product_query(vague_input: str) -> dict:
    """
    Use when the user's request is too vague to search meaningfully.

    Trigger examples: 'something nice', 'a gift', 'blue thing',
    'buy me something', 'I need that product'.

    Fetches REAL categories and brands from the DB so the LLM can ask
    ONE specific, data-driven follow-up question instead of guessing.

    Returns:
        {
          "vague_input"         : str,
          "available_categories": list[str],
          "available_brands"    : list[str],
          "suggested_questions" : list[str],
          "instruction"         : str
        }
    """
    categories = execute_query(
        "SELECT DISTINCT category FROM products "
        "WHERE stock > 0 ORDER BY category LIMIT 30"
    )
    brands = execute_query(
        "SELECT DISTINCT brand FROM products "
        "WHERE stock > 0 AND brand IS NOT NULL ORDER BY brand LIMIT 20"
    )

    cat_list   = [r["category"] for r in categories if r["category"]]
    brand_list = [r["brand"]    for r in brands     if r["brand"]]

    return {
        "vague_input":           vague_input,
        "available_categories":  cat_list,
        "available_brands":      brand_list,
        "suggested_questions": [
            "Which category interests you? Options: " + ", ".join(cat_list[:7]),
            "Do you have a budget in mind? (e.g. under ₹500, under ₹2,000)",
            "Any preferred brand? We carry: " + ", ".join(brand_list[:5]),
            "Is this for personal use or a gift?",
        ],
        "instruction": (
            "The user's request is too vague to search. "
            "Use available_categories and available_brands to ask "
            "ONE specific, friendly clarifying question. "
            "Do NOT search yet — wait for the user to narrow it down."
        ),
    }