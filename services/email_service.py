"""
services/email_service.py

Sends order-confirmation emails with a warm, branded HTML template.

Design points:
  - All user-derived data (name, product titles, order ID) is passed through
    html.escape() before being embedded in the HTML template.
    This prevents HTML injection if a product title or username ever contains
    characters like <, >, &, ", '.
  - Plain-text fallback attached alongside HTML.
  - Caller is responsible for non-blocking invocation (daemon thread in cart_tools).
  - Returns bool — True on success, False on failure.
  - Skips silently if EMAIL_SENDER / EMAIL_APP_PASSWORD not set in .env.

Currency: all prices displayed as ₹<amount> (INR).
"""

import os
import html
import smtplib
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from dotenv import load_dotenv

load_dotenv()

SENDER_EMAIL    = os.getenv("EMAIL_SENDER", "")
SENDER_PASSWORD = os.getenv("EMAIL_APP_PASSWORD", "")


# ── Public API ────────────────────────────────────────────────────────────────

def send_order_confirmation(
    to_email:  str,
    full_name: str,
    order_id:  str,
    placed_at: str,
    items:     list,
    total:     float,
) -> bool:
    """
    Send a warm order-confirmation email.

    Args:
        to_email  : customer's email address
        full_name : customer's display name
        order_id  : e.g. "ORD-A1B2C3D4"
        placed_at : datetime string
        items     : list of cart dicts [{title, price, ...}]
        total     : order total in INR

    Returns True on success, False on any failure.
    Called in a daemon thread from cart_tools — never blocks the order.
    """
    if not SENDER_EMAIL or not SENDER_PASSWORD:
        print("[EMAIL] Skipped — EMAIL_SENDER or EMAIL_APP_PASSWORD not set in .env")
        return False

    try:
        msg = _build_message(to_email, full_name, order_id, placed_at, items, total)
        with smtplib.SMTP_SSL("smtp.gmail.com", 465, timeout=10) as server:
            server.login(SENDER_EMAIL, SENDER_PASSWORD)
            server.send_message(msg)
        print(f"[EMAIL] Sent to {to_email} — {order_id}")
        return True
    except Exception as exc:
        print(f"[EMAIL] Failed for {to_email}: {exc}")
        return False


# ── Builders ──────────────────────────────────────────────────────────────────

def _build_message(
    to_email:  str,
    full_name: str,
    order_id:  str,
    placed_at: str,
    items:     list,
    total:     float,
) -> MIMEMultipart:
    msg = MIMEMultipart("alternative")
    msg["Subject"] = f"🛒 Order Confirmed – {order_id} | AI Shopping Assistant"
    msg["From"]    = f"AI Shopping Assistant <{SENDER_EMAIL}>"
    msg["To"]      = to_email

    msg.attach(MIMEText(_plain_text(full_name, order_id, placed_at, items, total), "plain"))
    msg.attach(MIMEText(_html_body (full_name, order_id, placed_at, items, total), "html"))
    return msg


def _plain_text(
    full_name: str,
    order_id:  str,
    placed_at: str,
    items:     list,
    total:     float,
) -> str:
    lines = [
        f"Hi {full_name},",
        "",
        "Your order has been confirmed!",
        f"Order ID : {order_id}",
        f"Placed at: {placed_at}",
        "",
        "Items ordered:",
    ]
    for item in items:
        lines.append(f"  • {item['title']}  —  ₹{item['price']}")
    lines += [
        "",
        f"Total: ₹{total}",
        "",
        "We'll notify you once your items are shipped.",
        "",
        "Warm regards,",
        "AI Shopping Assistant",
    ]
    return "\n".join(lines)


def _html_body(
    full_name: str,
    order_id:  str,
    placed_at: str,
    items:     list,
    total:     float,
) -> str:
    # ── Escape ALL user-derived strings before inserting into HTML ────────
    safe_name     = html.escape(full_name)
    safe_order_id = html.escape(order_id)
    safe_placed   = html.escape(placed_at)
    safe_total    = html.escape(str(total))

    rows_html = ""
    for item in items:
        safe_title = html.escape(str(item.get("title", "")))
        safe_price = html.escape(str(item.get("price", "")))
        rows_html += f"""
        <tr>
          <td style="padding:10px 8px;border-bottom:1px solid #f0f0f0;">
            <span style="font-size:15px;">📦</span>
            <strong style="margin-left:6px;">{safe_title}</strong>
          </td>
          <td style="padding:10px 8px;border-bottom:1px solid #f0f0f0;
                     text-align:right;color:#2e7d32;font-weight:600;">
            ₹{safe_price}
          </td>
        </tr>"""

    return f"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8"/>
  <meta name="viewport" content="width=device-width,initial-scale=1.0"/>
  <title>Order Confirmed</title>
</head>
<body style="margin:0;padding:0;background:#f0f4f8;
             font-family:'Segoe UI',Arial,sans-serif;">

  <table width="100%" cellpadding="0" cellspacing="0"
         style="background:#f0f4f8;padding:30px 0;">
    <tr><td align="center">

      <table width="600" cellpadding="0" cellspacing="0"
             style="background:#fff;border-radius:16px;overflow:hidden;
                    box-shadow:0 4px 24px rgba(0,0,0,0.08);">

        <!-- Header -->
        <tr>
          <td style="background:linear-gradient(135deg,#1b5e20 0%,#43a047 100%);
                     padding:36px 40px;text-align:center;">
            <h1 style="margin:0;color:#fff;font-size:28px;letter-spacing:-0.5px;">
              🎉 Order Confirmed!
            </h1>
            <p style="margin:8px 0 0;color:#c8e6c9;font-size:15px;">
              Thank you for shopping with us
            </p>
          </td>
        </tr>

        <!-- Greeting -->
        <tr>
          <td style="padding:32px 40px 0;">
            <p style="margin:0;font-size:17px;color:#333;">
              Hi <strong>{safe_name}</strong> 👋,
            </p>
            <p style="margin:12px 0 0;font-size:15px;color:#555;line-height:1.6;">
              Your order has been successfully placed and is being processed.
              Here's a summary of what you ordered:
            </p>
          </td>
        </tr>

        <!-- Order Meta -->
        <tr>
          <td style="padding:20px 40px;">
            <table width="100%" cellpadding="0" cellspacing="0"
                   style="background:#f8fdf8;border-radius:10px;
                          border:1px solid #e8f5e9;padding:16px;">
              <tr>
                <td style="padding:6px 12px;font-size:14px;color:#555;">
                  🧾 <strong>Order ID</strong>
                </td>
                <td style="padding:6px 12px;font-size:14px;color:#1b5e20;
                           font-weight:700;text-align:right;">
                  {safe_order_id}
                </td>
              </tr>
              <tr>
                <td style="padding:6px 12px;font-size:14px;color:#555;">
                  🕒 <strong>Placed At</strong>
                </td>
                <td style="padding:6px 12px;font-size:14px;
                           color:#333;text-align:right;">
                  {safe_placed}
                </td>
              </tr>
            </table>
          </td>
        </tr>

        <!-- Items -->
        <tr>
          <td style="padding:0 40px;">
            <p style="margin:0 0 10px;font-size:15px;font-weight:700;color:#333;">
              🛍️ Items Ordered
            </p>
            <table width="100%" cellpadding="0" cellspacing="0"
                   style="border:1px solid #e0e0e0;border-radius:10px;overflow:hidden;">
              <thead>
                <tr style="background:#f5f5f5;">
                  <th style="padding:10px 8px;text-align:left;
                             font-size:13px;color:#666;font-weight:600;">Product</th>
                  <th style="padding:10px 8px;text-align:right;
                             font-size:13px;color:#666;font-weight:600;">Price</th>
                </tr>
              </thead>
              <tbody>{rows_html}</tbody>
            </table>
          </td>
        </tr>

        <!-- Total -->
        <tr>
          <td style="padding:20px 40px;">
            <table width="100%" cellpadding="0" cellspacing="0"
                   style="background:linear-gradient(135deg,#e8f5e9,#f1f8e9);
                          border-radius:10px;padding:16px;">
              <tr>
                <td style="font-size:18px;font-weight:700;color:#1b5e20;">
                  💰 Total Amount
                </td>
                <td style="font-size:22px;font-weight:800;
                           color:#1b5e20;text-align:right;">
                  ₹{safe_total}
                </td>
              </tr>
            </table>
          </td>
        </tr>

        <!-- Shipping notice -->
        <tr>
          <td style="padding:0 40px 24px;">
            <p style="margin:0;padding:14px 16px;background:#fff8e1;
                      border-left:4px solid #ffc107;border-radius:6px;
                      font-size:14px;color:#555;line-height:1.6;">
              🚚 We'll send you a shipping confirmation once your order is on
              its way. Estimated delivery: <strong>3–5 business days</strong>.
            </p>
          </td>
        </tr>

        <!-- Divider -->
        <tr>
          <td style="padding:0 40px;">
            <hr style="border:none;border-top:1px solid #eee;margin:0;"/>
          </td>
        </tr>

        <!-- Footer -->
        <tr>
          <td style="padding:24px 40px 32px;text-align:center;">
            <p style="margin:0;font-size:14px;color:#999;">
              Questions? Just reply to this email — we're happy to help.
            </p>
            <p style="margin:16px 0 0;font-size:15px;color:#555;">
              With warm regards,<br/>
              <strong style="color:#2e7d32;font-size:16px;">
                🤖 AI Shopping Assistant
              </strong>
            </p>
            <p style="margin:12px 0 0;font-size:12px;color:#bbb;">
              Powered by Mistral AI · LangGraph · Built for India 🇮🇳
            </p>
          </td>
        </tr>

      </table>

    </td></tr>
  </table>

</body>
</html>"""