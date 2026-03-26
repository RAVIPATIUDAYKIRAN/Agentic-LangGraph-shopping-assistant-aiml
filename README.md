# LangGraph-shopping-assistant-aiml
Production-grade agentic AI shopping assistant built with pure LangGraph nodes + Mistral AI tool-calling, Redis, and MySQL — no intent routing, no keyword matching.




# 🛒 AI Shopping Assistant

A production-grade AI shopping assistant built with **pure LangGraph agentic architecture** 
and **Mistral AI tool-calling** — no intent routing, no keyword matching, no manual chaining.
Every user request flows through a state-driven graph where the LLM reasons, 
calls tools, and formats responses autonomously.

---

## 🧠 Architecture

This project is built on **pure LangGraph + LLM** principles:

- Zero intent routing — the LLM decides which tools to call
- Zero keyword filtering — the LLM translates user needs into search queries
- Pure state-driven graph transitions — Python logic only reads state, never parses text
- Two-node LLM design — `login_llm` (auth only) and `shopping_llm` (commerce only)
  with physically separate tool bindings, so cross-contamination is impossible at the
  framework level
```
auth_gate (Python) → login_llm / shopping_llm → tools → shopping_llm → END
```

---

## ✨ Features

- 🔐 **Secure authentication** — bcrypt password hashing, generic error messages 
  (no email enumeration), Redis session storage
- 🔍 **Intelligent product search** — relevance-ranked SQL (exact → title → category → brand),
  automatic fallback to popular products when nothing matches
- 🧠 **Context memory** — `last_products` persisted in LangGraph state so references like
  "add this", "tell me more", "buy it" resolve correctly across turns
- 🛒 **Full cart management** — add, remove, view, with Redis pipeline-based 
  optimistic locking (WATCH/MULTI) to prevent race conditions
- 📦 **Atomic order placement** — stock decrement + order INSERT run in a single 
  MySQL transaction; if either fails, both roll back
- ⚡ **Instant purchase** — `buy_now` tool orders a single product without 
  touching the existing cart
- 📧 **Async email notifications** — warm HTML order confirmation sent in a 
  background daemon thread; never blocks the order response
- 💰 **India-first** — all prices in INR (₹), built for Indian users

---

## 🏗️ Tech Stack

| Layer | Technology |
|---|---|
| Agent framework | LangGraph |
| LLM | Mistral AI (`mistral-large-latest`) |
| UI | Streamlit |
| Database | MySQL (orders, users, products) |
| Cache / Session | Redis |
| Auth | bcrypt |
| Email | Gmail SMTP (async, daemon thread) |

---

## 📁 Project Structure
```
├── app.py                  # Streamlit UI
├── graph.py                # LangGraph graph — nodes, routers, state
├── tools/
│   ├── auth_tools.py       # login_user tool
│   ├── product_tools.py    # search, filter, details, reviews, clarify
│   └── cart_tools.py       # add, remove, view, place_order, buy_now
├── services/
│   └── email_service.py    # HTML email with html.escape security
├── memory/
│   └── redis_memory.py     # Cart, session, order storage (atomic pipelines)
├── db/
│   └── db_client.py        # MySQL pool, execute_query, execute_transaction
└── .env.example            # Environment variable template
```

---

## ⚙️ Setup

**1. Clone the repo**
```bash
git clone https://github.com/RAVIPATIUDAYKIRAN/Agentic-LangGraph-shopping-assistant-aiml.git
cd Agentic-LangGraph-shopping-assistant-aiml
```

**2. Install dependencies**
```bash
pip install streamlit langchain-mistralai langgraph langchain-core \
            mysql-connector-python redis bcrypt python-dotenv
```

**3. Configure environment**
```bash
cp .env.example .env
# Fill in your Mistral API key, MySQL credentials, Redis config, Gmail App Password
```

**4. Set up MySQL**
```sql
CREATE DATABASE shopping_assistant;

CREATE TABLE users (
    user_id       INT PRIMARY KEY AUTO_INCREMENT,
    email         VARCHAR(255) UNIQUE NOT NULL,
    full_name     VARCHAR(255),
    password_hash VARCHAR(255) NOT NULL
);

CREATE TABLE products (
    product_id   INT PRIMARY KEY AUTO_INCREMENT,
    title        VARCHAR(255) NOT NULL,
    description  TEXT,
    category     VARCHAR(100),
    brand        VARCHAR(100),
    price        DECIMAL(10,2) NOT NULL,
    rating       DECIMAL(3,2) DEFAULT 0,
    rating_count INT DEFAULT 0,
    stock        INT DEFAULT 0
);

CREATE TABLE orders (
    order_id     VARCHAR(20) PRIMARY KEY,
    user_id      INT NOT NULL,
    total_amount DECIMAL(10,2),
    items_json   TEXT,
    created_at   TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (user_id) REFERENCES users(user_id)
);
```

**5. Run**
```bash
streamlit run app.py
```

---

## 💬 Example Conversations
```
User:  I'm thirsty, need something for hiking
Bot:   Here are some water bottles... [from DB only]

User:  Add the first one to my cart
Bot:   ✅ Added Milton Water Bottle (₹299) to your cart

User:  Place the order
Bot:   🎉 Order ORD-A1B2C3D4 confirmed! Total: ₹299
       [Email sent in background]
```

---

## 🔒 Security

- Passwords hashed with bcrypt (never stored in plain text)
- Auth failures return a single generic message (prevents user enumeration)
- HTML email template uses `html.escape()` on all user-derived data
- MySQL transactions ensure stock and orders are always consistent
- Redis cart operations use WATCH/MULTI for optimistic concurrency control

---

## 🇮🇳 Built for India

All prices displayed in Indian Rupees (₹).  
Designed for Indian e-commerce workflows and users.

---

## 📄 License

MIT
```

---

**GitHub Topics to add** (helps discoverability):
```
langgraph  mistral-ai  langchain  streamlit  ai-agent  
shopping-assistant  tool-calling  redis  mysql  python
