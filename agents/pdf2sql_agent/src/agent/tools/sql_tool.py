import os
import json
import pymysql
import pandas as pd
from typing import Optional, List, Any, Dict
from pymysql.cursors import DictCursor
from openai import OpenAI

from langchain_core.tools import BaseTool

class SQLQueryTool(BaseTool):
    model_config = {"arbitrary_types_allowed": True}
    name: str = "execute_sql_query"
    description: str = ("Converts natural-language business questions into safe SQL queries using GPT, executes them on the database, and returns pandas dataframe results.")

    api_key: str = ""
    db_host: str = "localhost"
    db_user: str = "readonly"
    db_password: str = "readonly"
    db_name: str = "business_db"
    # XAMPP configuration (commented out)
    # db_port: int = 3307
    # db_socket: str = "/Applications/XAMPP/xamppfiles/var/mysql/mysql2.sock"
    # Homebrew MySQL configuration
    db_port: int = 3306
    db_socket: str = ""

    client: Optional[OpenAI] = None

    def __init__(self, api_key: Optional[str] = None, db_host: str = None, db_user: str = None, db_password: str = None, db_name: str = None, db_port: int = None, db_socket: str = None, **kwargs):
        super().__init__(**kwargs)

        self.api_key = api_key or os.getenv("OPENAI_API_KEY")
        self.db_host = db_host or "localhost"
        self.db_user = db_user or "readonly"
        self.db_password = db_password or "readonly"
        self.db_name = db_name or "business_db"
        # XAMPP configuration (commented out)
        # self.db_port = db_port or 3307
        # self.db_socket = db_socket or "/Applications/XAMPP/xamppfiles/var/mysql/mysql2.sock"
        # Homebrew MySQL configuration
        self.db_port = db_port or 3306
        self.db_socket = db_socket or ""

        # Initialize OpenAI client
        self.client = OpenAI(api_key=self.api_key)

    def _connect(self):
        # Build connection params, only include unix_socket if it's set
        conn_params = {
            "host": self.db_host,
            "user": self.db_user,
            "password": self.db_password,
            "database": self.db_name,
            "port": self.db_port,
            "cursorclass": DictCursor
        }

        # Only add unix_socket if it's not empty (Homebrew MySQL uses TCP, not socket)
        if self.db_socket:
            conn_params["unix_socket"] = self.db_socket

        return pymysql.connect(**conn_params)

    def _generateQuery(self, question: str):
        """Generate a safe MySQL query from a natural-language question."""
        prompt = f"""
        You are a SQL expert assistant for a business management database.
        Convert the following natural-language question into a valid, executable MySQL query.

        "{question}"

        The database schema is as follows:

        VIEWS (Use these for most queries - they're optimized and pre-joined):
        - v_product_stock(product_id, sku, product_name, total_qty) - USE THIS for product inventory queries
        - v_customer_ltv(customer_id, customer_name, lifetime_value, orders_count)
        - v_open_purchase_orders(po_id, supplier_name, order_date, status, qty_remaining)

        TABLES:
        - departments(department_id, name)
        - employees(employee_id, department_id, manager_id, first_name, last_name, email, phone, hire_date, salary)
        - customers(customer_id, name, email, phone, billing_address, shipping_address, created_at)
        - suppliers(supplier_id, name, email, phone, address)
        - categories(category_id, parent_id, name, description)
        - products(product_id, category_id, supplier_id, sku, name, description, unit_price, cost_price, reorder_level, is_active)
        - warehouses(warehouse_id, name, address)
        - inventory(product_id, warehouse_id, quantity, safety_stock)
        - purchase_orders(po_id, supplier_id, employee_id, warehouse_id, order_date, expected_date, status, notes)
        - purchase_order_items(poi_id, po_id, product_id, qty_ordered, qty_received, unit_cost)
        - sales_orders(so_id, customer_id, employee_id, order_date, status, total_amount, shipping_address, billing_address)
        - sales_order_items(soi_id, so_id, product_id, quantity, unit_price, discount)
        - payments(payment_id, so_id, payment_date, amount, method, reference)
        - shipments(shipment_id, so_id, warehouse_id, ship_date, carrier, tracking_number)
        - returns(return_id, so_id, customer_id, product_id, return_date, quantity, reason, refund_amount)

        Rules:
        1. Output only the SQL query — no explanation.
        2. Use only SELECT statements. Never modify data.
        3. Include JOIN, GROUP BY, and ORDER BY as needed.
        4. Assume MySQL 8.0+ syntax.
        5. IMPORTANT: When searching for products, customers, suppliers, or any names, ALWAYS use LIKE with wildcards (e.g., LIKE '%keyword%') instead of exact equality (=).
           - Extract key identifying terms from the search phrase and search for those separately
           - Example: For "ASUS RTX 5090", search for the model number: WHERE product_name LIKE '%5090%' OR sku LIKE '%5090%'
           - Example: For "RTX 5070 Ti", use: WHERE (product_name LIKE '%5070%' OR sku LIKE '%5070%')
           - For customer names like "John Smith", use: WHERE name LIKE '%John%' AND name LIKE '%Smith%'
        6. When searching by product names or SKUs, prioritize searching for the most specific identifier (model numbers, unique SKUs) rather than brand names.
           - Model numbers (5090, 5070, 9070XT) are more unique than brand names (ASUS, NVIDIA)
        7. ALWAYS use the VIEWS for product inventory queries. For product stock/availability questions, use v_product_stock directly - do NOT join products and inventory tables.
        8. When using JOINs, ALWAYS prefix column names with the table/view alias to avoid ambiguity (e.g., p.product_id, not just product_id).
        """

        response = self.client.chat.completions.create(
            model="gpt-4",
            messages=[{"role": "user", "content": prompt}]
        )

        sqlQuery = response.choices[0].message.content.strip()

        if any(keyword in sqlQuery.lower() for keyword in ["insert", "update", "delete", "drop", "alter", "truncate", "create"]):
            raise ValueError("Unsafe SQL query detected.")

        return sqlQuery
    
    def _executeQuery(self, query: str):
        conn = self._connect()
        cursor = conn.cursor()

        cursor.execute(query)
        rows = cursor.fetchall()

        cursor.close()
        conn.close()

        return pd.DataFrame(rows)

    def _run(self, question: str):
        """
        Run the SQL query tool: generate, execute, and return summarized results.
        """
        try:
            query = self._generateQuery(question)
            df = self._executeQuery(query)

            if df.empty:
                return {
                    "error": "No results.",
                    "query": query,
                    "rows": 0,
                    "df": None
                }

            summary = {
                "schema": " | ".join(f"{feature} ({df[feature].dtype})" for feature in df.columns),
                "rows": len(df),
                "preview": df.head(5).to_dict(orient="records")
            }

            return {
                "query": query,
                "summary": summary,
                "df": df.to_json(orient="records"),
                "question": question
            }

        except Exception as e:
            return {
                "error": str(e),
                "query": None,
                "rows": 0,
                "df": None
            }


# Instantiate default instance
execute_sql_query = SQLQueryTool()