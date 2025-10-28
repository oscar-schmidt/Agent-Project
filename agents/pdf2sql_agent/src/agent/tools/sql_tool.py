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
    db_port: int = 3307
    db_socket: str = "/Applications/XAMPP/xamppfiles/var/mysql/mysql2.sock"

    client: Optional[OpenAI] = None

    def __init__(self, api_key: Optional[str] = None, db_host: str = None, db_user: str = None, db_password: str = None, db_name: str = None, db_port: int = None, db_socket: str = None, **kwargs):
        super().__init__(**kwargs)

        self.api_key = api_key or os.getenv("OPENAI_API_KEY")
        self.db_host = db_host or "localhost"
        self.db_user = db_user or "readonly"
        self.db_password = db_password or "readonly"
        self.db_name = db_name or "business_db"
        self.db_port = db_port or 3307
        self.db_socket = db_socket or "/Applications/XAMPP/xamppfiles/var/mysql/mysql2.sock"

        # Initialize OpenAI client
        self.client = OpenAI(api_key=self.api_key)

    def _connect(self):
        return pymysql.connect(
            host=self.db_host,
            user=self.db_user,
            password=self.db_password,
            database=self.db_name,
            port=self.db_port,
            unix_socket=self.db_socket,
            cursorclass=DictCursor
        )

    def _generateQuery(self, question: str):
        """Generate a safe MySQL query from a natural-language question."""
        prompt = f"""
        You are a SQL expert assistant for a business management database.
        Convert the following natural-language question into a valid, executable MySQL query.

        "{question}"

        The database schema is as follows:
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
        - v_product_stock(product_id, sku, product_name, total_qty)
        - v_customer_ltv(customer_id, customer_name, lifetime_value, orders_count)
        - v_open_purchase_orders(po_id, supplier_name, order_date, status, qty_remaining)

        Rules:
        1. Output only the SQL query — no explanation.
        2. Use only SELECT statements. Never modify data.
        3. Include JOIN, GROUP BY, and ORDER BY as needed.
        4. Assume MySQL 8.0+ syntax.
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