import pymysql
from pymysql.cursors import DictCursor
from openai import OpenAI
import os

api_key = os.getenv("OPENAI_API_KEY")

client = OpenAI()

question = "How much stock is in each warehouse" # put a question here :)

mydb = pymysql.connect(
    host="localhost",
    user="readonly",
    password="readonly",
    database="business_db",
    port=3307,
    unix_socket="/Applications/XAMPP/xamppfiles/var/mysql/mysql2.sock",
    cursorclass=DictCursor
)

cursor = mydb.cursor()

prompt1 = f"""
You are a SQL expert assistant for a business management database. 
Your task is to convert the following natural-language question into a valid, executable MySQL SQL query:

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
1. Output only the SQL query — no explanation, markdown, or comments.
2. Use proper JOINs based on table relationships.
3. Assume MySQL 8.0+ (supports CTEs, window functions, and DATE_FORMAT).
4. Use readable column aliases when appropriate.
5. Include WHERE, GROUP BY, and ORDER BY when logically implied.
6. When unclear which columns to show, choose the most relevant human-readable ones (names, totals, dates, etc.).
7. If the question implies a time range like “this month” or “this year,” use `CURDATE()` and MySQL date functions.

Example Input:
"Show all suppliers who provided products that are currently low in stock."

Example Output:
SELECT DISTINCT s.name AS supplier_name, p.name AS product_name, v.total_qty
FROM suppliers s
JOIN products p ON p.supplier_id = s.supplier_id
JOIN v_product_stock v ON v.product_id = p.product_id
WHERE v.total_qty <= p.reorder_level;

Example Input:
"Show total revenue per month this year."

Example Output:
SELECT DATE_FORMAT(order_date, '%Y-%m') AS month, SUM(total_amount) AS total_revenue
FROM sales_orders
WHERE YEAR(order_date) = YEAR(CURDATE()) AND status IN ('PAID','FULFILLED')
GROUP BY DATE_FORMAT(order_date, '%Y-%m')
ORDER BY month;
"""

response1 = client.chat.completions.create(
    model="gpt-4",
    messages=[{"role": "user", "content": prompt1}]
)

query = response1.choices[0].message.content

print(query)

if any(keyword in query.lower() for keyword in ["insert", "update", "delete", "drop", "alter", "truncate", "create"]):
    raise ValueError("Unsafe query")

cursor.execute(query)
result = cursor.fetchall()

print(result)

prompt2 = f"""
You are a SQL expert assistant for this archery database. The user has asked a question and you have generated a SQL query to answer it.
The question was: "{question}"
The SQL query was: "{query}"
The result of the query is: "{result}"
present the result with simple facts.
"""

response2 = client.chat.completions.create(
    model="gpt-4",
    messages=[{"role": "user", "content": prompt2}]
)

answer = response2.choices[0].message.content

print(answer)
