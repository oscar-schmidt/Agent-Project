import pandas as pd
import pymysql
import json
from pymysql.cursors import DictCursor
import plotly.express as px
from openai import OpenAI
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Image
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.lib.pagesizes import letter
import os

api_key = os.getenv("OPENAI_API_KEY")

client = OpenAI()

question = "Please give me a summary of payments by customer" # put a question here :)

mydb = pymysql.connect(
    host="localhost",
    user="readonly",
    password="readonly",
    database="business_db",
    port=3307,
    unix_socket="/Applications/XAMPP/xamppfiles/var/mysql/mysql2.sock",
    cursorclass=DictCursor
)

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

if any(keyword in query.lower() for keyword in ["insert", "update", "delete", "drop", "alter", "truncate", "create"]):
    raise ValueError("Unsafe query")

cursor = mydb.cursor()
cursor.execute(query)
result = cursor.fetchall()

print(result)

cursor.close()
mydb.close()

df = pd.DataFrame(result)

# dumb chart picker

    # numeric = df.select_dtypes(include=['number']).columns.tolist()
    # categorical = df.select_dtypes(include=['category']).columns.tolist()
    # time = df.select_dtypes(include=['datetime64']).columns.tolist()

    # fig = None

    # if time:
    #     dateCol = time[0]
    #     if len(numeric) >= 1:
    #         numericCol = numeric[0]
    #         fig = px.line(df, x=dateCol, y=numericCol, title='Line Chart')

    # elif len(numeric) == 1 and len(categorical) >= 1:
    #     numericCol = numeric[0]
    #     catCol = categorical[0]
    #     fig = px.bar(df, x=catCol, y=numericCol, title='Bar Chart')

    # elif len(numeric) >= 2:
    #     fig = px.scatter(df, x=numeric[0], y=numeric[1], title='Scatter Plot')

    # elif len(categorical) >= 2:
    #     cat_col1 = categorical[0]
    #     cat_col2 = categorical[1]
    #     pie_data = df.groupby(cat_col1)[cat_col2].count().reset_index()
    #     pie_data.columns = [cat_col1, 'count']
    #     fig = px.pie(pie_data, names=cat_col1, values='count', title='Pie Chart')


    # fig.show()

# Smarter one

def ChooseChart(df, query):
    schema = " | ".join(f"{feature} ({df[feature].dtype})" for feature in df.columns)
    sample = df.head().to_dict()

    # made with chatGPT
    prompt = f"""
    You are a data visualization expert.
    Given this table schema and sample data:
    {schema}
    Sample rows: {sample}

    The user query was: "{query}"

    The users question was: "{question}"

    Your task:
    1. Choose the best chart type for the data.
    2. Suggest the most appropriate column(s) for the X and Y axes (if applicable).

    Respond ONLY in JSON format like this:
    {{"type": "bar", "x": "category_name", "y": "sales"}}

    IMPORTANT - These are the only allowed chart types: ["bar", "line", "scatter", "histogram"].
    Only use column names that exist in the schema provided.
    If a chart type doesn't need both axes (like pie or histogram), omit the unused one.
    """

    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "system", "content": prompt}]
    )

    info = json.loads(response.choices[0].message.content)

    type = info.get("type")
    x = info.get("x")

    if info.get("y") is not None:
        y = info.get("y")
    else:
        y = None

    return type, x, y

def makeChart(df, type, x, y):
    fig = None

    if type == "line":
        fig = px.line(df, x=x, y=y, title='Line Chart')
    elif type == "bar":
        fig = px.bar(df, x=x, y=y, title='Bar Chart')
    elif type == "scatter":
        fig = px.scatter(df, x=x, y=y, title='Scatter Plot')
    elif type == "histogram":
        fig = px.histogram(df, x=x, title='Histogram')

    return fig

def reportGenerator(df, query, x, y, graph: bool = False):
    schema = " | ".join(f"{feature} ({df[feature].dtype})" for feature in df.columns)
    sample = json.loads(df.head().to_json(orient="records"))
    summary = json.loads(df.describe(include='all').to_json())

    if graph == True:
        prompt = f"""
        You are a data visualization expert.
        Using the following data details:

        Question: {question}
        Query: {query}
        Schema: {schema}
        Summary statistics: {summary}
        Sample rows: {sample}

        Please write a clear, structured, and professional report about the data. Do not add any titles.

        - In the first paragraph, summarize key findings and insights based on the schema, summary statistics, and sample data.
        - In the second paragraph, describe and interpret the chart created, explicitly referencing the X axis as '{x}' and the Y axis as '{y}' (if applicable).
        - In the third paragraph, provide any additional observations or potential implications from the data.

        The output should be suitable for inclusion in a PDF report.
        """
    else:
        prompt = f"""
        You are a data analysis expert.
        Using the following data details:

        Question: {question}
        Query: {query}
        Schema: {schema}
        Summary statistics: {summary}
        Sample rows: {sample}

        Please write a clear, structured, and professional report about the data. Do not add any titles.

        - In the first paragraph, summarize key findings and insights based on the schema, summary statistics, and sample data.
        - In the second paragraph, provide any additional observations or potential implications from the data.

        The output should be suitable for inclusion in a PDF report.
        """

    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "system", "content": prompt}]
    )

    return response.choices[0].message.content.strip()

type, x, y = ChooseChart(df, query)
print(f"Chosen chart type: {type}, x: {x}, y: {y}")

graph = False

if (x not in df.columns) or (y not in df.columns) or (x == y):
    y = None

if y != None:
    fig = makeChart(df, type, x, y)
    fig.write_image("tempChart.png")

    graph = True

    fig.show()

report = reportGenerator(df, query, x, y, graph)
print(report)

styles = getSampleStyleSheet()
story = []

title = Paragraph("SQL Data Report", styles['Title'])
story.append(title)
story.append(Spacer(1, 12))

paragraphs = [p.strip() for p in report.split('\n') if p.strip()]

for i, para in enumerate(paragraphs):
    p = Paragraph(para, styles['Normal'])
    story.append(p)
    story.append(Spacer(1, 12))
    if i == 1 & graph == True:
        img = Image("tempChart.png", width=400, height=300)
        story.append(img)
        story.append(Spacer(1, 12))

doc = SimpleDocTemplate("report.pdf", pagesize=letter)
doc.build(story)