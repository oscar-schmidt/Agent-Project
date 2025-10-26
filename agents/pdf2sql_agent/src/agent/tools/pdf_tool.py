import os
import json
import pandas as pd
from typing import Any, Dict
from openai import OpenAI
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Image
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.lib.pagesizes import letter
from typing import Optional
from decimal import Decimal

from langchain_core.tools import BaseTool

class DecimalEncoder(json.JSONEncoder):  # added this because the json .dumps was having trouble with dataframes
    def default(self, obj):
        if isinstance(obj, Decimal):
            return float(obj)
        if isinstance(obj, pd.Timestamp):
            return str(obj)
        return super().default(obj)

class PDFTool(BaseTool):
    model_config = {"arbitrary_types_allowed": True}
    name: str = "generate_pdf_report"
    description: str = ("Generates a structured PDF report summarizing a DataFrame's key statistics and trends, optionally including a data visualization.")

    api_key: str = ""
    outputPath: str = "report.pdf"

    client: Optional[OpenAI] = None

    def __init__(self, api_key: str = None, outputPath: str = None, **kwargs):
        super().__init__(**kwargs)
        
        self.api_key = api_key or os.getenv("OPENAI_API_KEY")
        self.outputPath = outputPath or "report.pdf"

        self.client = OpenAI(api_key=self.api_key)

    def _generate_text(self, df, query: str, question: str, x: str, y: str, graph: bool):
        schema = " | ".join(f"{col} ({df[col].dtype})" for col in df.columns)
        sample = json.loads(df.head().to_json(orient="records"))
        summary = json.loads(df.describe(include='all').convert_dtypes().to_json(default_handler=str))

        if graph:
            role = "data visualization expert"
            sections = """
            - In the first paragraph, summarize key findings based on schema and summary statistics.
            - In the second paragraph, describe and interpret the chart referencing '{x}' and '{y}'.
            - In the third paragraph, discuss any broader implications or patterns in the data.
            """
        else:
            role = "data analysis expert"
            sections = """
            - In the first paragraph, summarize key findings from schema, summary, and sample data.
            - In the second paragraph, discuss additional insights or implications.
            """

        prompt = f"""
        You are a {role}.
        Use the following context to write a professional, concise data report.

        Question: {question}
        Query: {query}
        Schema: {schema}
        Summary: {summary}
        Sample rows: {sample}

        {sections}

        The tone should be analytical and business-friendly.
        Do not include headings or bullet points.
        """

        response = self.client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "system", "content": prompt}]
        )

        return response.choices[0].message.content.strip()
    
    def _build_pdf(self, reportText: str, graph: bool = False, graphPath: str = "TEMPVIS.png"):
        styles = getSampleStyleSheet()
        story = []

        story.append(Paragraph("SQL Data Report", styles["Title"]))
        story.append(Spacer(1, 12))

        paragraphs = [p.strip() for p in reportText.split("\n") if p.strip()] # if p.strip to make sure there arent empty paragraphs

        for i, paraText in enumerate(paragraphs):
            story.append(Paragraph(paraText, styles["Normal"]))
            story.append(Spacer(1, 12))
            if i == 1 and graph and os.path.exists(graphPath):
                story.append(Image(graphPath, width=400, height=300))
                story.append(Spacer(1, 12))

        doc = SimpleDocTemplate(self.outputPath, pagesize=letter)
        doc.build(story)

    def _run(self, df: Any, query: str,question: str, x: str = "", y: str = "", graph: bool = False):
        try:
            if isinstance(df, str):
                try:
                    # Attempt to load serialized JSON
                    df = pd.DataFrame(json.loads(df))
                except Exception:
                    # Some nodes might send it as already valid JSON text (not list)
                    df = pd.read_json(df)

            report_text = self._generate_text(df, query, question, x, y, graph)
            self._build_pdf(report_text, graph=graph)

            return json.dumps({
                "message": "PDF report generated.",
                "file_path": self.output_path
            }, indent=2, cls=DecimalEncoder)
        
        except Exception as e:
            return json.dumps({
                "error": f"Failed to generate report: {str(e)}",
                "file_path": None
            }, indent=2, cls=DecimalEncoder)

generate_pdf_report = PDFTool()