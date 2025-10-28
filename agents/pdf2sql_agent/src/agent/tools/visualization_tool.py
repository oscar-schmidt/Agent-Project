import pandas as pd
import json
import plotly.express as px
from openai import OpenAI
import os
from typing import Optional, Any
from langchain_core.tools import BaseTool

class VisualizationTool(BaseTool):
    model_config = {"arbitrary_types_allowed": True}
    name: str = "execute_visualization_tool"
    description: str = ("Generates visualizations from pandas DataFrames based on user queries using GPT to choose chart types and axes.")

    api_key: str = ""
    df : Any = None
    query : str = ""
    question : str = ""
    outputPath: str = "TEMPVIS.png"
    client: Optional[OpenAI] = None # not 100% sure about this one

    def __init__(self, api_key: Optional[str] = None, df: Any = None, query: str = "", question: str = "", outputPath: str = "TEMPVIS.png", **kwargs):
        super().__init__(**kwargs)
        
        self.api_key = api_key or os.getenv("OPENAI_API_KEY")
        self.df = df or pd.DataFrame()
        self.query = query
        self.question = question
        self.outputPath = outputPath or "TEMPVIS.png"
        self.client = OpenAI(api_key=self.api_key)

    def _chooseChart(self, df: pd.DataFrame, query: str, question: str):
        schema = " | ".join(f"{feature} ({df[feature].dtype})" for feature in df.columns)
        sample = df.head().to_dict()

        print(f"Schema: {schema}")
        print(f"Sample data: {sample}")
        print(f"User query: {query}")
        print(f"User question: {question}")

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
        3. Suggest a file name to save the chart image as.

        Respond ONLY in JSON format like this:
        {{"type": "bar", "x": "category_name", "y": "sales", "filename": "sales_chart.png"}}

        IMPORTANT - These are the only allowed chart types: ["bar", "line", "scatter", "histogram"].
        Only use column names that exist in the schema provided.
        If a chart type doesn't need both axes (like pie or histogram), omit the unused one.
        """

        response = self.client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "system", "content": prompt}]
        )

        info = json.loads(response.choices[0].message.content)

        type = info.get("type")

        if info.get("x") is not None:
            x = info.get("x")
        else:
            x = None

        if info.get("y") is not None:
            y = info.get("y")
        else:
            y = None

        if info.get("filename") is not None:
            filename = info.get("filename")
        else:
            filename = "TEMPVIS.png"

        return type, x, y, filename

    def _makeChart(self, df: pd.DataFrame, type: str, x: str, y: str):
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

    def _run(self, df: str, query: str, question: str, outputPath: str = "TEMPVIS.png"):
        df = df
        query = query
        question = question
        outputPath = outputPath or self.outputPath
        fig = None

        try:
            if isinstance(df, str):
                df = pd.DataFrame(json.loads(df))
            elif isinstance(df, dict) or isinstance(df, list):
                df = pd.DataFrame(df)

            type, x, y, filename = self._chooseChart(df, query, question)

            if filename != None:
                outputPath = filename

            print(f"Chosen chart type: {type}, x: {x}, y: {y}, outputPath: {outputPath}")

            if (x not in df.columns) or (y not in df.columns) or (x == y):
                y = None

            if y != None:
                fig = self._makeChart(df, type, x, y)
                fig.write_image(outputPath)

            if fig == None:
                return json.dumps({
                    "error": "No chart made.",
                }, indent=2)
            
            summary = {
                "schema": " | ".join(f"{feature} ({df[feature].dtype})" for feature in df.columns),
                "x-axis": x,
                "y-axis": y,
                "type": type
            }

            return json.dumps({
                "summary": summary,
                "image_path": outputPath
            }, indent=2)
        
        except Exception as e:
            return json.dumps({
                "error": str(e),
                "query": None,
            }, indent=2)
        
execute_visualization_tool = VisualizationTool()