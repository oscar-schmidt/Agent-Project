from langchain_core.tools import BaseTool
import pandas as pd


class CSVTool(BaseTool):
    """
    Tool that can read and parse both Excel file types and csv returns a string of text may update later to return json for a
    more structured response
    """
    name: str = "ParsCSVOrExcel"
    description: str = "A tool that is able to read a csv or excel file contents given a link"

    def _run(self, path: str) -> str:
        if path.endswith(".csv"):
            df = pd.read_csv(path)
            return df.to_string(index=False)
        elif path.endswith(".xlsx"):
            df = pd.read_excel(path)
            return df.to_string(index=False)
        else:
            return f"File type {path.split('.')[-1]} not supported."


