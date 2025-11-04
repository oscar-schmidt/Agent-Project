# Project D
## 🏁 Getting Started
### Prerequisites
Before you begin, please install the following software:

1.  **Python**
    * **Requirements:** `Python:latest`
    * **Installation**: Can be installed from [Python Org Website](https://www.python.org/downloads/)

2.  **A Python Package Manager**
    You will need a Python package manager to install all the necessary packages. You can use `pip` or a better alternative, `uv`.
    * `pip` is installed by default with Python, so no extra installation is required.
    * `uv` can be installed from [The Astral Website](https://docs.astral.sh/uv/getting-started/installation/).

3.  **LLM Models**
    At the moment, the application supports **Ollama**, **OpenAI**, or **Anthropic**.

    3.1. **Ollama**
    You will need the following Ollama models pulled:
    * `qwen3:8b` for the chat completion requirements
    * `unknown` for the embedding models

    3.2. **OpenAI**
    You will simply need to set the API key as `OPENAI_API_KEY`. The application should automatically use the right models, which are `gpt-4o-mini` and `text-embedding-small1`.

    3.3. **Anthropic**
    Similar to OpenAI, the Anthropic API key needs to be set as `ANTHROPIC_API_KEY`.

### Installation

Start the installation process by obtaining the code, which can be cloned from the [GitHub repository](https://github.com/oscar-schmidt/Agent-Project.git). If you don't wish to clone it, you can install and unzip the code manually. Head to the project directory and open a terminal window.

After, follow the following steps for **`pip`**:

1.  Start by creating the virtual environment:
    ```python -m venv .venv```
2.  Then, on Linux or macOS, run the following to enter the virtual environment:
    ```source .venv/bin/activate```
    Or on Windows, run:
    ```.venv\Scripts\activate```
    (On Windows, if you get an ExecutionPolicy error, run this command first: `Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser`)
3.  Then run the following to install all the necessary packages:
    ```pip install -r requirements.txt```
    
Or for **`uv`**:
1.  Run `uv sync`. It should automatically create the virtual environment and install all the necessary packages.

### Running The Application
1.  Navigate to the project directory.
**For `uv`:**
1.  We have provided a `proc` file that can be used to run all the necessary servers and agents for you at the same time. This can be done by opening a terminal window and running `uv run honcho start`. However, for testing purposes, this is not recommended as you will miss the output of the agents because everything will run in the same terminal window.
2.  Or, open 6 terminal windows and run the following commands in the order provided:
    1.  Websocket Server: `uv run python -m communication_server.server`
    2.  Directory Agent: `uv run python -m agents.directory_agent.directory_agent`
    3.  Web Agent: `uv run python -m agents.web_agent.web_agent`
    4.  Classification Agent: `uv run python -m agents.classification_agent.classification_agent`
    5.  PDF-to-SQL Agent: `uv run python -m agents.pdf2sql_agent.pdf2sql_agent`
    6.  Main Agent: `uv run main.py`

**For `pip`:**
1.  Similar to `uv`, you can run the `proc` file by activating a virtual environment in one terminal. (If you don't know how, please look at the second step in the installation instructions).
2.  Or, open 6 terminals, activate the virtual environment in all six, and run the following commands in the order provided below:
    1.  Websocket Server: `python -m communication_server.server`
    2.  Directory Agent: `python -m agents.directory_agent.directory_agent`
    3.  Web Agent: `python -m agents.web_agent.web_agent`
    4.  Classification Agent: `python -m agents.classification_agent.classification_agent`
    5.  PDF-to-SQL Agent: `python -m agents.pdf2sql_agent.pdf2sql_agent`
    6.  Main Agent: `python main.py`
3.  After running all the necessary agents and the communication server, you should automatically be served with a nice GUI webpage, and you can simply interact with this agent.

## Usage
The ideal use case of this multi-agent system is for business improvement through reviews. However, with the modular implementation of the project, it can be extended for almost any use case.
A quick way you can test the whole system is by providing a link to the main agent (which you can interact with using the webpage). Give it a link to a critical review of a product and ask it to: "retrieve the review, perform sentiment analysis and classification on it, and provide me with solutions for improving any issues gathered from the review."
Ideally, the main agent should contact the necessary agents for help and provided


**NOTE** incomplete at the moment will update later.
