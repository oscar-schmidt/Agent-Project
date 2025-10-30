from dotenv import load_dotenv
import os

load_dotenv()

DATA_PATH = os.getenv("DATA_PATH", ".agents/classification_agent/data/tech_service_reviews_500_with_names_ratings.csv")

# LLM
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "llama3.2")

# Sentiment Analysis Configuration
ENABLE_SENTIMENT = os.getenv("ENABLE_SENTIMENT", "true").lower() in ("true", "1", "yes")
SENTIMENT_MODEL = os.getenv("SENTIMENT_MODEL", "yangheng/deberta-v3-base-absa-v1.1")
SENTIMENT_CONFIDENCE_THRESHOLD = float(os.getenv("SENTIMENT_CONFIDENCE_THRESHOLD", "0.8"))
SENTIMENT_BOOST_THRESHOLD = float(os.getenv("SENTIMENT_BOOST_THRESHOLD", "-0.85"))

# Notion
NOTION_API_KEY = os.getenv("NOTION_API_KEY")
NOTION_DATABASE_ID = os.getenv("NOTION_DATABASE_ID")

# OpenAI
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
OPENAI_MODEL = os.getenv("OPENAI_MODEL")

# Anthropic Claude Configuration (for agentic workflows)
ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY")
AGENT_MODEL = os.getenv("AGENT_MODEL", "claude-3-5-sonnet-20241022")
AGENT_TEMPERATURE = float(os.getenv("AGENT_TEMPERATURE", "0"))
AGENT_CHECKPOINT_DB = os.getenv("AGENT_CHECKPOINT_DB", "classification_agent.db")
AGENT_VERBOSE = os.getenv("AGENT_VERBOSE", "true").lower() in ("true", "1", "yes")
AGENT_MAX_ITERATIONS = int(os.getenv("AGENT_MAX_ITERATIONS", "10"))

# Database Configuration (Cloud/Supabase support)
DATABASE_URL = os.getenv("DATABASE_URL")

# Memory Configuration (local, no docker/api needed)
MEMORY_ENABLED = os.getenv("MEMORY_ENABLED", "false").lower() in ("true", "1", "yes")
MEMORY_STORAGE_PATH = os.getenv("MEMORY_STORAGE_PATH", "./memory_storage")

# Embeddings (local sentence-transformers, no api key needed)
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "all-MiniLM-L6-v2")
