# Core dependencies: dataclasses, environment variables, and path handling.
from dataclasses import dataclass
import os
from dotenv import load_dotenv

# Load .env before reading environment variables.
# If .env is not in the project root, pass an explicit dotenv_path.
load_dotenv()


# MinerU service configuration.
@dataclass
class MineruConfig:
    base_url: str
    api_key : str

mineru_config = MineruConfig(
    base_url=os.getenv("MINERU_BASE_URL"),
    api_key=os.getenv("MINERU_API_TOKEN")
)
