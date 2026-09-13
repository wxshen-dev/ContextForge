# app/utils/path_utils.py
from pathlib import Path
from dotenv import load_dotenv
import os
from pathlib import Path

def get_path_dir(ps:int = 0)->Path:
    """
    Return an ancestor directory for this file using pathlib.Path.parents.
    parents[0] is equivalent to .parent, parents[1] to .parent.parent, and so on.
    :param ps:
    :return:
    """
    dir_path = Path(__file__).parents[ps]
    return dir_path


def get_project_root(identifier: str = ".env") -> Path:
    # Step 1: prefer the PROJECT_ROOT environment variable in production.
    env_root = os.getenv("PROJECT_ROOT")
    if env_root and Path(env_root).absolute().exists():
        return Path(env_root).absolute()

    # Step 2: find and load the root .env file before continuing.
    current_dir = Path(__file__).absolute().parent
    while current_dir != current_dir.parent:
        if (current_dir / identifier).exists():
            load_dotenv(dotenv_path=current_dir / identifier)
            break
        current_dir = current_dir.parent

    # Step 3: recursively find the identifier as a development fallback.
    current_dir = Path(__file__).absolute().parent
    while current_dir != current_dir.parent:
        if (current_dir / identifier).exists():
            return current_dir
        current_dir = current_dir.parent

    raise FileNotFoundError(
        f"Project root identifier '{identifier}' was not found, and PROJECT_ROOT is not configured."
    )


PROJECT_ROOT = get_project_root(".env")
