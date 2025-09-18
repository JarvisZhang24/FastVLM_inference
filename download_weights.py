from huggingface_hub import snapshot_download, login
from dotenv import load_dotenv
import os
from pathlib import Path
import sys

# set project root
PROJECT_ROOT = Path(__file__).parent
MODEL_DIR = PROJECT_ROOT / "checkpoints" / "FastVLM-1.5B"
sys.path.append(str(PROJECT_ROOT))

# load environment variables from .env file
load_dotenv()
HF_token = os.getenv("HF_TOKEN")

# login to huggingface
login(token=HF_token)
#   download the model
snapshot_download(
    repo_id="apple/FastVLM-1.5B",
    local_dir=str(MODEL_DIR),
    local_dir_use_symlinks=False 
)



