"""Central configuration: paths and model declaration.

Model name is declared here in source (not in .env) per assignment rule 9.4 —
only the API key lives in .env.
"""
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT_DIR / "data"
INPUT_DIR = ROOT_DIR / "input"
OUTPUT_DIR = ROOT_DIR / "output"
LOGGING_DIR = ROOT_DIR / "logging"
TRACE_PATH = LOGGING_DIR / "trace.jsonl"
METADATA_PATH = LOGGING_DIR / "metadata.json"

# Groq-hosted model used by every LLM-backed agent node. 8B params, under the
# 10B/agent cap in README section 9.1.
GROQ_MODEL_NAME = "llama-3.1-8b-instant"
GROQ_MODEL_PARAMS = "8B"
LLM_TEMPERATURE = 0.0

POLICY_VERSION = "EC_POLICY_V1"

MAX_ENTITY_IDS = 5
MAX_EVIDENCE_IDS = 10
MAX_ROOT_CAUSES = 3
MAX_RESPONSIBLE_PARTIES = 3
MAX_ACTIONS = 5
