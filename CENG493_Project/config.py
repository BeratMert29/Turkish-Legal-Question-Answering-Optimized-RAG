from pathlib import Path

BASE_DIR = Path(__file__).parent

# Chunking
CHUNK_SIZE = 1400
CHUNK_OVERLAP = 180
CORPUS_DOC_MIN_CHARS = 180

# Data
QA_EVAL_EXPECTED = 300
QA_GOLD_FILE = "qa_eval.jsonl"
RAW_DATA_PATH = BASE_DIR.parent / "combined_dataset.csv"
PROCESSED_DIR = BASE_DIR / "data/processed"
INDEX_DIR = BASE_DIR / "index"
INDEX_FILE = "faiss.index"
METADATA_FILE = "metadata.jsonl"
RESULTS_DIR = BASE_DIR / "results/stage1"

# Embedding
EMBEDDING_MODEL = "intfloat/multilingual-e5-large"
EMBEDDING_DIM = 1024
EMBEDDING_BATCH_SIZE = 32

# Retrieval
TOP_K_RETRIEVAL = 10
TOP_K_FOR_GENERATION = 5
CONTEXT_WINDOW_CHARS = 8000

# Re-ranker (Stage 2 retrieval)
RERANKER_MODEL = "BAAI/bge-reranker-v2-m3"
RERANKER_CANDIDATES = 50   # initial dense/RRF pool before cross-encoder re-ranking
RRF_K = 60                 # RRF smoothing constant

# LLM (Ollama — free, no API key)
LLM_MODEL = "qwen2.5:7b"
LLM_BASE_URL = "http://localhost:11434/v1"
LLM_API_KEY = "ollama"
LLM_TEMPERATURE = 0.0
LLM_MAX_TOKENS = 512

# Evaluation
HALLUCINATION_SAMPLE_SIZE = 150

# Hallucination stratification thresholds (applied to top-1 retrieval score)
HALLUCINATION_HIT_THRESHOLD = 0.7
HALLUCINATION_PARTIAL_THRESHOLD = 0.4

# BM25 tokenization
BM25_MIN_TOKEN_LENGTH = 2

# Oracle relevance (scripts/03_evaluate_retrieval.py)
TOP_K_ORACLE = 5
