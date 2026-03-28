"""Generate answers for test set via Ollama. Supports checkpoint/resume."""
import json
import sys
import time
from pathlib import Path
from tqdm import tqdm
_project_root = str(Path(__file__).parent.parent)
if _project_root not in sys.path:
    sys.path.append(_project_root)
import config
from data.data_processor import DataProcessor
from retrieval.embedder import Embedder
from retrieval.retriever import Retriever
from generation.rag_pipeline import RAGPipeline

def check_ollama():
    """Pre-flight: verify Ollama is running."""
    try:
        import httpx
        r = httpx.get("http://localhost:11434/api/tags", timeout=5.0)
        r.raise_for_status()
    except Exception as e:
        print("ERROR: Ollama is not running.")
        print(f"  Start it with: ollama serve")
        print(f"  Then pull the model: ollama pull {config.LLM_MODEL}")
        print(f"  Details: {e}")
        sys.exit(1)

def count_valid_lines(path) -> int:
    """Count valid JSON lines in checkpoint file. Truncates corrupt last line."""
    if not path.exists():
        return 0
    lines = path.read_text(encoding="utf-8").splitlines()
    count = 0
    for line in lines:
        try:
            json.loads(line)
            count += 1
        except json.JSONDecodeError:
            # Assumption: corruption (if any) only occurs at the final line due to per-record flush.
            # If a mid-file corrupt line is found, stop here and truncate — this is a very rare edge case.
            break
    # Truncate file to only valid lines if needed
    if count < len(lines):
        with open(path, "w", encoding="utf-8") as f:
            f.write("\n".join(lines[:count]) + ("\n" if count else ""))
    return count

def main():
    check_ollama()
    config.RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    # Load eval set
    eval_path = config.PROCESSED_DIR / config.QA_GOLD_FILE
    eval_set = DataProcessor.load_jsonl(eval_path)
    print(f"Loaded {len(eval_set)} QA examples")

    # Checkpoint/resume
    predictions_path = config.RESULTS_DIR / "qa_predictions.jsonl"
    already_done = count_valid_lines(predictions_path)
    if already_done > 0:
        print(f"Resuming from checkpoint: {already_done}/{len(eval_set)} already done")
    remaining = eval_set[already_done:]

    # Load retriever
    index_path = config.INDEX_DIR / config.INDEX_FILE
    metadata_path = config.INDEX_DIR / config.METADATA_FILE
    print("Loading embedding model and index...")
    embedder = Embedder()
    embedder.load_model()
    retriever = Retriever(embedder)
    retriever.load_index(index_path, metadata_path)
    pipeline = RAGPipeline(retriever)

    # Batch-retrieve all remaining questions at once (single embedding + index.search call)
    print(f"Batch-retrieving {len(remaining)} questions...")
    t0 = time.time()
    remaining_questions = [e['question'] for e in remaining]
    all_retrieved = retriever.batch_retrieve(remaining_questions, top_k=config.TOP_K_RETRIEVAL)
    print(f"  Retrieval done in {time.time()-t0:.1f}s")

    print(f"Generating {len(remaining)} answers...")

    with open(predictions_path, "a", encoding="utf-8") as f:
        for i, (example, retrieved_chunks) in enumerate(tqdm(zip(remaining, all_retrieved), total=len(remaining), desc="Generating answers")):
            global_i = already_done + i + 1
            try:
                context_used, context_chunks = pipeline.assemble_context(retrieved_chunks)
                answer = pipeline.generate(example['question'], context_used)
                record = {
                    "query_id": example['query_id'],
                    "question": example['question'],
                    "predicted": answer,
                    "expected": example['answer'],
                    "sources": [c['source'] for c in context_chunks],
                    "retrieved_chunks": [dict(c) for c in context_chunks],
                }
            except Exception as e:
                record = {
                    "query_id": example['query_id'],
                    "question": example['question'],
                    "predicted": "",
                    "expected": example['answer'],
                    "sources": [],
                    "retrieved_chunks": [],
                    "error": str(e),
                }
                print(f"  ERROR at query {global_i}: {e}")

            f.write(json.dumps(record, ensure_ascii=False) + "\n")
            f.flush()

    total = already_done + len(remaining)
    print(f"\n✓ Generation complete. {total} predictions written to {predictions_path}")

if __name__ == '__main__':
    main()
