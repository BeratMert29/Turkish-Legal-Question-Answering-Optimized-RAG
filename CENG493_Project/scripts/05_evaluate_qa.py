"""Evaluate QA metrics and run hallucination analysis."""
import json
import sys
from pathlib import Path
_project_root = str(Path(__file__).parent.parent)
if _project_root not in sys.path:
    sys.path.append(_project_root)
import config
from data.data_processor import DataProcessor
from evaluation.qa_metrics import compute_all_qa_metrics
from evaluation.hallucination import stratified_sample, run_hallucination_analysis

def main():
    config.RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    # Load predictions
    predictions_path = config.RESULTS_DIR / "qa_predictions.jsonl"
    print(f"Loading predictions from {predictions_path}")
    predictions = DataProcessor.load_jsonl(predictions_path)

    # Separate errors
    errors = [p for p in predictions if "error" in p]
    valid = [p for p in predictions if "error" not in p]
    if errors:
        print(f"WARNING: {len(errors)} predictions had errors and were excluded from metrics")
    print(f"Valid predictions: {len(valid)}")

    # QA metrics
    print("\nComputing QA metrics...")
    qa_input = [{"predicted": p["predicted"], "expected": p["expected"]} for p in valid]
    qa_metrics = compute_all_qa_metrics(qa_input)
    qa_metrics["error_count"] = len(errors)

    print(f"\n=== QA Metrics ===")
    for k, v in qa_metrics.items():
        print(f"  {k}: {v:.4f}" if isinstance(v, float) else f"  {k}: {v}")

    # Save QA results
    qa_results_path = config.RESULTS_DIR / "qa_results.json"
    with open(qa_results_path, "w", encoding="utf-8") as f:
        json.dump(qa_metrics, f, ensure_ascii=False, indent=2)
    print(f"\n  Saved: {qa_results_path}")

    # Hallucination analysis
    print("\nRunning hallucination analysis...")

    # Load full retrieval results (for building context in faithfulness eval)
    full_results_path = config.RESULTS_DIR / "retrieval_full_results.json"
    if full_results_path.exists():
        with open(full_results_path, "r", encoding="utf-8") as f:
            retrieved_results = json.load(f)
    else:
        print("WARNING: retrieval_full_results.json not found; using retrieved_chunks from predictions")
        retrieved_results = {p["query_id"]: p.get("retrieved_chunks", []) for p in valid}

    # Load NLI model
    print("Loading NLI model: cross-encoder/nli-deberta-v3-small (~180 MB, first run downloads)")
    from sentence_transformers import CrossEncoder
    nli_model = CrossEncoder("cross-encoder/nli-deberta-v3-small")

    # Stratified sample + hallucination analysis
    sample = stratified_sample(valid, config.HALLUCINATION_SAMPLE_SIZE)
    print(f"Stratified sample: hits={len(sample['hits'])}, partial={len(sample['partial'])}, misses={len(sample['misses'])}")

    hall_results = run_hallucination_analysis(sample, retrieved_results, nli_model)

    summary = hall_results["summary"]
    print(f"\n=== Hallucination Analysis ===")
    print(f"  Total analyzed: {summary['total']}")
    print(f"  Faithful: {summary['faithful_count']} ({summary['faithful_rate']:.2%})")
    print(f"  By category: {summary['by_category']}")

    # Save
    hall_path = config.RESULTS_DIR / "hallucination_results.json"
    with open(hall_path, "w", encoding="utf-8") as f:
        json.dump(hall_results, f, ensure_ascii=False, indent=2)
    print(f"\n  Saved: {hall_path}")

    print("\n✓ Evaluation complete")

if __name__ == '__main__':
    main()
