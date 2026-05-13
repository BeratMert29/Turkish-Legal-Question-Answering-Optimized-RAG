#!/usr/bin/env python3
"""Build cross-reference graph from FAISS metadata; writes index/graph.json."""

import argparse
import json
import logging
import sys
import time
from pathlib import Path

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

import config  # noqa: E402
from retrieval.graph_builder import (  # noqa: E402
    build_graph_from_metadata,
    graph_stats,
    save_graph,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(name)s %(levelname)s  %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--metadata", type=Path,
        default=config.INDEX_DIR / config.METADATA_FILE,
        help="Path to metadata.jsonl (default: index/metadata.jsonl)",
    )
    ap.add_argument(
        "--output", type=Path,
        default=config.INDEX_DIR / getattr(config, "GRAPH_FILE", "graph.json"),
        help="Output graph path (default: index/graph.json)",
    )
    ap.add_argument(
        "--dry-run", action="store_true",
        help="Build graph in memory but skip writing to disk",
    )
    args = ap.parse_args()

    log.info("Loading metadata from %s", args.metadata)
    with open(args.metadata, "r", encoding="utf-8") as f:
        metadata = [json.loads(line) for line in f if line.strip()]
    log.info("Loaded %d chunks", len(metadata))

    t0 = time.time()
    graph = build_graph_from_metadata(metadata)
    elapsed = time.time() - t0
    log.info("Graph built in %.1fs", elapsed)

    stats = graph_stats(graph)
    log.info("Stats:\n%s", json.dumps(stats, indent=2, ensure_ascii=False))

    if args.dry_run:
        log.info("--dry-run: graph NOT written to disk")
    else:
        save_graph(graph, args.output)
        log.info("Done → %s", args.output)


if __name__ == "__main__":
    main()
