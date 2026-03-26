from dataclasses import dataclass, asdict
from typing import Iterator
import json
import warnings
import pathlib

import pandas as pd
import config


@dataclass
class CorpusChunk:
    chunk_id: str   # f"{source}_{doc_id}_{chunk_index}"
    doc_id: str
    text: str
    source: str
    char_len: int


@dataclass
class QAExample:
    query_id: str
    question: str
    answer: str
    context: str    # "" for test/train rows (null in CSV)
    source: str
    data_type: str


class DataProcessor:
    def __init__(self, csv_path):
        self.csv_path = csv_path
        self._df: pd.DataFrame | None = None

    # ------------------------------------------------------------------
    # Loading / validation
    # ------------------------------------------------------------------

    def load_and_validate(self) -> dict:
        """Load CSV, check required columns, return summary dict."""
        self._df = pd.read_csv(self.csv_path, encoding="utf-8")

        if self._df.empty:
            raise ValueError(f"CSV is empty: {self.csv_path}")

        required_columns = {"id", "question", "answer", "context", "source", "data_type", "score", "split"}
        missing = required_columns - set(self._df.columns)
        if missing:
            raise ValueError(f"CSV is missing columns: {missing}")

        summary = {
            "total_rows": len(self._df),
            "columns": list(self._df.columns),
            "split_counts": self._df["split"].value_counts().to_dict(),
            "null_context_count": int(self._df["context"].isna().sum()),
        }
        return summary

    def _ensure_loaded(self):
        if self._df is None:
            self.load_and_validate()

    # ------------------------------------------------------------------
    # Row accessors
    # ------------------------------------------------------------------

    def get_corpus_rows(self) -> pd.DataFrame:
        """Rows where split == 'kaggle' (have context)."""
        self._ensure_loaded()
        return self._df[self._df["split"] == "kaggle"].reset_index(drop=True)

    def get_qa_split(self, split: str) -> pd.DataFrame:
        self._ensure_loaded()
        return self._df[self._df["split"] == split].reset_index(drop=True)

    # ------------------------------------------------------------------
    # Chunking
    # ------------------------------------------------------------------

    def chunk_text(self, text: str, doc_id: str, source: str) -> list[CorpusChunk]:
        """Sliding-window chunking.  Returns [] for short texts.

        Texts shorter than CHUNK_SIZE are emitted as a single chunk.
        (Texts shorter than CORPUS_DOC_MIN_CHARS are skipped entirely.)
        """
        if len(text) < config.CORPUS_DOC_MIN_CHARS:
            return []

        step = config.CHUNK_SIZE - config.CHUNK_OVERLAP
        chunks: list[CorpusChunk] = []

        for i, start in enumerate(range(0, len(text), step)):
            chunk_text = text[start: start + config.CHUNK_SIZE]
            if len(chunk_text) < config.CORPUS_DOC_MIN_CHARS:
                continue
            chunks.append(CorpusChunk(
                chunk_id=f"{source}_{doc_id}_{i}",
                doc_id=doc_id,
                text=chunk_text,
                source=source,
                char_len=len(chunk_text),
            ))

        return chunks

    # ------------------------------------------------------------------
    # Corpus builder (generator)
    # ------------------------------------------------------------------

    def build_corpus_chunks(self) -> Iterator[CorpusChunk]:
        """Generator — yields CorpusChunk objects for every corpus row."""
        for row in self.get_corpus_rows().itertuples(index=False):
            context = row.context if pd.notna(row.context) else ""
            if not context:
                continue
            for chunk in self.chunk_text(str(context), str(row.id), str(row.source)):
                yield chunk

    # ------------------------------------------------------------------
    # QA set builders
    # ------------------------------------------------------------------

    def _rows_to_qa_examples(self, df: pd.DataFrame) -> list[QAExample]:
        examples: list[QAExample] = []
        for row in df.itertuples(index=False):
            raw = row._asdict()
            context_val = raw.get("context", "")
            context_str = "" if pd.isna(context_val) else str(context_val)

            source_val = raw.get("source", "")
            source_str = "" if pd.isna(source_val) else str(source_val)

            data_type_val = raw.get("data_type", "")
            data_type_str = "" if pd.isna(data_type_val) else str(data_type_val)

            question_val = raw.get("question", "")
            question_str = str(question_val) if pd.notna(question_val) else ""

            answer_val = raw.get("answer", "")
            answer_str = str(answer_val) if pd.notna(answer_val) else ""

            examples.append(QAExample(
                query_id=str(raw.get("id", "")),
                question=question_str,
                answer=answer_str,
                context=context_str,
                source=source_str,
                data_type=data_type_str,
            ))
        return examples

    def build_qa_eval_set(self) -> list[QAExample]:
        """Build QA eval (test split), capped at QA_EVAL_EXPECTED examples."""
        df = self.get_qa_split("test")
        if len(df) > config.QA_EVAL_EXPECTED:
            df = df.iloc[: config.QA_EVAL_EXPECTED]
        examples = self._rows_to_qa_examples(df)
        return examples

    def build_qa_train_set(self) -> list[QAExample]:
        """Build QA train set (train split)."""
        df = self.get_qa_split("train")
        return self._rows_to_qa_examples(df)

    # ------------------------------------------------------------------
    # Ground-truth relevance linker
    # ------------------------------------------------------------------

    @staticmethod
    def build_relevant_chunk_map(
        corpus_chunks: list["CorpusChunk"],
        qa_examples: list["QAExample"],
        retriever=None,
        top_k_relevant: int = 20,
        score_threshold: float = 0.5,
    ) -> dict[str, list[str]]:
        """
        Returns {query_id: [chunk_id, ...]} for chunks relevant to each QA example.

        When *retriever* is provided, uses semantic similarity: encodes each
        answer/context and retrieves the most similar corpus chunks via the
        same FAISS index.  This is far more reliable than token overlap when
        gold context labels are unavailable (e.g. test split has no source).

        Falls back to an improved token-overlap heuristic (Turkish-aware
        normalisation, stopword removal, 50 % threshold) when no retriever
        is supplied.
        """
        if retriever is not None:
            return DataProcessor._build_semantic_relevant_map(
                qa_examples, retriever, top_k_relevant, score_threshold,
            )
        return DataProcessor._build_token_relevant_map(corpus_chunks, qa_examples)

    @staticmethod
    def _build_semantic_relevant_map(
        qa_examples: list["QAExample"],
        retriever,
        top_k: int,
        score_threshold: float,
    ) -> dict[str, list[str]]:
        """Find relevant chunks by embedding the answer and searching the index."""
        ref_texts: list[str] = []
        valid_indices: list[int] = []
        for i, qa in enumerate(qa_examples):
            ref_text = qa.context if qa.context else qa.answer
            if ref_text:
                ref_texts.append(ref_text)
                valid_indices.append(i)

        all_results = retriever.batch_retrieve(ref_texts, top_k=top_k)

        result: dict[str, list[str]] = {qa.query_id: [] for qa in qa_examples}
        for idx, retrieved in zip(valid_indices, all_results):
            qa = qa_examples[idx]
            result[qa.query_id] = [
                c["chunk_id"] for c in retrieved
                if c["score"] >= score_threshold
            ]
        return result

    @staticmethod
    def _build_token_relevant_map(
        corpus_chunks: list["CorpusChunk"],
        qa_examples: list["QAExample"],
    ) -> dict[str, list[str]]:
        """
        Improved token-overlap fallback with Turkish-aware normalisation,
        stopword removal, and a 50 % threshold.
        """
        from collections import defaultdict
        import unicodedata

        _STOPWORDS = frozenset({
            'bir', 've', 'bu', 'de', 'da', 'ile', 'için', 'olan', 'olarak',
            'veya', 'ise', 'gibi', 'her', 'daha', 'en', 'o', 'ne', 'ya',
            'ki', 'mi', 'mu', 'dir', 'den', 'dan', 'ler', 'lar',
            'nin', 'nun', 'in', 'un', 'ın', 'ya', 'ye', 'ta', 'te',
            'kadar', 'sonra', 'dair', 'göre', 'başka', 'ancak', 'ayrıca',
        })

        def _normalize(text: str) -> set[str]:
            text = text.replace('\u0130', 'i').replace('I', '\u0131')
            text = text.lower()
            text = unicodedata.normalize('NFC', text)
            return set(text.split()) - _STOPWORDS

        token_index: dict[str, list[int]] = defaultdict(list)
        chunk_token_sets: list[set] = []
        chunk_ids: list[str] = []

        for i, chunk in enumerate(corpus_chunks):
            tokens = _normalize(chunk.text)
            chunk_token_sets.append(tokens)
            chunk_ids.append(chunk.chunk_id)
            for tok in tokens:
                token_index[tok].append(i)

        result: dict[str, list[str]] = {}
        for qa in qa_examples:
            ref_text = qa.context if qa.context else qa.answer
            if not ref_text:
                result[qa.query_id] = []
                continue
            ref_tokens = _normalize(ref_text)
            if len(ref_tokens) < 3:
                result[qa.query_id] = []
                continue

            candidate_hits: dict[int, int] = defaultdict(int)
            for tok in ref_tokens:
                for cidx in token_index.get(tok, []):
                    candidate_hits[cidx] += 1

            threshold = max(0.5 * len(ref_tokens), 5)
            relevant = [
                chunk_ids[cidx]
                for cidx, hit_count in candidate_hits.items()
                if hit_count >= threshold
            ]
            result[qa.query_id] = relevant
        return result

    # ------------------------------------------------------------------
    # JSONL I/O
    # ------------------------------------------------------------------

    @staticmethod
    def save_jsonl(items, path) -> int:
        """Write items to a JSONL file.  Creates parent dirs.  Returns count."""
        p = pathlib.Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        count = 0
        with p.open("w", encoding="utf-8") as f:
            for item in items:
                record = asdict(item) if hasattr(item, "__dataclass_fields__") else item
                f.write(json.dumps(record, ensure_ascii=False) + "\n")
                count += 1
        return count

    @staticmethod
    def load_jsonl(path) -> list[dict]:
        """Load a JSONL file and return a list of raw dicts."""
        p = pathlib.Path(path)
        MAX_JSONL_BYTES = 2 * 1024 ** 3  # 2 GB
        file_size = p.stat().st_size
        if file_size > MAX_JSONL_BYTES:
            raise ValueError(
                f"JSONL file too large to load: {file_size / 1024**3:.1f} GB > 2 GB limit: {p}"
            )
        results: list[dict] = []
        with p.open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    results.append(json.loads(line))
        return results
