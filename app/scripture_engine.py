from __future__ import annotations

import gzip
import json
import os
import re
from pathlib import Path
from typing import Any

import numpy as np

_TOKEN = re.compile(r"[a-z0-9']+")


class SemanticEngineUnavailable(RuntimeError):
    pass


class ScriptureSemanticEngine:
    """Latent Semantic Analysis retrieval over verified Bible verses.

    Candidate text always originates in the configured Bible corpus. The model
    is distributional (term co-occurrence in context), not direct keyword rank.
    Manually curated relations can overlay richer doctrine/parallel metadata.
    """

    def __init__(self, data_dir: Path | None = None) -> None:
        root = data_dir or Path(os.getenv("SCRIPTURE_DATA_DIR", "app/data"))
        corpus_path = root / "scripture_corpus.json.gz"
        model_path = root / "semantic_model.json"
        if not corpus_path.is_file() or not model_path.is_file():
            raise SemanticEngineUnavailable("Semantic index has not been built from Bible data.")
        with gzip.open(corpus_path, "rt", encoding="utf-8") as stream:
            self.verses = json.load(stream)
        model = json.loads(model_path.read_text(encoding="utf-8"))
        if model.get("version") != 1 or len(model.get("records", [])) != len(self.verses):
            raise SemanticEngineUnavailable("Semantic index does not match its Bible corpus.")
        self.vocabulary: dict[str, int] = model["vocabulary"]
        arrays_path = root / "semantic_model.npz"
        if not arrays_path.is_file():
            raise SemanticEngineUnavailable("Semantic model arrays are missing; rebuild the semantic index.")
        with np.load(arrays_path) as arrays:
            self.idf = arrays["idf"].copy()
            self.components = arrays["components"].copy()
            vectors = arrays["vectors"].copy()
        self.dimensions = int(model["dimensions"])
        self.references = model["records"]
        if (
            self.components.shape != (self.dimensions, len(self.idf))
            or vectors.shape != (len(self.references), self.dimensions)
        ):
            raise SemanticEngineUnavailable("Semantic model dimensions are invalid.")
        self.by_reference = {
            (r["bookId"], int(r["chapter"]), int(r["verse"])): i
            for i, r in enumerate(self.references)
        }
        self.vectors = vectors
        daily_path = root / "daily_verses.json"
        self.daily_verses = json.loads(daily_path.read_text(encoding="utf-8")) if daily_path.is_file() else []
        self.curated = self._load_curated(root)

    def _load_curated(self, root: Path) -> dict[tuple[str, int, int], list[dict[str, Any]]]:
        path = root / "curated_connections.json"
        if not path.exists():
            return {}
        payload = json.loads(path.read_text(encoding="utf-8"))
        result: dict[tuple[str, int, int], list[dict[str, Any]]] = {}
        for edge in payload if isinstance(payload, list) else []:
            try:
                source = (edge["source"]["bookId"], int(edge["source"]["chapter"]), int(edge["source"]["verse"]))
                target = (edge["target"]["bookId"], int(edge["target"]["chapter"]), int(edge["target"]["verse"]))
                relation = str(edge["relationType"])
                if source in self.by_reference and target in self.by_reference and source != target and relation in {"doctrine", "parallel", "explanation", "event", "person", "topic", "context"}:
                    result.setdefault(source, []).append({"target": target, "relationType": relation})
            except (KeyError, TypeError, ValueError):
                continue
        return result

    def _encode_query(self, text: str) -> np.ndarray:
        counts: dict[int, float] = {}
        words = _TOKEN.findall(text.lower())
        tokens = list(words)
        tokens.extend(f"{a} {b}" for a, b in zip(words, words[1:]))
        for token in tokens:
            index = self.vocabulary.get(token)
            if index is not None:
                counts[index] = counts.get(index, 0.0) + 1.0
        if not counts:
            return np.zeros(self.dimensions, dtype=np.float32)
        total = max(1, len(tokens))
        weighted = np.zeros(len(self.idf), dtype=np.float32)
        for index, count in counts.items():
            weighted[index] = (1.0 + np.log(count)) * float(self.idf[index]) / total
        latent = self.components @ weighted
        norm = float(np.linalg.norm(latent))
        return latent / norm if norm > 1e-8 else latent

    def related(self, source: tuple[str, int, int], limit: int = 12) -> list[dict[str, Any]]:
        source_index = self.by_reference.get(source)
        if source_index is None:
            raise KeyError("Source verse is not present in the indexed Bible dataset.")
        source_vector = np.asarray(self.vectors[source_index], dtype=np.float32)
        scores = np.asarray(self.vectors, dtype=np.float32) @ source_vector
        scores[source_index] = -1.0
        order = np.argpartition(scores, -min(100, len(scores)))[-min(100, len(scores)):]
        order = order[np.argsort(scores[order])[::-1]]
        curated_edges = self.curated.get(source, [])
        curated_by_target = {edge["target"]: edge["relationType"] for edge in curated_edges}
        output = []
        used = set()
        for index in order:
            score = float(scores[index])
            if score < 0.18:
                break
            row = self.references[int(index)]
            reference = (row["bookId"], int(row["chapter"]), int(row["verse"]))
            if reference in used:
                continue
            used.add(reference)
            output.append({"bookId": reference[0], "chapter": reference[1], "verse": reference[2], "score": round(score, 4), "relationType": curated_by_target.get(reference, "semantic_theme")})
            if len(output) >= limit:
                break
        for edge in curated_edges:
            ref = edge["target"]
            if ref in used or len(output) >= limit:
                continue
            row_index = self.by_reference[ref]
            row = self.references[row_index]
            output.append({"bookId": ref[0], "chapter": ref[1], "verse": ref[2], "score": 1.0, "relationType": edge["relationType"]})
        return output

    def daily_reference(self, day_key: str) -> dict[str, Any]:
        # Stable per calendar day, selecting a verse known in the same verified corpus.
        import hashlib
        if not self.daily_verses:
            raise SemanticEngineUnavailable("Verified daily verse references are missing.")
        digest = hashlib.sha256(day_key.encode("utf-8")).digest()
        index = int.from_bytes(digest[:8], "big") % len(self.daily_verses)
        return self.daily_verses[index]
