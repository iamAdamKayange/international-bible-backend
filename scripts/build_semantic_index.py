from __future__ import annotations

import argparse
import gzip
import json
import re
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(description="Build a verified verse corpus and LSA semantic index from the app Bible JSON.")
    parser.add_argument("--source", required=True, help="Path to the existing en_kjv.json file")
    parser.add_argument("--output", default="data", help="Output directory")
    args = parser.parse_args()
    source = Path(args.source)
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    books = json.loads(source.read_text(encoding="utf-8"))
    records = []
    for book in books:
        book_id = book.get("abbrev")
        if not isinstance(book_id, str) or not book_id:
            continue
        for chapter_number, verses in enumerate(book.get("chapters", []), start=1):
            if not isinstance(verses, list):
                continue
            for verse_number, text in enumerate(verses, start=1):
                if isinstance(text, str) and text.strip():
                    records.append({"bookId": book_id, "chapter": chapter_number, "verse": verse_number, "text": text.strip()})
    if len(records) < 1000:
        raise SystemExit("Source Bible dataset is incomplete; refusing to build an index.")
    corpus = output / "scripture_corpus.json.gz"
    with gzip.open(corpus, "wt", encoding="utf-8") as stream:
        json.dump(records, stream, ensure_ascii=False, separators=(",", ":"))

    # A reviewed set of familiar passages, filtered against the real source.
    daily_candidates = [
        ("gn", 1, 1), ("js", 1, 9), ("sl", 23, 1), ("sl", 27, 1),
        ("sl", 46, 1), ("sl", 91, 1), ("sl", 119, 105), ("pv", 3, 5),
        ("pv", 3, 6), ("is", 40, 31), ("is", 41, 10), ("mt", 5, 16),
        ("mt", 6, 33), ("mt", 11, 28), ("jo", 1, 1), ("jo", 3, 16),
        ("jo", 14, 6), ("rm", 8, 28), ("rm", 12, 2), ("1co", 10, 13),
        ("1co", 13, 4), ("2co", 5, 7), ("gl", 5, 22), ("ef", 2, 8),
        ("fp", 4, 6), ("fp", 4, 13), ("hb", 11, 1), ("tg", 1, 5),
        ("1jo", 4, 8), ("ap", 21, 4),
    ]
    verse_ids = {(row["bookId"], row["chapter"], row["verse"]) for row in records}
    daily_verses = [{"bookId": book, "chapter": chapter, "verse": verse}
                    for book, chapter, verse in daily_candidates
                    if (book, chapter, verse) in verse_ids]
    if len(daily_verses) < 10:
        raise SystemExit("Too few selected daily passages exist in the source Bible; refusing to publish them.")
    (output / "daily_verses.json").write_text(json.dumps(daily_verses, separators=(",", ":")), encoding="utf-8")

    # Latent Semantic Analysis learns concept co-occurrence from the existing
    # verse corpus. It is intentionally not a keyword overlap ranker.
    from sklearn.decomposition import TruncatedSVD
    from sklearn.feature_extraction.text import TfidfVectorizer
    import numpy as np

    texts = [row["text"] for row in records]
    vectorizer = TfidfVectorizer(
        lowercase=True, strip_accents="unicode", stop_words="english",
        max_features=14000, min_df=2, max_df=0.92,
        ngram_range=(1, 2), sublinear_tf=True, dtype=np.float32,
    )
    matrix = vectorizer.fit_transform(texts)
    dimensions = min(96, matrix.shape[0] - 1, matrix.shape[1] - 1)
    if dimensions < 16:
        raise SystemExit("Bible corpus did not yield enough semantic features.")
    svd = TruncatedSVD(n_components=dimensions, n_iter=7, random_state=17)
    vectors = svd.fit_transform(matrix).astype(np.float32)
    norms = np.linalg.norm(vectors, axis=1, keepdims=True)
    vectors /= np.maximum(norms, 1e-8)
    components = svd.components_.astype(np.float32)
    vocabulary = vectorizer.vocabulary_
    model = {
        "version": 1,
        "translationId": "en_kjv",
        "dimensions": dimensions,
        "vocabulary": {key: int(value) for key, value in vocabulary.items()},
        "idf": vectorizer.idf_.tolist(),
        "records": [{k: row[k] for k in ("bookId", "chapter", "verse")} for row in records],
    }
    (output / "semantic_model.json").write_text(json.dumps(model, separators=(",", ":")), encoding="utf-8")
    np.savez_compressed(output / "semantic_model.npz", idf=vectorizer.idf_.astype(np.float32), components=components, vectors=vectors)
    print(f"Built {len(records)} verse records and {dimensions}-dimension semantic vectors from {source}")


if __name__ == "__main__":
    main()
