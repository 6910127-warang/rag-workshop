"""ค้นคืน chunk ที่เกี่ยวข้องกับคำถาม (เวิร์กช็อป 2 และ 4) ทำงานในเครื่องทั้งหมด ไม่เรียก API

python -m rag.retrieve "คำถาม" -k 4 [--method tfidf|hybrid] [--include-superseded] [--json]
"""
import argparse
import json
import sys

import numpy as np
from pythainlp.tokenize import word_tokenize
from rank_bm25 import BM25Okapi
from sklearn.feature_extraction.text import TfidfVectorizer

from rag.chunk import chunk_docs


# พจนานุกรมคำพ้อง สร้างจากคำถามที่พลาดในชุด dev เท่านั้น (Q35: "อีวี" ไม่ตรงกับ "รถยนต์ไฟฟ้า" ในเอกสาร)
SYNONYMS = {
    "อีวี": ["รถยนต์ไฟฟ้า"],
}


def expand_query(query: str) -> str:
    extra = [w for k, ws in SYNONYMS.items() if k in query for w in ws]
    return f"{query} {' '.join(extra)}" if extra else query


def tokenize(text: str) -> list[str]:
    return [t for t in word_tokenize(text, engine="newmm") if t.strip()]


def _minmax(x: np.ndarray) -> np.ndarray:
    span = x.max() - x.min()
    return (x - x.min()) / span if span > 0 else np.zeros_like(x)


class Retriever:
    def __init__(self, chunks: list[dict] | None = None, method: str = "tfidf"):
        """method="tfidf": TF-IDF ระดับตัวอักษร (W2) · method="hybrid": วิธีที่ปรับปรุงแล้ว (W4)"""
        if method not in ("tfidf", "hybrid"):
            raise ValueError(f"method={method!r} ไม่รองรับ (tfidf | hybrid)")
        self.method = method
        self.chunks = chunk_docs() if chunks is None else chunks
        # ภาษาไทยไม่เว้นวรรคระหว่างคำ จึงใช้ n-gram ระดับตัวอักษรแทนการตัดคำ
        self.vectorizer = TfidfVectorizer(analyzer="char_wb", ngram_range=(2, 4))
        self.matrix = self.vectorizer.fit_transform(
            f"{c['title']} {c['section']} {c['text']}" for c in self.chunks
        )
        if method == "hybrid":
            self.bm25 = BM25Okapi([tokenize(f"{c['title']} {c['section']} {c['text']}") for c in self.chunks])
        self.is_active = np.array([c["status"] == "active" for c in self.chunks])

    def search(self, query: str, k: int = 4, include_superseded: bool = True) -> list[dict]:
        """คืน chunk k อันที่คะแนนสูงสุด เรียงจากมากไปน้อย แต่ละอันเพิ่มช่อง "score" """
        # เวกเตอร์ TF-IDF ถูก normalize แล้ว ผลคูณจุดจึงเท่ากับ cosine similarity
        if self.method == "hybrid":
            query = expand_query(query)
        scores = (self.matrix @ self.vectorizer.transform([query]).T).toarray().ravel()
        if self.method == "hybrid":
            bm = np.asarray(self.bm25.get_scores(tokenize(query)), dtype=float)
            scores = 0.5 * _minmax(scores) + 0.5 * _minmax(bm)
        if not include_superseded:
            scores = np.where(self.is_active, scores, -1.0)
        order = np.argsort(-scores, kind="stable")
        if not include_superseded:
            order = [i for i in order if self.is_active[i]]
        return [{**self.chunks[i], "score": float(scores[i])} for i in order[:k]]


def main():
    ap = argparse.ArgumentParser(description="ค้นคืน chunk ที่เกี่ยวข้องกับคำถาม")
    ap.add_argument("query")
    ap.add_argument("-k", type=int, default=4)
    ap.add_argument("--method", default="hybrid", choices=["tfidf", "hybrid"])
    ap.add_argument("--include-superseded", action="store_true", help="รวมเอกสารฉบับเดิม (superseded) ด้วย (ค่าเริ่มต้น: กรองออก)")
    ap.add_argument("--json", action="store_true", help="พิมพ์ผลเป็น JSON")
    a = ap.parse_args()
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    res = Retriever(method=a.method).search(a.query, k=a.k, include_superseded=a.include_superseded)
    if a.json:
        print(json.dumps(res, ensure_ascii=False, indent=2))
        return
    for i, r in enumerate(res, 1):
        print(f"{i}. [{r['chunk_id']}] {r['title']} › {r['section']}  score={r['score']:.3f}  status={r['status']}")
        print("   " + r["text"].split("\n", 1)[-1].replace("\n", "\n   "))


if __name__ == "__main__":
    main()
