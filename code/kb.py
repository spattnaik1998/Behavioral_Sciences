"""
Knowledge base: load ALL literature PDFs, chunk exactly as in the paper
(RecursiveCharacterTextSplitter, 500 / 50), embed with text-embedding-3-small,
index with FAISS (exact L2; OpenAI vectors are unit-norm so this equals cosine).

Fixes the original notebook bug where only the first uploaded PDF
("Susceptibility of Mental Accounting Principles") was indexed.

PDF selection (see config.kb_pdfs / config.original_kb_pdfs):
  revised KB  : every PDF in knowledge_base/, or - if that is empty - every PDF in ../Materials/
  original KB : knowledge_base/original/*.pdf, or the "Mental Accounting" PDF in ../Materials/

Outputs (results/02_knowledge_base/):
  kb_<name>_documents.csv   file, pages, characters, chunks
  kb_<name>_chunks.jsonl    every chunk with source + page (for auditability)
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

import config as C

OUT = C.RESULTS_DIR / "02_knowledge_base"


@dataclass
class Chunk:
    chunk_id: int
    source: str
    page: int
    text: str


def load_and_chunk(pdfs) -> tuple[list[Chunk], pd.DataFrame]:
    from pypdf import PdfReader
    from langchain_text_splitters import RecursiveCharacterTextSplitter

    splitter = RecursiveCharacterTextSplitter(chunk_size=C.CHUNK_SIZE, chunk_overlap=C.CHUNK_OVERLAP)
    chunks, rows = [], []
    if isinstance(pdfs, (str, Path)):
        pdfs = sorted(Path(pdfs).glob("*.pdf"))
    for pdf in pdfs:
        reader = PdfReader(str(pdf))
        n_chunks = n_chars = 0
        for pno, page in enumerate(reader.pages):
            text = page.extract_text() or ""
            n_chars += len(text)
            # same behaviour as LangChain's split_documents on PyPDFLoader pages
            for piece in splitter.split_text(text):
                chunks.append(Chunk(len(chunks), pdf.name, pno + 1, piece))
                n_chunks += 1
        rows.append({"file": pdf.name, "pages": len(reader.pages), "characters": n_chars, "chunks": n_chunks})
    return chunks, pd.DataFrame(rows)


class KnowledgeBase:
    def __init__(self, name: str, chunks: list[Chunk], vectors: np.ndarray):
        import faiss
        self.name, self.chunks = name, chunks
        self.vectors = vectors.astype("float32")
        self.index = faiss.IndexFlatL2(self.vectors.shape[1])
        self.index.add(self.vectors)

    @classmethod
    def build(cls, client, pdfs, name: str) -> "KnowledgeBase | None":
        chunks, docs = load_and_chunk(pdfs)
        if not chunks:
            print(f"[kb:{name}] no PDFs / no extractable text in {pdfs}")
            return None
        OUT.mkdir(parents=True, exist_ok=True)
        docs.to_csv(OUT / f"kb_{name}_documents.csv", index=False)
        with (OUT / f"kb_{name}_chunks.jsonl").open("w", encoding="utf-8") as fh:
            for c in chunks:
                fh.write(json.dumps(c.__dict__, ensure_ascii=False) + "\n")
        vecs = client.embed([c.text for c in chunks], name=f"kb_{name}")
        print(f"[kb:{name}] {len(docs)} PDFs, {len(chunks)} chunks indexed")
        return cls(name, chunks, vecs)

    def search(self, query_vecs: np.ndarray, k: int) -> np.ndarray:
        _, idx = self.index.search(query_vecs.astype("float32"), k)
        return idx

    def context(self, ids) -> str:
        # LangChain's default document formatting in the original chain was the
        # repr of the Document list; we use plain text joined by blank lines.
        return "\n\n".join(self.chunks[i].text for i in ids)

    def context_langchain_repr(self, ids) -> str:
        """How the original LangChain chain rendered {context}: the str() of the
        list of retrieved Document objects. Used only for the original-replication run."""
        docs = [f"Document(metadata={{'source': {self.chunks[i].source!r}, 'page': {self.chunks[i].page - 1}}}, "
                f"page_content={self.chunks[i].text!r})" for i in ids]
        return "[" + ", ".join(docs) + "]"

    def sources(self, ids) -> list[str]:
        return [f"{self.chunks[i].source}#p{self.chunks[i].page}" for i in ids]


def retrieval_diagnostics(kb: KnowledgeBase, retrieved: dict[str, list[int]]) -> pd.DataFrame:
    """Share of retrieved chunks coming from each PDF (shows what the RAG actually reads)."""
    counts = pd.Series([kb.chunks[i].source for ids in retrieved.values() for i in ids]).value_counts()
    df = counts.rename("n_retrieved").to_frame()
    df["share"] = (df["n_retrieved"] / df["n_retrieved"].sum()).round(4)
    OUT.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUT / f"retrieval_sources_{kb.name}.csv")
    return df
