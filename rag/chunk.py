"""แบ่งเอกสารเป็น chunk พร้อม metadata (เวิร์กช็อป 1)

แต่ละ chunk เป็น dict ที่มีอย่างน้อย:
chunk_id ("<doc_id>#<ลำดับ>"), doc_id, title, section, status, effective_date, text
"""
from rag.ingest import load_docs

MAX_CHARS = 600


def split_sections(body: str, default_title: str) -> list[tuple[str, str]]:
    """แยกเนื้อหาเป็น (ชื่อหัวข้อ, ข้อความ) ตามบรรทัด '## ' · ข้อความก่อน ## แรกใช้ชื่อเอกสารเป็นหัวข้อ"""
    sections, title, lines = [], default_title, []
    for line in body.splitlines():
        if line.startswith("## "):
            sections.append((title, lines))
            title, lines = line[3:].strip(), []
        elif not (line.startswith("# ") and not sections and not lines):
            lines.append(line)
    sections.append((title, lines))
    return [(t, "\n".join(ls).strip()) for t, ls in sections if "\n".join(ls).strip()]


def split_lines(text: str, limit: int) -> list[str]:
    """รวมบรรทัดเข้าด้วยกันไม่ให้เกิน limit ตัวอักษร · บรรทัดที่ยาวเกิน limit เองจะถูกตัดตามความยาว"""
    pieces, cur = [], ""
    for line in text.splitlines():
        while len(line) > limit:
            if cur:
                pieces.append(cur)
                cur = ""
            pieces.append(line[:limit])
            line = line[limit:]
        if cur and len(cur) + 1 + len(line) > limit:
            pieces.append(cur)
            cur = line
        else:
            cur = f"{cur}\n{line}" if cur else line
    if cur:
        pieces.append(cur)
    return pieces


def chunk_docs(docs: list[dict] | None = None, max_chars: int = MAX_CHARS) -> list[dict]:
    """หนึ่งหัวข้อ ## ต่อหนึ่ง chunk · หัวข้อที่ยาวเกิน max_chars ตัดตามบรรทัดเป็นหลาย chunk
    text ขึ้นต้นด้วยชื่อหัวข้อทุก chunk (รวมแล้วไม่เกิน max_chars) และทุก chunk เก็บ metadata ของเอกสาร"""
    docs = load_docs() if docs is None else docs
    chunks = []
    for d in docs:
        n = 0
        for section, text in split_sections(d["body"], d.get("title", d["doc_id"])):
            prefix = f"{section}\n"
            for piece in split_lines(text, max_chars - len(prefix)):
                n += 1
                chunks.append({
                    "chunk_id": f"{d['doc_id']}#{n}",
                    "doc_id": d["doc_id"],
                    "title": d.get("title", ""),
                    "department": d.get("department", ""),
                    "version": d.get("version", ""),
                    "section": section,
                    "status": d.get("status", ""),
                    "effective_date": d.get("effective_date", ""),
                    "text": prefix + piece,
                })
    return chunks


if __name__ == "__main__":
    cs = chunk_docs()
    print(len(cs), "chunks · ยาวสุด", max(len(c["text"]) for c in cs), "ตัวอักษร")
