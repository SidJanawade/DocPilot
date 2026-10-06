"""Load markdown docs and split them into heading-aware, overlapping chunks."""
from __future__ import annotations
import re
from dataclasses import dataclass
from pathlib import Path


@dataclass
class Chunk:
    id: str
    doc: str
    heading: str
    text: str


def split_sections(md: str):
    heading, buf = "Introduction", []
    for line in md.splitlines():
        m = re.match(r"^#{1,6}\s+(.*)", line)
        if m:
            if "".join(buf).strip():
                yield heading, "\n".join(buf).strip()
            heading, buf = m.group(1).strip(), []
        else:
            buf.append(line)
    if "".join(buf).strip():
        yield heading, "\n".join(buf).strip()


def window(words, size=120, overlap=20):
    if len(words) <= size:
        yield words
        return
    step = size - overlap
    for i in range(0, len(words), step):
        yield words[i : i + size]
        if i + size >= len(words):
            break


def load_chunks(docs_dir: Path, size=120, overlap=20) -> list[Chunk]:
    chunks: list[Chunk] = []
    for path in sorted(Path(docs_dir).glob("**/*.md")):
        for heading, body in split_sections(path.read_text(encoding="utf-8")):
            for w in window(body.split(), size, overlap):
                chunks.append(Chunk(f"{path.stem}:{len(chunks)}", path.name, heading, " ".join(w)))
    return chunks
