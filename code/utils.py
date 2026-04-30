from __future__ import annotations

import re
import xml.etree.ElementTree as ET
from pathlib import Path
import json 
from typing import Any

_WS_RE = re.compile(r"[ \t]+")
_NL_RE = re.compile(r"\n{3,}")


def _clean_text(text: str) -> str:
    text = (text or "").replace("\r\n", "\n").replace("\r", "\n")
    text = _WS_RE.sub(" ", text)
    text = _NL_RE.sub("\n\n", text)
    return text.strip()


_TAG_RE = re.compile(r"<[^>]+>")


def _fallback_strip_tags(raw: str) -> str:
    # Extremely forgiving fallback for malformed XML/HTML-like input.
    raw = (raw or "").replace("\x00", "")
    raw = _TAG_RE.sub(" ", raw)
    return _clean_text(raw)


def xml2txt(xml_path: str | Path) -> str:
    """Extract plain text from an XML document.

    - Joins all element text nodes in document order.
    - Useful for policy XMLs (e.g., your GoPPC-style structured policies).
    """
    xml_path = Path(xml_path)
    try:
        root = ET.parse(str(xml_path)).getroot()
        parts: list[str] = []
        for t in root.itertext():
            t = (t or "").strip()
            if t:
                parts.append(t)
        return _clean_text("\n".join(parts))
    except ParseError:
        # Many real-world "XML" exports are malformed. Fall back to a forgiving parser.
        raw = xml_path.read_text(encoding="utf-8", errors="replace")
        try:
            from bs4 import BeautifulSoup  # type: ignore

            soup = BeautifulSoup(raw, "xml")
            txt = soup.get_text("\n")
            if txt and txt.strip():
                return _clean_text(txt)
        except Exception:
            pass
        return _fallback_strip_tags(raw)


def md2txt(md_path: str | Path) -> str:
    """Best-effort Markdown to plain text conversion (no heavy deps).

    Keeps headings and paragraph text, strips most formatting.
    """
    md_path = Path(md_path)
    s = md_path.read_text(encoding="utf-8", errors="replace")

    # Remove fenced code blocks entirely
    s = re.sub(r"```[\s\S]*?```", "", s)

    # Inline code
    s = re.sub(r"`([^`]+)`", r"\1", s)

    # Images: ![alt](url) -> alt
    s = re.sub(r"!\[([^\]]*)\]\([^)]+\)", r"\1", s)

    # Links: [text](url) -> text
    s = re.sub(r"\[([^\]]+)\]\([^)]+\)", r"\1", s)

    # Strip common emphasis markers
    s = s.replace("**", "").replace("__", "").replace("*", "").replace("_", "")

    # Blockquote/list markers
    s = re.sub(r"(?m)^\s{0,3}>\s?", "", s)
    s = re.sub(r"(?m)^\s*[-*+]\s+", "", s)
    s = re.sub(r"(?m)^\s*\d+\.\s+", "", s)

    # Headings: remove leading #'s
    s = re.sub(r"(?m)^\s{0,3}#{1,6}\s*", "", s)

    return _clean_text(s)


def pdf2txt(pdf_path: str | Path) -> str:
    """Extract text from a PDF using `pypdf`."""
    pdf_path = Path(pdf_path)
    try:
        from pypdf import PdfReader
    except Exception as e:  # pragma: no cover
        raise ImportError("pdf2txt requires `pypdf`. Install with `pip install pypdf`.") from e

    reader = PdfReader(str(pdf_path))
    parts: list[str] = []
    for page in reader.pages:
        parts.append(page.extract_text() or "")
    return _clean_text("\n".join(parts))

def _normalize_text(t: str) -> str:
    t = t or ""
    t = t.replace("\r\n", "\n").replace("\r", "\n")
    t = re.sub(r"[ \t]+", " ", t)
    t = re.sub(r"\n{3,}", "\n\n", t)
    return t.strip()


def _load_scope_articles_text(scope_json_path: str | Path) -> str:
    p = Path(scope_json_path)
    try :
        data = json.loads(p.read_text(encoding="utf-8", errors="replace"))
    except Exception as e:
        print(f"Error loading scope articles text: {e}")
        return """
        "articles": [
    {
      "number": 2,
      "title": "Material scope",
      "text": "This Regulation applies to the processing of personal data wholly or partly by automated means and to the processing other than by automated means of personal data which form part of a filing system or are intended to form part of a filing system.\n\nThis Regulation does not apply to the processing of personal data:\n- in the course of an activity which falls outside the scope of Union law;\n- by the Member States when carrying out activities which fall within the scope of Chapter 2 of Title V of the TEU;\n- by a natural person in the course of a purely personal or household activity;\n- by competent authorities for the purposes of the prevention, investigation, detection or prosecution of criminal offences or the execution of criminal penalties, including the safeguarding against and the prevention of threats to public security.\n\nFor the processing of personal data by the Union institutions, bodies, offices and agencies, Regulation (EC) No 45/2001 applies. Regulation (EC) No 45/2001 and other Union legal acts applicable to such processing of personal data shall be adapted to the principles and rules of this Regulation in accordance with Article 98.\n\nThis Regulation shall be without prejudice to the application of Directive 2000/31/EC, in particular of the liability rules of intermediary service providers in Articles 12 to 15 of that Directive."
    },
    {
      "number": 3,
      "title": "Territorial scope",
      "text": "This Regulation applies to the processing of personal data in the context of the activities of an establishment of a controller or a processor in the Union, regardless of whether the processing takes place in the Union or not.\n\nThis Regulation applies to the processing of personal data of data subjects who are in the Union by a controller or processor not established in the Union, where the processing activities are related to:\n- the offering of goods or services, irrespective of whether a payment of the data subject is required, to such data subjects in the Union; or\n- the monitoring of their behaviour as far as their behaviour takes place within the Union.\n\nThis Regulation applies to the processing of personal data by a controller not established in the Union, but in a place where Member State law applies by virtue of public international law."
    }
        
        """
    arts = data.get("articles") or []
    parts: list[str] = []
    for a in arts:
        try:
            n = int(a.get("number"))
        except Exception:
            continue
        title = (a.get("title") or "").strip()
        body = (a.get("text") or "").strip()
        if not body:
            continue
        header = f"Art. {n} — {title}".strip(" —")
        parts.append(f"{header}\n{body}")
    return "\n\n".join(parts).strip()

def preprocess_file(file_path: str):
    if file_path.endswith('.xml'):

        text = xml2txt(file_path)
    elif file_path.endswith('.md'):
        text = md2txt(file_path)
    elif file_path.endswith('.pdf'):
        text = pdf2txt(file_path)
    else:
        raise ValueError(f"Unsupported file type: {file_path}")
    return text


def _chunk_text(text: str, *, chunk_chars: int = 1200, overlap_chars: int = 200) -> list[dict[str, Any]]:
    """Structure-aware chunking (headings + paragraphs) with fallback overlap splitting.

    Strategy:
    - First, try to split into coherent *sections* using heading detection and blank-line paragraphing.
    - If a section is larger than `chunk_chars`, sub-split that section into overlapping windows.

    Returned offsets are in the normalized text coordinate space (post `_normalize_text`).
    """
    text = _normalize_text(text)
    if not text:
        return []
    if chunk_chars <= 0:
        chunk_chars = 1200
    if overlap_chars < 0:
        overlap_chars = 0

    def is_heading(line: str) -> bool:
        s = (line or "").strip()
        if not s:
            return False
        if s.startswith("#"):  # markdown heading
            return True
        if len(s) > 120:
            return False
        if re.match(r"^\d+(\.\d+){0,4}\s+.+$", s):  # 1. / 1.2.3 Title
            return True
        if re.match(r"^[A-Z0-9][A-Z0-9 \-]{6,}$", s) and any(c.isalpha() for c in s):
            # ALL CAPS heading-ish
            return True
        if re.match(r"^([A-Z][a-z]+)(\s+[A-Z][a-z]+){0,8}$", s) and len(s.split()) <= 8:
            # Title Case short line
            return True
        if s.endswith(":") and len(s.split()) <= 8:  # "Retention:" style
            return True
        return False

    # Build section spans with offsets in normalized text
    line_iter = list(re.finditer(r".*(?:\n|$)", text))
    sections: list[dict[str, Any]] = []

    cur_start = 0
    cur_heading: str | None = None

    def flush(end_pos: int) -> None:
        nonlocal cur_start, cur_heading
        if end_pos <= cur_start:
            return
        sec_text = text[cur_start:end_pos].strip()
        if not sec_text:
            cur_start = end_pos
            cur_heading = None
            return
        sections.append(
            {
                "start": cur_start,
                "end": end_pos,
                "heading": cur_heading,
                "text": sec_text,
            }
        )
        cur_start = end_pos
        cur_heading = None

    for m in line_iter:
        line = m.group(0)
        line_start = m.start()

        if is_heading(line):
            # Start a new section at this heading; flush what came before.
            flush(line_start)
            cur_start = line_start
            cur_heading = line.strip()
            continue

        # Paragraph boundary: two newlines already normalized; if this is a blank line
        if line.strip() == "":
            # keep accumulating; blank line ends paragraph but not necessarily section
            continue

    flush(len(text))

    # If heading detection failed (single section), still do paragraph-aware split
    if len(sections) <= 1:
        paras: list[dict[str, Any]] = []
        for pm in re.finditer(r"(?:[^\n].*?)(?:\n\n|$)", text, flags=re.S):
            p_text = pm.group(0).strip()
            if not p_text:
                continue
            paras.append({"start": pm.start(), "end": pm.end(), "heading": None, "text": p_text})
        if paras:
            sections = paras

    def split_with_overlap(sec: dict[str, Any], *, idx0: int) -> list[dict[str, Any]]:
        start = int(sec["start"])
        end = int(sec["end"])
        sec_text = text[start:end]
        sec_len = len(sec_text)
        if sec_len <= chunk_chars:
            return [
                {
                    "chunk_id": f"c{idx0}",
                    "start": start,
                    "end": end,
                    "text": sec_text.strip(),
                    "heading": sec.get("heading"),
                }
            ]
        step = max(1, chunk_chars - overlap_chars)
        out: list[dict[str, Any]] = []
        j = 0
        while j < sec_len:
            sub_start = start + j
            sub_end = min(end, start + j + chunk_chars)
            sub_text = text[sub_start:sub_end].strip()
            out.append(
                {
                    "chunk_id": f"c{idx0 + len(out)}",
                    "start": sub_start,
                    "end": sub_end,
                    "text": sub_text,
                    "heading": sec.get("heading"),
                }
            )
            if sub_end >= end:
                break
            j += step
        return out

    chunks: list[dict[str, Any]] = []
    idx = 0
    for sec in sections:
        new_chunks = split_with_overlap(sec, idx0=idx)
        chunks.extend(new_chunks)
        idx += len(new_chunks)

    # Final fallback: ensure we always return at least one chunk
    if not chunks:
        chunks = [{"chunk_id": "c0", "start": 0, "end": len(text), "text": text, "heading": None}]
    return chunks
