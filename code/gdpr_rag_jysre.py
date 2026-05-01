"""
GDPR RAG System for Agentic AI Compliance Checking
===================================================
Hierarchical retrieval: Chapter > Section > Article
Compressed summaries via Gemini 2.5 Flash Lite (through local_model.py)
Storage: SQLite | Retrieval: Direct function calls

Usage:
    1. First run: python gdpr_rag.py --setup          (parse JSON → SQLite)
    2. Compress:  python gdpr_rag.py --compress        (LLM compress articles)
    3. Query:     Import and call GDPRRetriever.query()

Dependencies:
    - local_model.py (must be in same directory)
    - pip install google-genai pydantic python-dotenv
    - GEMINI_API_KEY set in env or .api file
"""

import json
import sqlite3
import os
import time
import argparse
from typing import Optional


# ============================================================================
# CONFIGURATION
# ============================================================================

DB_PATH = "gdpr_rag.db"
GDPR_JSON_PATH = "gdpr.json"

# Roman numeral mapping for chapter numbers in the GDPR JSON
ROMAN_TO_INT = {
    "I": 1, "II": 2, "III": 3, "IV": 4, "V": 5,
    "VI": 6, "VII": 7, "VIII": 8, "IX": 9, "X": 10, "XI": 11
}
INT_TO_ROMAN = {v: k for k, v in ROMAN_TO_INT.items()}

# Keys that the LLM must return in its compressed summary
COMPRESSION_KEYS = [
    "core_obligation",
    "applies_to",
    "key_conditions",
    "exceptions",
    "cross_references",
]


# ============================================================================
# STEP 1: DATABASE SCHEMA
# ============================================================================

SCHEMA_SQL = """
CREATE TABLE IF NOT EXISTS gdpr_articles (
    id                 INTEGER PRIMARY KEY AUTOINCREMENT,
    chapter_num        INTEGER NOT NULL,
    chapter_roman      TEXT NOT NULL,
    chapter_title      TEXT NOT NULL,
    section_num        INTEGER,
    section_title      TEXT,
    article_num        INTEGER NOT NULL,
    article_title      TEXT NOT NULL,
    full_text          TEXT NOT NULL,
    compressed_summary TEXT,
    is_compressed      INTEGER DEFAULT 0
);

CREATE INDEX IF NOT EXISTS idx_chapter
    ON gdpr_articles(chapter_num);
CREATE INDEX IF NOT EXISTS idx_chapter_section
    ON gdpr_articles(chapter_num, section_num);
CREATE INDEX IF NOT EXISTS idx_article
    ON gdpr_articles(article_num);
CREATE UNIQUE INDEX IF NOT EXISTS idx_unique_article
    ON gdpr_articles(article_num);
"""


def init_db(db_path: str = DB_PATH) -> sqlite3.Connection:
    """Create database and tables if they don't exist."""
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    conn.executescript(SCHEMA_SQL)
    conn.commit()
    return conn


# ============================================================================
# STEP 2: PARSER — Extract hierarchy from GDPR JSON
# ============================================================================

def _extract_article_text(article: dict) -> str:
    """
    Recursively extract full text from an article's contents.
    Handles the nested points → subpoints structure in gdpr.json.

    Input structure per point:
        {"number": "1", "text": "...", "subpoints": [{"number": "a", "text": "..."}, ...]}

    Output:
        (1) The data subject shall have the right...
          (a) the personal data are no longer necessary...
          (b) the data subject withdraws consent...
    """
    lines = []
    for point in article.get("contents", []):
        point_num = point.get("number", "")
        point_text = point.get("text", "")

        if point_text:
            lines.append(f"({point_num}) {point_text}")

        for subpoint in point.get("subpoints", []):
            sp_num = subpoint.get("number", "")
            sp_text = subpoint.get("text", "")
            if sp_text:
                lines.append(f"  ({sp_num}) {sp_text}")

    return "\n".join(lines)


def parse_gdpr_json(json_path: str = GDPR_JSON_PATH) -> list[dict]:
    """
    Parse the GDPR JSON file into a flat list of article records.

    The JSON hierarchy is:
        chapters[] → contents[] → either:
            type "article"  (direct, no section wrapper)
            type "section"  → contents[] → type "article"

    Returns list of dicts ready for SQLite insertion.
    """
    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    records = []

    for chapter in data["chapters"]:
        ch_roman = chapter["number"]
        ch_num = ROMAN_TO_INT.get(ch_roman, 0)
        ch_title = chapter["title"]

        for item in chapter["contents"]:
            if item["type"] == "article":
                # Article sits directly under chapter (no section)
                records.append({
                    "chapter_num": ch_num,
                    "chapter_roman": ch_roman,
                    "chapter_title": ch_title,
                    "section_num": None,
                    "section_title": None,
                    "article_num": int(item["number"]),
                    "article_title": item.get("title", f"Article {item['number']}"),
                    "full_text": _extract_article_text(item),
                })

            elif item["type"] == "section":
                # Section wrapping one or more articles
                sec_num = int(item["number"])
                sec_title = item["title"]

                for sub_item in item["contents"]:
                    if sub_item["type"] == "article":
                        records.append({
                            "chapter_num": ch_num,
                            "chapter_roman": ch_roman,
                            "chapter_title": ch_title,
                            "section_num": sec_num,
                            "section_title": sec_title,
                            "article_num": int(sub_item["number"]),
                            "article_title": sub_item.get("title", f"Article {sub_item['number']}"),
                            "full_text": _extract_article_text(sub_item),
                        })

    records.sort(key=lambda r: r["article_num"])
    return records


def load_into_db(records: list[dict], db_path: str = DB_PATH):
    """Insert parsed article records into SQLite."""
    conn = init_db(db_path)
    cursor = conn.cursor()

    cursor.execute("DELETE FROM gdpr_articles")

    for rec in records:
        cursor.execute("""
            INSERT INTO gdpr_articles
                (chapter_num, chapter_roman, chapter_title,
                 section_num, section_title,
                 article_num, article_title, full_text)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            rec["chapter_num"], rec["chapter_roman"], rec["chapter_title"],
            rec["section_num"], rec["section_title"],
            rec["article_num"], rec["article_title"], rec["full_text"],
        ))

    conn.commit()
    print(f"[Parser] Loaded {len(records)} articles into {db_path}")
    conn.close()


# ============================================================================
# STEP 3: LLM COMPRESSOR — via local_model._build_json_llm_agent
# ============================================================================

SYSTEM_PROMPT = """You are a GDPR legal analyst. You will receive a GDPR article's full text.
Compress it into a structured summary with these fields:

- core_obligation: What this article requires in plain language (2-3 sentences max)
- applies_to: A list of roles this article applies to. Pick ONLY from: controller, processor, data_subject, supervisory_authority, member_state
- key_conditions: When and how it applies (1-2 sentences)
- exceptions: Any carve-outs or exemptions (1 sentence, or "None")
- cross_references: A list of other GDPR articles explicitly mentioned (e.g. ["Art. 6", "Art. 9"])

Be concise but accurate. This will be used for automated compliance checking."""


def _build_user_prompt(article_row: dict) -> str:
    """Build the user prompt from an article's DB row."""
    section_info = ""
    if article_row["section_num"]:
        section_info = f"\nSection {article_row['section_num']}: {article_row['section_title']}"

    return f"""Compress this GDPR article:

Article {article_row['article_num']}: {article_row['article_title']}
Chapter {article_row['chapter_roman']}: {article_row['chapter_title']}{section_info}

=== FULL TEXT ===
{article_row['full_text']}
=== END TEXT ==="""


def compress_all_articles(db_path: str = DB_PATH, force: bool = False):
    """
    Compress all uncompressed articles using _build_json_llm_agent from local_model.py.

    Flow:
        1. _build_json_llm_agent(local=False) loads Gemini ONCE → returns invoke()
        2. For each article: invoke(system_prompt, user_prompt) → dict
        3. json.dumps(dict) → store in compressed_summary column
    """
    # ------------------------------------------------------------------
    # Import and initialize the LLM agent (model loads ONCE here)
    # ------------------------------------------------------------------
    try:
        from local_model import _build_json_llm_agent
    except ImportError as e:
        print(f"[Compressor] ERROR: Cannot import local_model.py: {e}")
        print("  Make sure local_model.py is in the same directory as gdpr_rag.py")
        return

    print("[Compressor] Initializing Gemini via _build_json_llm_agent(local=False)...")

    try:
        invoke = _build_json_llm_agent(
            required_keys=COMPRESSION_KEYS,
            local=False,
        )
    except Exception as e:
        print(f"[Compressor] ERROR: Failed to initialize LLM agent: {e}")
        return

    # ------------------------------------------------------------------
    # Fetch articles that need compression
    # ------------------------------------------------------------------
    conn = init_db(db_path)
    cursor = conn.cursor()

    if force:
        rows = cursor.execute(
            "SELECT * FROM gdpr_articles ORDER BY article_num"
        ).fetchall()
    else:
        rows = cursor.execute(
            "SELECT * FROM gdpr_articles WHERE is_compressed = 0 ORDER BY article_num"
        ).fetchall()

    if not rows:
        print("[Compressor] All articles already compressed!")
        conn.close()
        return

    print(f"[Compressor] Compressing {len(rows)} articles...")
    print()

    # ------------------------------------------------------------------
    # Loop: call invoke() per article, store result
    # ------------------------------------------------------------------
    success_count = 0

    for i, row in enumerate(rows):
        row_dict = dict(row)
        art_num = row_dict["article_num"]
        art_title = row_dict["article_title"]

        print(f"  [{i+1}/{len(rows)}] Article {art_num}: {art_title}...", end=" ", flush=True)

        try:
            # invoke() returns a dict with COMPRESSION_KEYS
            result_dict = invoke(
                system_prompt=SYSTEM_PROMPT,
                user_prompt=_build_user_prompt(row_dict),
            )

            # invoke() returns dict → serialize to JSON string for SQLite storage
            summary_json = json.dumps(result_dict, ensure_ascii=False)

            cursor.execute(
                "UPDATE gdpr_articles SET compressed_summary = ?, is_compressed = 1 WHERE article_num = ?",
                (summary_json, art_num),
            )
            conn.commit()
            success_count += 1
            print("OK")

        except Exception as e:
            print(f"FAILED ({e})")

        # Rate limiting: stay under free-tier RPM limits
        if i < len(rows) - 1:
            time.sleep(2)

    print(f"\n[Compressor] Done: {success_count}/{len(rows)} articles compressed")
    conn.close()


# ============================================================================
# STEP 4: RETRIEVER — Direct function call (no HTTP server)
# ============================================================================

class GDPRRetriever:
    """
    Hierarchical GDPR retriever.

    Query modes:
        - chapter only       → overview of all articles in chapter
        - chapter + section   → articles in that section
        - article only        → direct article lookup

    Detail levels:
        - "obligations_only" → just ref + core obligation
        - "summary"          → ref + title + full compressed summary fields
        - "full"             → everything including raw article text
    """

    def __init__(self, db_path: str = DB_PATH):
        self.db_path = db_path
        self._conn = None

    @property
    def conn(self) -> sqlite3.Connection:
        """Lazy connection — opens only on first query."""
        if self._conn is None:
            self._conn = sqlite3.connect(self.db_path)
            self._conn.row_factory = sqlite3.Row
        return self._conn

    def close(self):
        if self._conn:
            self._conn.close()
            self._conn = None

    # ------------------------------------------------------------------
    # CORE QUERY METHOD
    # ------------------------------------------------------------------

    def query(
        self,
        chapter: Optional[int] = None,
        section: Optional[int] = None,
        article: Optional[int] = None,
        detail_level: str = "summary",
    ) -> dict:
        """
        Main retrieval method. Called by compliance agents.

        Args:
            chapter:      Chapter number (1-11)
            section:      Section number within chapter (optional)
            article:      Article number (1-99) (optional)
            detail_level: "obligations_only" | "summary" | "full"

        Returns:
            {
                "scope": "chapter" | "section" | "article",
                "count": int,
                "query": {"chapter": ..., "section": ..., "article": ...},
                "results": [...]
            }
        """
        if detail_level not in ("obligations_only", "summary", "full"):
            detail_level = "summary"

        # Build SQL WHERE clause based on specificity
        conditions = []
        params = []

        if article is not None:
            conditions.append("article_num = ?")
            params.append(article)
            scope = "article"

        elif chapter is not None and section is not None:
            conditions.append("chapter_num = ?")
            conditions.append("section_num = ?")
            params.extend([chapter, section])
            scope = "section"

        elif chapter is not None:
            conditions.append("chapter_num = ?")
            params.append(chapter)
            scope = "chapter"

        else:
            return {
                "scope": "error",
                "count": 0,
                "query": {"chapter": chapter, "section": section, "article": article},
                "results": [],
                "error": "At least 'chapter' or 'article' must be provided.",
            }

        where_clause = " AND ".join(conditions)
        sql = f"SELECT * FROM gdpr_articles WHERE {where_clause} ORDER BY article_num"
        rows = self.conn.execute(sql, params).fetchall()

        results = [self._format_row(dict(row), detail_level) for row in rows]

        return {
            "scope": scope,
            "count": len(results),
            "query": {"chapter": chapter, "section": section, "article": article},
            "results": results,
        }

    # ------------------------------------------------------------------
    # CONVENIENCE METHODS
    # ------------------------------------------------------------------

    def get_article(self, article_num: int) -> dict:
        """Get full details for a single article."""
        return self.query(article=article_num, detail_level="full")

    def get_chapter_overview(self, chapter_num: int) -> dict:
        """Get obligation summaries for all articles in a chapter."""
        return self.query(chapter=chapter_num, detail_level="obligations_only")

    def get_section_articles(self, chapter_num: int, section_num: int) -> dict:
        """Get summaries for all articles in a section."""
        return self.query(chapter=chapter_num, section=section_num, detail_level="summary")

    def search_by_keyword(self, keyword: str, detail_level: str = "summary") -> dict:
        """Search articles by keyword in title or full text."""
        sql = """
            SELECT * FROM gdpr_articles
            WHERE article_title LIKE ? OR full_text LIKE ?
            ORDER BY article_num
        """
        pattern = f"%{keyword}%"
        rows = self.conn.execute(sql, (pattern, pattern)).fetchall()
        results = [self._format_row(dict(row), detail_level) for row in rows]

        return {
            "scope": "search",
            "count": len(results),
            "query": {"keyword": keyword},
            "results": results,
        }

    def list_structure(self) -> dict:
        """Return the full GDPR structure: chapters, sections, article counts."""
        sql = """
            SELECT chapter_num, chapter_roman, chapter_title,
                   section_num, section_title,
                   COUNT(*) as article_count,
                   MIN(article_num) as first_article,
                   MAX(article_num) as last_article
            FROM gdpr_articles
            GROUP BY chapter_num, section_num
            ORDER BY chapter_num, section_num
        """
        rows = self.conn.execute(sql).fetchall()

        structure = {}
        for row in rows:
            row = dict(row)
            ch_key = row["chapter_num"]
            if ch_key not in structure:
                structure[ch_key] = {
                    "chapter_roman": row["chapter_roman"],
                    "chapter_title": row["chapter_title"],
                    "sections": {},
                    "total_articles": 0,
                }

            sec_key = row["section_num"] or "direct"
            structure[ch_key]["sections"][sec_key] = {
                "section_title": row["section_title"] or "(No section)",
                "article_count": row["article_count"],
                "articles": f"{row['first_article']}-{row['last_article']}",
            }
            structure[ch_key]["total_articles"] += row["article_count"]

        return structure

    # ------------------------------------------------------------------
    # RESPONSE FORMATTER
    # ------------------------------------------------------------------

    def _format_row(self, row: dict, detail_level: str) -> dict:
        """
        Format a single DB row based on detail level.

        The compressed_summary column stores a JSON string produced by
        _build_json_llm_agent's invoke(). We parse it back into a dict
        and extract fields based on the requested detail level.
        """
        ref = f"Art. {row['article_num']} GDPR"

        # Parse the stored JSON summary
        summary_data = {}
        if row.get("compressed_summary"):
            try:
                summary_data = json.loads(row["compressed_summary"])
            except json.JSONDecodeError:
                summary_data = {"core_obligation": row["compressed_summary"]}

        if detail_level == "obligations_only":
            return {
                "reference": ref,
                "article_num": row["article_num"],
                "title": row["article_title"],
                "core_obligation": summary_data.get("core_obligation", "Not yet compressed"),
            }

        elif detail_level == "summary":
            return {
                "reference": ref,
                "article_num": row["article_num"],
                "title": row["article_title"],
                "chapter": f"Chapter {row['chapter_roman']}: {row['chapter_title']}",
                "section": f"Section {row['section_num']}: {row['section_title']}"
                           if row["section_num"] else None,
                "core_obligation": summary_data.get("core_obligation", "Not yet compressed"),
                "applies_to": summary_data.get("applies_to", []),
                "key_conditions": summary_data.get("key_conditions", ""),
                "exceptions": summary_data.get("exceptions", ""),
                "cross_references": summary_data.get("cross_references", []),
            }

        else:  # "full"
            return {
                "reference": ref,
                "article_num": row["article_num"],
                "title": row["article_title"],
                "chapter": f"Chapter {row['chapter_roman']}: {row['chapter_title']}",
                "section": f"Section {row['section_num']}: {row['section_title']}"
                           if row["section_num"] else None,
                "core_obligation": summary_data.get("core_obligation", "Not yet compressed"),
                "applies_to": summary_data.get("applies_to", []),
                "key_conditions": summary_data.get("key_conditions", ""),
                "exceptions": summary_data.get("exceptions", ""),
                "cross_references": summary_data.get("cross_references", []),
                "full_text": row["full_text"],
            }


# ============================================================================
# STEP 5: CLI ENTRY POINT
# ============================================================================

def print_query_result(result: dict):
    """Pretty-print a query result to console."""
    print(f"\n{'='*70}")
    print(f"Scope: {result['scope']}  |  Results: {result['count']}")
    print(f"Query: {result['query']}")
    print(f"{'='*70}")

    for item in result["results"]:
        print(f"\n  [{item['reference']}] {item.get('title', '')}")

        if "core_obligation" in item:
            print(f"  Obligation: {item['core_obligation']}")

        if item.get("applies_to"):
            roles = item["applies_to"]
            if isinstance(roles, list):
                print(f"  Applies to: {', '.join(str(r) for r in roles)}")
            else:
                print(f"  Applies to: {roles}")

        if item.get("key_conditions"):
            print(f"  Conditions: {item['key_conditions']}")

        if item.get("exceptions"):
            print(f"  Exceptions: {item['exceptions']}")

        if item.get("cross_references"):
            refs = item["cross_references"]
            if isinstance(refs, list):
                print(f"  Cross-refs: {', '.join(str(r) for r in refs)}")
            else:
                print(f"  Cross-refs: {refs}")

        if item.get("full_text"):
            preview = item["full_text"][:200] + "..." if len(item["full_text"]) > 200 else item["full_text"]
            print(f"  Text: {preview}")

    print()


def main():
    parser = argparse.ArgumentParser(description="GDPR RAG System")
    parser.add_argument("--setup", action="store_true",
                        help="Parse gdpr.json and load into SQLite")
    parser.add_argument("--compress", action="store_true",
                        help="Compress articles using Gemini via local_model.py")
    parser.add_argument("--force-compress", action="store_true",
                        help="Re-compress all articles (even already done)")
    parser.add_argument("--json-path", default=GDPR_JSON_PATH,
                        help="Path to gdpr.json file")
    parser.add_argument("--db-path", default=DB_PATH,
                        help="Path to SQLite database")

    # Query arguments
    parser.add_argument("--chapter", type=int, help="Query by chapter (1-11)")
    parser.add_argument("--section", type=int, help="Query by section")
    parser.add_argument("--article", type=int, help="Query by article (1-99)")
    parser.add_argument("--search", type=str, help="Keyword search")
    parser.add_argument("--detail", default="summary",
                        choices=["obligations_only", "summary", "full"],
                        help="Detail level for query results")
    parser.add_argument("--structure", action="store_true",
                        help="Show GDPR structure overview")
    parser.add_argument("--demo", action="store_true",
                        help="Run demo queries")

    args = parser.parse_args()

    # --- Setup ---
    if args.setup:
        print("[Setup] Parsing GDPR JSON...")
        records = parse_gdpr_json(args.json_path)
        print(f"[Setup] Found {len(records)} articles")
        load_into_db(records, args.db_path)
        print("[Setup] Done! Database ready at:", args.db_path)
        return

    # --- Compress ---
    if args.compress or args.force_compress:
        compress_all_articles(args.db_path, force=args.force_compress)
        return

    # --- Queries ---
    retriever = GDPRRetriever(args.db_path)

    if args.structure:
        structure = retriever.list_structure()
        print("\n=== GDPR STRUCTURE ===\n")
        for ch_num, ch_data in structure.items():
            print(f"Chapter {ch_data['chapter_roman']}: {ch_data['chapter_title']} "
                  f"({ch_data['total_articles']} articles)")
            for sec_key, sec_data in ch_data["sections"].items():
                print(f"  Section {sec_key}: {sec_data['section_title']} "
                      f"→ Articles {sec_data['articles']}")
        print()
        return

    if args.search:
        result = retriever.search_by_keyword(args.search, args.detail)
        print_query_result(result)
        return

    if args.chapter or args.article:
        result = retriever.query(
            chapter=args.chapter,
            section=args.section,
            article=args.article,
            detail_level=args.detail,
        )
        print_query_result(result)
        return

    if args.demo:
        print("\n" + "="*70)
        print(" GDPR RAG SYSTEM — DEMO QUERIES")
        print("="*70)

        print("\n--- Demo 1: Get Article 17 (Right to Erasure) ---")
        result = retriever.get_article(17)
        print_query_result(result)

        print("\n--- Demo 2: Chapter 3 Overview (Rights of Data Subject) ---")
        result = retriever.get_chapter_overview(3)
        print_query_result(result)

        print("\n--- Demo 3: Chapter 4, Section 2 (Security of Personal Data) ---")
        result = retriever.get_section_articles(4, 2)
        print_query_result(result)

        print("\n--- Demo 4: Search for 'consent' ---")
        result = retriever.search_by_keyword("consent", "obligations_only")
        print_query_result(result)

        return

    parser.print_help()


if __name__ == "__main__":
    main()
