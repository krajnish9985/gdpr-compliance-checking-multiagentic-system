"""
GDPR RAG System for Agentic AI Compliance Checking
===================================================
Hierarchical retrieval: Chapter > Section > Article
Compressed summaries via Gemini 2.5 Flash Lite (through local_model.py)
Storage: SQLite | Retrieval: Direct function calls

Compression modes:
    1. ALL articles at once:   python gdpr_rag.py --compress
    2. SPECIFIC article(s):    python gdpr_rag.py --compress-article 17
                                python gdpr_rag.py --compress-article 17,33,34
    3. ON-DEMAND from agent:   rag.query(article=17, ..., auto_compress=True)
       (default: True — uncompressed articles are compressed when queried)

Once compressed, the summary is stored permanently in SQLite.
Future queries for the same article skip the LLM call and read from DB.

Usage:
    1. First run: python gdpr_rag.py --setup          (parse JSON → SQLite)
    2. Compress:  python gdpr_rag.py --compress-article 17,33  (cheap)
                  OR python gdpr_rag.py --compress (all 99, more expensive)
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


# ----------------------------------------------------------------------------
# Lazy-loaded singleton for the LLM agent
# ----------------------------------------------------------------------------
# We keep this at module level so multiple compression calls reuse the same
# Gemini client. _get_invoke() is called only when actually needed — i.e. only
# when an article requires compression. If everything is already cached in DB,
# the LLM is never imported or initialized.
# ----------------------------------------------------------------------------

_INVOKE_FN = None


def _get_invoke():
    """
    Lazy-init the LLM agent. Loads ONCE on first call, reused after.
    Returns the invoke(system_prompt, user_prompt) → dict callable.
    """
    global _INVOKE_FN
    if _INVOKE_FN is not None:
        return _INVOKE_FN

    try:
        from local_model import _build_json_llm_agent
    except ImportError as e:
        raise ImportError(
            "Cannot import local_model.py — make sure it's in the same directory. "
            f"Original error: {e}"
        )

    print("[Compressor] Initializing Gemini (one-time)...")
    _INVOKE_FN = _build_json_llm_agent(
        required_keys=COMPRESSION_KEYS,
        local=False,
    )
    return _INVOKE_FN


def compress_single_article(article_num: int, db_path: str = DB_PATH,
                            force: bool = False, conn: Optional[sqlite3.Connection] = None) -> bool:
    """
    Compress ONE specific article and store it in SQLite.

    Args:
        article_num: GDPR article number (1-99)
        db_path:     Path to SQLite database
        force:       If True, re-compresses even if already compressed
        conn:        Optional existing connection (avoids reopening)

    Returns:
        True if compression succeeded (or article was already compressed),
        False on failure.
    """
    # Use provided connection or open a new one
    own_conn = conn is None
    if own_conn:
        conn = init_db(db_path)

    try:
        cursor = conn.cursor()

        # Fetch the article
        row = cursor.execute(
            "SELECT * FROM gdpr_articles WHERE article_num = ?",
            (article_num,)
        ).fetchone()

        if row is None:
            print(f"[Compressor] ERROR: Article {article_num} not found in DB")
            return False

        row_dict = dict(row)

        # Skip if already compressed (unless --force)
        if row_dict["is_compressed"] and not force:
            return True  # already done, nothing to do

        # Lazy-init Gemini and call it
        try:
            invoke = _get_invoke()
        except Exception as e:
            print(f"[Compressor] ERROR: Failed to initialize LLM: {e}")
            return False

        print(f"[Compressor] Compressing Article {article_num}: {row_dict['article_title']}...",
              end=" ", flush=True)

        try:
            result_dict = invoke(
                system_prompt=SYSTEM_PROMPT,
                user_prompt=_build_user_prompt(row_dict),
            )
            summary_json = json.dumps(result_dict, ensure_ascii=False)

            cursor.execute(
                "UPDATE gdpr_articles SET compressed_summary = ?, is_compressed = 1 WHERE article_num = ?",
                (summary_json, article_num),
            )
            conn.commit()
            print("OK")
            return True

        except Exception as e:
            print(f"FAILED ({e})")
            return False

    finally:
        if own_conn:
            conn.close()


def compress_articles_batch(article_nums: list[int], db_path: str = DB_PATH, force: bool = False):
    """
    Compress a batch of specific articles (uses same LLM session).

    Useful for: python gdpr_rag.py --compress-article 17,33,34
    """
    if not article_nums:
        print("[Compressor] No article numbers provided")
        return

    conn = init_db(db_path)
    success = 0
    skipped = 0
    failed = 0

    print(f"[Compressor] Processing {len(article_nums)} article(s): {article_nums}")
    print()

    for i, art_num in enumerate(article_nums):
        # Check if already compressed before calling LLM
        cursor = conn.cursor()
        existing = cursor.execute(
            "SELECT is_compressed FROM gdpr_articles WHERE article_num = ?",
            (art_num,)
        ).fetchone()

        if existing is None:
            print(f"[Compressor] Article {art_num} not found — skipping")
            failed += 1
            continue

        if existing["is_compressed"] and not force:
            print(f"[Compressor] Article {art_num} already compressed — skipping (use --force to redo)")
            skipped += 1
            continue

        ok = compress_single_article(art_num, db_path, force=force, conn=conn)
        if ok:
            success += 1
        else:
            failed += 1

        # Rate limit between calls
        if i < len(article_nums) - 1:
            time.sleep(2)

    conn.close()
    print(f"\n[Compressor] Done: {success} compressed, {skipped} already done, {failed} failed")


def compress_all_articles(db_path: str = DB_PATH, force: bool = False):
    """
    Compress ALL uncompressed articles (the original bulk operation).
    """
    conn = init_db(db_path)
    cursor = conn.cursor()

    if force:
        rows = cursor.execute(
            "SELECT article_num FROM gdpr_articles ORDER BY article_num"
        ).fetchall()
    else:
        rows = cursor.execute(
            "SELECT article_num FROM gdpr_articles WHERE is_compressed = 0 ORDER BY article_num"
        ).fetchall()

    if not rows:
        print("[Compressor] All articles already compressed!")
        conn.close()
        return

    article_nums = [row["article_num"] for row in rows]
    conn.close()

    print(f"[Compressor] Compressing {len(article_nums)} article(s)...")
    compress_articles_batch(article_nums, db_path=db_path, force=force)


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
        - "obligations_only" → just ref + core obligation (4 fields)
        - "summary"          → ref + title + full compressed summary fields (9 fields)
        - "full"             → everything including raw article text (10 fields)

    On-demand compression:
        - When auto_compress=True (default), uncompressed articles in the result
          are automatically compressed via LLM and persisted to DB before returning.
        - Already-compressed articles never trigger an LLM call.
        - Set auto_compress=False to skip LLM and return placeholder text.
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
        auto_compress: bool = True,
    ) -> dict:
        """
        Main retrieval method. Called by compliance agents.

        Args:
            chapter:        Chapter number (1-11)
            section:        Section number within chapter (optional)
            article:        Article number (1-99) (optional)
            detail_level:   "obligations_only" | "summary" | "full"
            auto_compress:  If True (default), automatically compress any
                            uncompressed articles in the result via LLM.
                            Set False to skip LLM (returns placeholder).

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

        # ------------------------------------------------------------------
        # ON-DEMAND COMPRESSION
        # ------------------------------------------------------------------
        # If any matched article is uncompressed and auto_compress is True,
        # call the LLM for that specific article and save it to the DB.
        # ------------------------------------------------------------------
        if auto_compress:
            uncompressed_nums = [
                row["article_num"] for row in rows if not row["is_compressed"]
            ]

            if uncompressed_nums:
                print(f"[Retriever] {len(uncompressed_nums)} article(s) need compression: {uncompressed_nums}")
                for art_num in uncompressed_nums:
                    compress_single_article(art_num, self.db_path, conn=self.conn)

                # Re-fetch rows with the now-updated compressed_summary values
                rows = self.conn.execute(sql, params).fetchall()

        # Format results based on detail level
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

    def get_article(self, article_num: int, auto_compress: bool = True) -> dict:
        """Get full details for a single article."""
        return self.query(article=article_num, detail_level="full",
                          auto_compress=auto_compress)

    def get_chapter_overview(self, chapter_num: int, auto_compress: bool = True) -> dict:
        """Get obligation summaries for all articles in a chapter."""
        return self.query(chapter=chapter_num, detail_level="obligations_only",
                          auto_compress=auto_compress)

    def get_section_articles(self, chapter_num: int, section_num: int,
                             auto_compress: bool = True) -> dict:
        """Get summaries for all articles in a section."""
        return self.query(chapter=chapter_num, section=section_num,
                          detail_level="summary", auto_compress=auto_compress)

    def search_by_keyword(self, keyword: str, detail_level: str = "summary",
                          auto_compress: bool = False) -> dict:
        """
        Search articles by keyword in title or full text.
        Default: auto_compress=False (avoid surprise LLM bills on broad searches).
        """
        sql = """
            SELECT * FROM gdpr_articles
            WHERE article_title LIKE ? OR full_text LIKE ?
            ORDER BY article_num
        """
        pattern = f"%{keyword}%"
        rows = self.conn.execute(sql, (pattern, pattern)).fetchall()

        if auto_compress:
            uncompressed_nums = [r["article_num"] for r in rows if not r["is_compressed"]]
            if uncompressed_nums:
                print(f"[Retriever] Compressing {len(uncompressed_nums)} article(s) from search...")
                for art_num in uncompressed_nums:
                    compress_single_article(art_num, self.db_path, conn=self.conn)
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

    def compression_status(self) -> dict:
        """Show how many articles are compressed vs. pending."""
        compressed = self.conn.execute(
            "SELECT COUNT(*) as c FROM gdpr_articles WHERE is_compressed = 1"
        ).fetchone()["c"]

        total = self.conn.execute(
            "SELECT COUNT(*) as c FROM gdpr_articles"
        ).fetchone()["c"]

        compressed_articles = [
            row["article_num"] for row in
            self.conn.execute(
                "SELECT article_num FROM gdpr_articles WHERE is_compressed = 1 ORDER BY article_num"
            ).fetchall()
        ]

        return {
            "total_articles": total,
            "compressed": compressed,
            "pending": total - compressed,
            "compressed_article_nums": compressed_articles,
        }

    # ------------------------------------------------------------------
    # RESPONSE FORMATTER
    # ------------------------------------------------------------------

    def _format_row(self, row: dict, detail_level: str) -> dict:
        """Format a single DB row based on detail level."""
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


def _parse_article_list(value: str) -> list[int]:
    """Parse '17,33,34' or '17' into [17, 33, 34] / [17]."""
    nums = []
    for part in value.split(","):
        part = part.strip()
        if part:
            nums.append(int(part))
    return nums


def main():
    parser = argparse.ArgumentParser(
        description="GDPR RAG System with on-demand compression",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Compression options:
  --compress                       Compress ALL 99 articles (heavy)
  --compress-article 17            Compress just article 17
  --compress-article 17,33,34      Compress specific articles
  --force-compress                 Re-compress even if already done
  --status                         Show compression progress

Query examples:
  --article 17 --detail full
  --chapter 3 --detail obligations_only
  --chapter 4 --section 4 --detail summary
  --search "data breach"
""",
    )
    parser.add_argument("--setup", action="store_true",
                        help="Parse gdpr.json and load into SQLite")

    # Compression flags
    parser.add_argument("--compress", action="store_true",
                        help="Compress ALL uncompressed articles")
    parser.add_argument("--compress-article", type=str, default=None,
                        help="Compress specific article(s), comma-separated (e.g. 17 or 17,33,34)")
    parser.add_argument("--force-compress", action="store_true",
                        help="Re-compress even if already compressed")
    parser.add_argument("--status", action="store_true",
                        help="Show compression status")

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
    parser.add_argument("--no-auto-compress", action="store_true",
                        help="Disable auto-compression on query (return placeholder for uncompressed)")
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

    # --- Status ---
    if args.status:
        retriever = GDPRRetriever(args.db_path)
        status = retriever.compression_status()
        retriever.close()

        print(f"\nCompression Status")
        print(f"  Total articles:    {status['total_articles']}")
        print(f"  Compressed:        {status['compressed']}")
        print(f"  Pending:           {status['pending']}")
        if status["compressed_article_nums"]:
            print(f"  Done so far:       {status['compressed_article_nums']}")
        print()
        return

    # --- Compress specific article(s) ---
    if args.compress_article:
        try:
            article_nums = _parse_article_list(args.compress_article)
        except ValueError:
            print("[Compressor] ERROR: invalid format. Use e.g. --compress-article 17 or --compress-article 17,33,34")
            return
        compress_articles_batch(article_nums, db_path=args.db_path, force=args.force_compress)
        return

    # --- Compress all ---
    if args.compress or args.force_compress:
        compress_all_articles(args.db_path, force=args.force_compress)
        return

    # --- Queries ---
    retriever = GDPRRetriever(args.db_path)
    auto_compress = not args.no_auto_compress

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
        result = retriever.search_by_keyword(
            args.search, args.detail, auto_compress=auto_compress
        )
        print_query_result(result)
        return

    if args.chapter or args.article:
        result = retriever.query(
            chapter=args.chapter,
            section=args.section,
            article=args.article,
            detail_level=args.detail,
            auto_compress=auto_compress,
        )
        print_query_result(result)
        return

    if args.demo:
        print("\n" + "="*70)
        print(" GDPR RAG SYSTEM — DEMO QUERIES")
        print("="*70)

        print("\n--- Demo 1: Get Article 17 (auto-compresses if needed) ---")
        result = retriever.get_article(17)
        print_query_result(result)

        print("\n--- Demo 2: Chapter 3 Overview ---")
        result = retriever.get_chapter_overview(3)
        print_query_result(result)

        print("\n--- Demo 3: Chapter 4, Section 4 (DPO articles) ---")
        result = retriever.get_section_articles(4, 4)
        print_query_result(result)

        print("\n--- Demo 4: Search for 'consent' (no auto-compress) ---")
        result = retriever.search_by_keyword("consent", "obligations_only",
                                             auto_compress=False)
        print_query_result(result)

        return

    parser.print_help()


if __name__ == "__main__":
    main()
