import json
from typing import Any, TypedDict
from langgraph.graph import END, START, StateGraph

from dataclass import Document, ArticlePolicy, KeywordNode, Prompts
from states import WorkflowState
from local_model import _build_json_llm_agent
from utils import _normalize_text, preprocess_file, _load_scope_articles_text, _chunk_text
from prompts import DEFAULT_PROMPTS
from rag import dummy_rag_fetch_article


def load_metadata(load_article_policies: bool = True, load_keyword_nodes: bool = True):
    if load_article_policies:
        with open('../metadata/article_policies.json', 'r') as f:
            article_policies = [ArticlePolicy(**p) for p in json.load(f)]
            keyword_nodes = None
    if load_keyword_nodes:
        with open('../metadata/keyword_nodes.json', 'r') as f:
            keyword_nodes = [KeywordNode(**k) for k in json.load(f)]
            article_policies = None
    return article_policies, keyword_nodes

def filter_nonskip_articles(articles: Iterable[ArticlePolicy]) -> Iterable[ArticlePolicy]:
    for a in articles:
        if a.priority != "skip":
            yield a

def build_graph(local : bool = False):

    scope_agent = _build_json_llm_agent(required_keys=["applies", "reasons", "evidence", "hil_required"], local=local)
    # Mapping agent classifies a small bundle of policy chunks into relevant GDPR articles.
    mapping_agent = _build_json_llm_agent(required_keys=["article_numbers", "notes"], local=local)
    check_agent = _build_json_llm_agent(
        required_keys=["status", "gaps", "evidence", "risk", "needs_human_review", "notes"], local=local
    )

    def ingestion_node(state: WorkflowState) -> WorkflowState:
        paths = state.get("document_paths") or []
        docs = [preprocess_file(p) for p in paths] # todo : here provision for multiple documents with policies for the same company, can add multiple documents with policies for the same company, 
        full_text = _normalize_text("\n\n".join(d for d in docs))
        
        return {"documents": docs, "full_text": full_text}
    
    def load_reference_node(state: WorkflowState) -> WorkflowState:
        article_policies, _ = load_metadata(load_article_policies=True)[0]
        relevant = list(filter_nonskip_articles(article_policies))
        
        return { "relevant_articles": relevant}

    def scope_gate_node(state: WorkflowState) -> WorkflowState:

        scope_ref_text = _load_scope_articles_text("../metadata/scope_gate.json")
        prompts = DEFAULT_PROMPTS
        full_text = state.get("full_text") or ""
        user_prompt = (
            "Use the following GDPR reference text for scope assessment.\n\n"
            f"{scope_ref_text}\n\n"
            "Now analyze this company policy text:\n\n"
            + (full_text[:1200])
        )
        scope = scope_agent(prompts.scope_gate_system, user_prompt) or {
            "applies": "unclear",
            "reasons": [],
            "evidence": [],
            "hil_required": True,
        }
        return {"scope": scope}
    

    def chunking_node(state: WorkflowState) -> WorkflowState:

        full_text = state.get("full_text") or ""
        chunk_chars = int(state.get("chunk_chars") or 1200)
        overlap_chars = int(state.get("overlap_chars") or 200)
        chunks = _chunk_text(full_text, chunk_chars=chunk_chars, overlap_chars=overlap_chars)
        
        return {"chunks": chunks}
    
    def mapping_node(state: WorkflowState) -> WorkflowState:

  # Load pre-filtered keyword nodes (relevant articles only).
        _, kw_nodes_raw = load_metadata(load_keyword_nodes=True)

        chunks = state.get("chunks") or []
        relevant_articles = state.get("relevant_articles") or []
        mapping_bundle_size = max(1, int(state.get("mapping_bundle_size") or 2))
        mapping_max_bundles = int(state.get("mapping_max_bundles") or 0)  # 0 => no cap
        print("Mapping bundle size", mapping_bundle_size)
        print("Mapping max bundles", mapping_max_bundles)
        relevant_set = {int(a.number) for a in relevant_articles}

        # Put ALL article keywords once in the system prompt. Keep it minimal.
        keyword_reference = json.dumps(kw_nodes_raw, ensure_ascii=False)
        # print("Keyword reference", keyword_reference)
        mapping_system_prompt = (
            "You are a GDPR mapping agent.\n"
            "Given a small bundle of policy text, return which GDPR articles it is relevant to.\n"
            "Return JSON with keys:\n"
            "- article_numbers: array of integers (GDPR article numbers). Use [] if none.\n"
            "- notes: short string.\n\n"
            "Keyword reference JSON (each item: {node:<article_number>, keywords:[...]}):\n"
            f"{keyword_reference}"
        )

        article_store: dict[int, list[dict[str, Any]]] = {n: [] for n in relevant_set}
        seen_by_art: dict[int, set[str]] = {n: set() for n in relevant_set}

        # print("bundle size", mapping_bundle_size, "Chunks", chunks)
        res_js = []
        bundle_count = 0
        for i in range(0, len(chunks), mapping_bundle_size):
            print("Bundle count", bundle_count)
            if mapping_max_bundles > 0 and bundle_count >= mapping_max_bundles:
                break
            b_chunks = chunks[i : i + mapping_bundle_size]
            if not b_chunks:
                continue

            bundle_text = "\n\n".join((c.get("text") or "").strip() for c in b_chunks if (c.get("text") or "").strip())
            if not bundle_text:
                continue
            # print("Bundle text", bundle_text)
            js = mapping_agent(mapping_system_prompt, bundle_text) or {}
            if local : 
                res_js.append(js)
                bundle_count += 1
                continue
            
            arts = js.get("article_numbers") or []
            if isinstance(arts, (int, str)):
                arts = [arts]

            cleaned: list[int] = []
            for a in arts:
                try:
                    n = int(str(a).strip())
                except Exception:
                    continue
                if n in relevant_set:
                    cleaned.append(n)

            for art_no in cleaned:
                for c in b_chunks:
                    cid = str(c.get("chunk_id"))
                    if cid in seen_by_art[art_no]:
                        continue
                    article_store[art_no].append(c)
                    seen_by_art[art_no].add(cid)

            bundle_count += 1

        if local:
            return res_js
        return {"article_store": article_store}
    
    def rag_fetch_node(state: WorkflowState) -> WorkflowState:
        relevant_articles = state.get("relevant_articles") or []
        rag: dict[int, dict[str, Any]] = {}
        for art in relevant_articles:
            rag[art.number] = dummy_rag_fetch_article(art.number)
        return {"rag_articles": rag}
    

    





     