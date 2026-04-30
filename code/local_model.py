from __future__ import annotations


from dataclasses import dataclass
from typing import Any, Dict, Optional, TypedDict
from pathlib import Path
from dotenv import load_dotenv

import json
import re
from dataclasses import dataclass
from typing import Any, Dict, Optional, TypedDict
import os
load_dotenv()


class GraphState(TypedDict, total=False):
    """LangGraph state for a single JSON-producing call."""

    # Input
    system_prompt: str
    user_prompt: str

    # Output
    raw_model_text: str
    json_output: Dict[str, Any]

    # Error handling
    error: str


JSON_OBJECT_RE = re.compile(r"\{[\s\S]*\}")


def _extract_first_json_object(text: str) -> str:
    """Best-effort extraction of the first JSON object from a model response."""
    m = JSON_OBJECT_RE.search(text or "")
    if not m:
        raise ValueError("No JSON object found in model output.")
    return m.group(0).strip()


def _parse_json(text: str) -> Dict[str, Any]:
    obj_text = _extract_first_json_object(text)
    return json.loads(obj_text)


def _json_only_prompt(system_prompt: str, user_prompt: str, required_keys: Optional[list[str]] = None) -> str:
    required_keys = required_keys or ["answer"]
    keys_str = ", ".join(required_keys)
    return f"""<|turn>system
You must output ONLY valid JSON (no markdown, no prose, no code fences).
The JSON must be an object with these keys: {keys_str}.
If you are unsure, still output JSON and set missing fields to null.

{system_prompt.strip()}
<turn|>
<|turn>user
{user_prompt.strip()}
<turn|>
<|turn>model
"""


@dataclass
class LocalMLXModel:
    """Local model wrapper using mlx_vlm (as in models.ipynb)."""

    model_id: str = "mlx-community/gemma-4-e4b-it-4bit"
    _model: Any = None
    _tokenizer: Any = None

    def load(self):
        if self._model is not None and self._tokenizer is not None:
            return self._model, self._tokenizer

        from mlx_vlm import load  # local import to keep module importable without mlx_vlm

        self._model, self._tokenizer = load(self.model_id)
        return self._model, self._tokenizer

    def generate(self, prompt: str, max_tokens: Optional[int] = None) -> str:
        from mlx_vlm import generate  # local import to keep module importable without mlx_vlm

        model, tokenizer = self.load()
        kwargs: Dict[str, Any] = {"prompt": prompt}
        if max_tokens is not None:
            kwargs["max_tokens"] = max_tokens

        result = generate(model, tokenizer, **kwargs)
        # mlx_vlm returns GenerationResult(text=..., ...)
        return getattr(result, "text", str(result))


def build_graph(
    *,
    model: Optional[LocalMLXModel] = None,
    required_keys: Optional[list[str]] = None,
):
    """Create a LangGraph with a single node that returns json into state.

    Usage:
        graph = build_graph(required_keys=["answer", "rationale", "citations"])
        out = graph.invoke({"system_prompt": "...", "user_prompt": "..."})
        out["json_output"]  # -> dict
    """

    try:
        from langgraph.graph import START, StateGraph
    except Exception as e:  # pragma: no cover
        raise ImportError(
            "LangGraph is not installed. Install with `pip install langgraph`."
        ) from e

    model = model or LocalMLXModel()
    required_keys = required_keys or ["answer"]

    def first_node(state: GraphState) -> GraphState:
        system_prompt = (state.get("system_prompt") or "").strip()
        user_prompt = (state.get("user_prompt") or "").strip()

        prompt = _json_only_prompt(system_prompt, user_prompt, required_keys=required_keys)
        try:
            raw = model.generate(prompt)
            parsed = _parse_json(raw)

            # Ensure required keys exist (fill nulls when missing)
            for k in required_keys:
                parsed.setdefault(k, None)

            return {"raw_model_text": raw, "json_output": parsed}
        except Exception as ex:
            return {"raw_model_text": locals().get("raw", ""), "error": str(ex)}

    g = StateGraph(GraphState)
    g.add_node("json_node", first_node)
    g.add_edge(START, "json_node")
    compiled = g.compile()
    return compiled
 

def make_prompt_for_gemma(*, system_prompt: str, user_prompt: str, required_keys: list[str]) -> str:
    """Gemma-style prompt wrapper with `<|turn|>` and `<|channel|>` tags.

    We instruct the model to output ONLY JSON in the final channel.
    """
    keys_str = ", ".join(required_keys)
    return f"""<|turn>system
You must output ONLY valid JSON (no markdown, no prose, no code fences).
The JSON must be an object with these keys: {keys_str}.
If you are unsure, still output JSON and set missing fields to null.

<|think|>
{system_prompt.strip()}
<turn|>
<|turn>user
{user_prompt.strip()}
<turn|>
<|turn>model
<|channel>thought
Think step-by-step privately. Do not reveal reasoning. Produce ONLY the required JSON in the final channel.
<channel|>
"""


def _extract_first_json_object(text: str) -> str:
    import re

    m = re.search(r"\{[\s\S]*\}", text or "")
    if not m:
        raise ValueError("No JSON object found in model output.")
    return m.group(0).strip()


def _parse_json(text: str) -> dict[str, Any]:
    return json.loads(_extract_first_json_object(text))

def _maybe_parse_json_string(value: Any) -> Any:
    """If `value` looks like a JSON object/array encoded as a string, parse it."""
    if not isinstance(value, str):
        return value
    s = value.strip()
    if not s:
        return value
    if (s.startswith("{") and s.endswith("}")) or (s.startswith("[") and s.endswith("]")):
        try:
            return json.loads(s)
        except Exception:
            return value
    return value


def _deep_fix_json_strings(obj: Any) -> Any:
    """Recursively convert JSON-in-string fields into real JSON objects."""
    obj = _maybe_parse_json_string(obj)
    if isinstance(obj, dict):
        return {k: _deep_fix_json_strings(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_deep_fix_json_strings(v) for v in obj]
    return obj



def _build_json_llm_agent(*, required_keys: list[str], local: bool):
    """Create a reusable JSON-only agent graph.

    Requirements:
    - LLM-only (no heuristic fallback)
    - If `local=True`, load the local model ONCE up-front, then reuse for all calls.
    """
    # --- Local (MLX) path ---
    if local:
      
        model = LocalMLXModel()
        # Load once, so downstream node calls reuse the same model/tokenizer.
        try:
            model.load()
        except ModuleNotFoundError as e:  # pragma: no cover
            raise ModuleNotFoundError(
                "Local model backend is missing. Install `mlx-vlm` to use local=True.\n"
                "Example: `pip install mlx-vlm`"
            ) from e

        def invoke(system_prompt: str, user_prompt: str) -> dict[str, Any]:
            prompt = make_prompt_for_gemma(
                system_prompt=system_prompt, user_prompt=user_prompt, required_keys=required_keys
            )
            raw = model.generate(prompt)
            return raw

            # parsed = _deep_fix_json_strings(_parse_json(raw))
            # for k in required_keys:
            #     parsed.setdefault(k, None)
            # return parsed
        print("Local model loaded")
        return invoke

    # --- Remote (Gemini API) path with structured output ---
    try:
        from google import genai  # type: ignore
        from google.genai import types  # type: ignore
        from pydantic import BaseModel, Field, create_model
    except Exception as e:  # pragma: no cover
        raise ImportError(
            "Gemini API path requires `google-genai` (and pydantic). Install with `pip install google-genai`."
        ) from e


    def load_gemini_api_key_from_dot_api(dot_api_path: str | Path) -> str | None:
        """Load GEMINI_API_KEY from a `.api` file.

        Expected format (per line):
          GEMINI_API_KEY = "xyz"
        Also supports:
          GEMINI_API_KEY=xyz
          export GEMINI_API_KEY="xyz"

        Returns None if file/key is missing or unreadable.
        """
        p = Path(dot_api_path)
        if not p.exists() or not p.is_file():
            return None
        try:
            raw = p.read_text(encoding="utf-8", errors="replace")
        except Exception:
            return None

        # Very small .env/.api style parser; ignores comments/blank lines.
        for line in raw.splitlines():
            s = line.strip()
            if not s or s.startswith("#"):
                continue
            if s.startswith("export "):
                s = s[len("export ") :].strip()

            m = re.match(r"^([A-Za-z_][A-Za-z0-9_]*)\s*=\s*(.*)$", s)
            if not m:
                continue
            k = m.group(1).strip()
            v = m.group(2).strip()
            if k not in {"GEMINI_API_KEY", "GOOGLE_API_KEY"}:
                continue

            # Strip quotes if present
            if (v.startswith('"') and v.endswith('"')) or (v.startswith("'") and v.endswith("'")):
                v = v[1:-1]
            v = v.strip()
            if v:
                return v
        return None

    api_key = os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")
    if not api_key:
        # Try project-root `.api` file (can remain ignored/untracked)
        api_key = load_gemini_api_key_from_dot_api(".api")
    if not api_key:
        raise EnvironmentError(
            "Missing Gemini API key. Set `GEMINI_API_KEY` (preferred) or `GOOGLE_API_KEY`."
        )
    # api_key = os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")
    # if not api_key:
    #     raise EnvironmentError(
    #         "Missing Gemini API key. Set `GEMINI_API_KEY` (preferred) or `GOOGLE_API_KEY`."
    #     )

    # Let user override model name; default to a fast/cheap-ish option.
    model_name = os.getenv("GEMINI_MODEL", "gemini-2.5-flash-lite")
    client = genai.Client(api_key=api_key)

    # Build a permissive schema that still forces JSON object + keys.
    fields: dict[str, tuple[Any, Any]] = {k: (Any, Field(default=None)) for k in required_keys}
    ResponseModel = create_model("AgentResponse", **fields)  # type: ignore

    def invoke(system_prompt: str, user_prompt: str) -> dict[str, Any]:
        resp = client.models.generate_content(
            model=model_name,
            contents=user_prompt,
            config=types.GenerateContentConfig(
                system_instruction=system_prompt,
                response_mime_type="application/json",
                response_schema=ResponseModel,
            ),
        )
        parsed = getattr(resp, "parsed", None)
        if parsed is None:
            # Fallback: parse raw text if schema parsing failed
            text = getattr(resp, "text", "") or ""
            parsed_obj = _deep_fix_json_strings(_parse_json(text))
        else:
            parsed_obj = parsed.model_dump() if hasattr(parsed, "model_dump") else dict(parsed)

        parsed_obj = _deep_fix_json_strings(parsed_obj)
        for k in required_keys:
            parsed_obj.setdefault(k, None)
        return parsed_obj
    print("Gemini API model loaded")
    return invoke
