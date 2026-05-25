import os
import json
import threading
import time
from datetime import datetime, timezone
from typing import Dict, Optional, Tuple, Any

from dotenv import load_dotenv
from utils.prompt_utils import build_prompt


# =============================================================================
# utils/llm_utils.py — Multi-Engine LLM Caller
# =============================================================================
#
# Supports 6 LLM engines via a unified generate_llm_response() interface.
# Includes retry logic and token usage logging via LLM_CALL_LOG.
#
#   openai        : OpenAI API (GPT series)
#   gemini        : Google Gemini via AI Studio
#   vertex        : Claude via GCP Vertex AI (Anthropic SDK)
#   vertex_openai : Llama / Qwen / Mistral via Vertex AI Partner MaaS
#   anthropic     : Claude via AWS Bedrock
#   together      : Together AI (OpenAI-compatible)
#
# Usage logging: set LLM_CALL_LOG=data/logs/llm_usage.jsonl in .env.local
# to append one JSONL record per call (ts, engine, model, tokens, latency).
#
# =============================================================================


_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
load_dotenv(dotenv_path=os.path.join(_PROJECT_ROOT, ".env.local"))
PROMPT_SEPARATOR = "<System Prompt/User Prompt>"
_LOG_LOCK = threading.Lock()


def _append_usage_log(record: dict) -> None:
    log_path = os.environ.get("LLM_CALL_LOG")
    if not log_path:
        return
    try:
        os.makedirs(os.path.dirname(log_path) or ".", exist_ok=True)
        with _LOG_LOCK:
            with open(log_path, "a", encoding="utf-8") as f:
                f.write(json.dumps(record, ensure_ascii=False) + "\n")
    except Exception as e:
        # Logger errors must not break LLM calls
        print(f"  [usage log warning] {e}")


_DEFAULT_MODEL_CONFIG_PATH = os.path.join(_PROJECT_ROOT, "config", "config_default_model.json")
_DEFAULT_MODELS_CACHE: dict | None = None

def _load_default_models() -> dict:
    global _DEFAULT_MODELS_CACHE
    if _DEFAULT_MODELS_CACHE is None:
        try:
            with open(_DEFAULT_MODEL_CONFIG_PATH, "r", encoding="utf-8") as f:
                _DEFAULT_MODELS_CACHE = json.load(f)
        except FileNotFoundError:
            raise RuntimeError(f"config_default_model.json not found at: {_DEFAULT_MODEL_CONFIG_PATH}")
        except Exception as e:
            raise RuntimeError(f"Failed to load config_default_model.json: {e}")
    return _DEFAULT_MODELS_CACHE


def get_pipeline_model(pipeline: str) -> dict:
    """Return {"model_type": ..., "model_name": ...} for the given pipeline key."""
    models = _load_default_models()
    if pipeline not in models:
        raise RuntimeError(f"No model config for '{pipeline}' in config_default_model.json")
    return models[pipeline]


def _split_prompt(prompt: str):
    if PROMPT_SEPARATOR in prompt:
        parts = prompt.split(PROMPT_SEPARATOR, 1)
        return parts[0].strip(), parts[1].strip()
    return "", prompt.strip()


def _log_call(engine: str, model: str, system_content: str, user_content: str):
    sys_preview  = system_content[:50].replace("\n", " ") if system_content else "(none)"
    user_preview = user_content[:50].replace("\n", " ")   if user_content  else "(none)"
    print(f">>> [{engine}] {model}")
    print(f"    system: {sys_preview}")
    print(f"    user  : {user_preview}")



# ============================================================================================================================
# OPENAI
# ============================================================================================================================
def call_openai(prompt_filename: str, variables: Dict[str, str], model_name: str, use_api_defaults: bool = False) -> Tuple[str, str, str, str]:
    try:
        from openai import OpenAI
    except ImportError:
        raise RuntimeError("openai package not installed")

    key = os.getenv("OPENAI_API_KEY")
    if not key:
        raise RuntimeError("OPENAI_API_KEY not set")

    prompt = build_prompt(prompt_filename, variables)
    system_content, user_content = _split_prompt(prompt)

    client = OpenAI(api_key=key)
    kwargs = dict(
        model=model_name,
        messages=[
            {"role": "system", "content": system_content},
            {"role": "user", "content": user_content},
        ],
        max_completion_tokens=8000,
    )
    # GPT-5 family (reasoning models) only accepts the default temperature=1.
    _no_temperature = model_name.startswith("gpt-5")
    if not use_api_defaults and not _no_temperature:
        kwargs["temperature"] = 0

    t0 = time.time()
    status = "ok"
    response = None
    text = ""
    try:
        response = client.chat.completions.create(**kwargs)
        text = response.choices[0].message.content or ""
    except Exception as e:
        status = f"error:{type(e).__name__}"
        raise
    finally:
        usage = getattr(response, "usage", None) if response is not None else None
        _append_usage_log({
            "ts": datetime.now(timezone.utc).isoformat(),
            "engine": "openai",
            "model": model_name,
            "prompt_filename": prompt_filename,
            "system_chars": len(system_content),
            "user_chars":   len(user_content),
            "output_chars": len(text),
            "input_tokens":  getattr(usage, "prompt_tokens", None) if usage else None,
            "output_tokens": getattr(usage, "completion_tokens", None) if usage else None,
            "total_tokens":  getattr(usage, "total_tokens", None) if usage else None,
            "latency_ms": int((time.time() - t0) * 1000),
            "status": status,
        })

    return text, model_name, system_content, user_content



# ============================================================================================================================
# GEMINI
# ============================================================================================================================
_GEMINI_CLIENT_LOCK = threading.Lock()
_GEMINI_CLIENT_CACHE: dict = {}

def _get_gemini_client(api_key: str):
    with _GEMINI_CLIENT_LOCK:
        if api_key not in _GEMINI_CLIENT_CACHE:
            from google import genai
            _GEMINI_CLIENT_CACHE[api_key] = genai.Client(api_key=api_key)
        return _GEMINI_CLIENT_CACHE[api_key]


def call_gemini(prompt_filename: str, variables: Dict[str, str], model_name: str, use_api_defaults: bool = False) -> Tuple[str, str, str, str]:
    try:
        from google.genai import types as genai_types
        from google.genai.errors import ServerError, ClientError
    except ImportError:
        raise RuntimeError("google-genai package not installed. Run: pip install google-genai")

    key = os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")
    if not key:
        raise RuntimeError("GEMINI_API_KEY/GOOGLE_API_KEY not set")

    prompt = build_prompt(prompt_filename, variables)
    system_content, user_content = _split_prompt(prompt)

    client = _get_gemini_client(key)

    # Pipeline 5 (use_api_defaults=True): cap output at 8000 for uniformity with other engines.
    # Deterministic mode (evaluation/pipeline 1-4): leave uncapped so grader JSON / extraction
    # output isn't truncated for long rubrics or detailed analyses.
    config_kwargs: dict = {}
    if use_api_defaults:
        config_kwargs["max_output_tokens"] = 8000
    if system_content:
        config_kwargs["system_instruction"] = system_content
    if not use_api_defaults:
        config_kwargs["temperature"] = 0

    config = genai_types.GenerateContentConfig(**config_kwargs) if config_kwargs else None

    t0 = time.time()
    status = "ok"
    response = None
    text = ""
    try:
        max_retries = 5
        for attempt in range(max_retries):
            try:
                response = client.models.generate_content(
                    model=model_name,
                    contents=user_content,
                    config=config,
                )
                break
            except ServerError as e:
                if attempt < max_retries - 1:
                    wait = 2 ** attempt * 10  # 10, 20, 40, 80 sec
                    print(f"  [Gemini 503] attempt {attempt+1}/{max_retries}, retrying in {wait}s...")
                    time.sleep(wait)
                else:
                    status = f"error:ServerError"
                    raise
            except ClientError as e:
                if "429" in str(e) and attempt < max_retries - 1:
                    wait = 2 ** attempt * 15  # 15, 30, 60, 120 sec
                    print(f"  [Gemini 429] attempt {attempt+1}/{max_retries}, retrying in {wait}s...")
                    time.sleep(wait)
                else:
                    status = f"error:ClientError"
                    raise
        try:
            text = response.text or ""
        except Exception:
            text = str(response)
    except Exception:
        # Status already captured above where we know which error type
        if status == "ok":
            status = "error:unknown"
        raise
    finally:
        meta = getattr(response, "usage_metadata", None) if response is not None else None
        _append_usage_log({
            "ts": datetime.now(timezone.utc).isoformat(),
            "engine": "gemini",
            "model": model_name,
            "prompt_filename": prompt_filename,
            "system_chars": len(system_content),
            "user_chars":   len(user_content),
            "output_chars": len(text),
            "input_tokens":  getattr(meta, "prompt_token_count", None) if meta else None,
            "output_tokens": getattr(meta, "candidates_token_count", None) if meta else None,
            "total_tokens":  getattr(meta, "total_token_count", None) if meta else None,
            "latency_ms": int((time.time() - t0) * 1000),
            "status": status,
        })
    return text, model_name, system_content, user_content



# ============================================================================================================================
# Anthropic
# ============================================================================================================================
_BEDROCK_CLIENT_LOCK = threading.Lock()
_BEDROCK_CLIENT_CACHE: dict = {}

def _get_bedrock_client(region: str, access_key: str, secret_key: str):
    cache_key = (region, access_key)
    with _BEDROCK_CLIENT_LOCK:
        if cache_key not in _BEDROCK_CLIENT_CACHE:
            import boto3
            _BEDROCK_CLIENT_CACHE[cache_key] = boto3.client(
                "bedrock-runtime",
                region_name=region,
                aws_access_key_id=access_key,
                aws_secret_access_key=secret_key,
            )
        return _BEDROCK_CLIENT_CACHE[cache_key]


def call_anthropic_bedrock(prompt_filename: str, variables: Dict[str, str], model_name: str, use_api_defaults: bool = False) -> Tuple[str, str, str, str]:
    """Invoke a Claude model on AWS Bedrock using the Anthropic messages format."""
    try:
        from botocore.exceptions import ClientError
    except ImportError:
        raise RuntimeError("boto3 not installed. Run: pip install boto3")

    access_key = os.getenv("AWS_ACCESS_KEY") or os.getenv("AWS_ACCESS_KEY_ID")
    secret_key = os.getenv("AWS_SECRET_KEY") or os.getenv("AWS_SECRET_ACCESS_KEY")
    region     = os.getenv("AWS_REGION") or os.getenv("AWS_DEFAULT_REGION") or "us-east-1"
    if not access_key or not secret_key:
        raise RuntimeError("AWS_ACCESS_KEY / AWS_SECRET_KEY not set")

    prompt = build_prompt(prompt_filename, variables)
    system_content, user_content = _split_prompt(prompt)

    client = _get_bedrock_client(region, access_key, secret_key)

    body: dict = {
        "anthropic_version": "bedrock-2023-05-31",
        "messages": [{"role": "user", "content": user_content}],
        "max_tokens": 8000,
    }
    if system_content:
        body["system"] = system_content
    if not use_api_defaults:
        body["temperature"] = 0

    t0 = time.time()
    status = "ok"
    text = ""
    usage = None
    try:
        max_retries = 5
        for attempt in range(max_retries):
            try:
                resp = client.invoke_model(modelId=model_name, body=json.dumps(body))
                payload = json.loads(resp["body"].read())
                text = "".join(b.get("text", "") for b in payload.get("content", []) if b.get("type") == "text")
                usage = payload.get("usage", {})
                break
            except ClientError as e:
                code = e.response.get("Error", {}).get("Code", "")
                if code in ("ThrottlingException", "ServiceUnavailableException") and attempt < max_retries - 1:
                    wait = 2 ** attempt * 10
                    print(f"  [Bedrock {code}] attempt {attempt+1}/{max_retries}, retrying in {wait}s...")
                    time.sleep(wait)
                else:
                    status = f"error:{code or 'ClientError'}"
                    raise
    except Exception:
        if status == "ok":
            status = "error:unknown"
        raise
    finally:
        _append_usage_log({
            "ts": datetime.now(timezone.utc).isoformat(),
            "engine": "bedrock",
            "model": model_name,
            "prompt_filename": prompt_filename,
            "system_chars": len(system_content),
            "user_chars":   len(user_content),
            "output_chars": len(text),
            "input_tokens":  (usage or {}).get("input_tokens"),
            "output_tokens": (usage or {}).get("output_tokens"),
            "total_tokens":  ((usage or {}).get("input_tokens", 0) + (usage or {}).get("output_tokens", 0)) if usage else None,
            "latency_ms": int((time.time() - t0) * 1000),
            "status": status,
        })

    return text, model_name, system_content, user_content



# ============================================================================================================================
# VERTEX
# ============================================================================================================================
_VERTEX_CLIENT_LOCK = threading.Lock()
_VERTEX_CLIENT_CACHE: dict = {}

def _get_vertex_client(region: str, project_id: str):
    cache_key = (region, project_id)
    with _VERTEX_CLIENT_LOCK:
        if cache_key not in _VERTEX_CLIENT_CACHE:
            from anthropic import AnthropicVertex
            _VERTEX_CLIENT_CACHE[cache_key] = AnthropicVertex(region=region, project_id=project_id)
        return _VERTEX_CLIENT_CACHE[cache_key]


def call_vertex_anthropic(prompt_filename: str, variables: Dict[str, str], model_name: str, use_api_defaults: bool = False) -> Tuple[str, str, str, str]:
    """Invoke a Claude model on GCP Vertex AI via the Anthropic SDK's Vertex client.
    Auth uses Application Default Credentials (set up via `gcloud auth application-default login`
    or GOOGLE_APPLICATION_CREDENTIALS pointing to a service account JSON).
    """
    try:
        from anthropic import APIStatusError
    except ImportError:
        raise RuntimeError("anthropic SDK not installed. Run: pip install 'anthropic[vertex]'")

    project_id = os.getenv("GOOGLE_CLOUD_PROJECT") or os.getenv("GCP_PROJECT_ID")
    region     = os.getenv("VERTEX_REGION") or os.getenv("GOOGLE_CLOUD_REGION") or "us-east5"
    if not project_id:
        raise RuntimeError("GOOGLE_CLOUD_PROJECT (or GCP_PROJECT_ID) not set")

    prompt = build_prompt(prompt_filename, variables)
    system_content, user_content = _split_prompt(prompt)

    client = _get_vertex_client(region, project_id)

    kwargs: dict = {
        "model": model_name,
        "messages": [{"role": "user", "content": user_content}],
        "max_tokens": 8000,
    }
    if system_content:
        kwargs["system"] = system_content
    # Opus 4.7+ deprecated `temperature` (sampling controlled by extended thinking).
    # Drop it for affected model lines; other Claude models still accept it.
    _no_temperature = "opus-4-7" in model_name
    if not use_api_defaults and not _no_temperature:
        kwargs["temperature"] = 0

    t0 = time.time()
    status = "ok"
    text = ""
    usage = None
    try:
        max_retries = 5
        for attempt in range(max_retries):
            try:
                resp = client.messages.create(**kwargs)
                text = "".join(b.text for b in resp.content if getattr(b, "type", None) == "text")
                usage = getattr(resp, "usage", None)
                break
            except APIStatusError as e:
                if e.status_code in (429, 503, 529) and attempt < max_retries - 1:
                    wait = 2 ** attempt * 10
                    print(f"  [Vertex {e.status_code}] attempt {attempt+1}/{max_retries}, retrying in {wait}s...")
                    time.sleep(wait)
                else:
                    status = f"error:{e.status_code}"
                    raise
    except Exception:
        if status == "ok":
            status = "error:unknown"
        raise
    finally:
        _append_usage_log({
            "ts": datetime.now(timezone.utc).isoformat(),
            "engine": "vertex",
            "model": model_name,
            "prompt_filename": prompt_filename,
            "system_chars": len(system_content),
            "user_chars":   len(user_content),
            "output_chars": len(text),
            "input_tokens":  getattr(usage, "input_tokens", None) if usage else None,
            "output_tokens": getattr(usage, "output_tokens", None) if usage else None,
            "total_tokens":  (getattr(usage, "input_tokens", 0) + getattr(usage, "output_tokens", 0)) if usage else None,
            "latency_ms": int((time.time() - t0) * 1000),
            "status": status,
        })

    return text, model_name, system_content, user_content



# ============================================================================================================================
# VERTEX_OPENAI
# ============================================================================================================================
_VERTEX_OAI_CLIENT_LOCK = threading.Lock()
_VERTEX_OAI_CLIENT_CACHE: dict = {}

def _get_vertex_openai_client(region: str, project_id: str):
    """OpenAI-compatible client pointed at Vertex AI's partner-model endpoint.
    Used for Llama / Qwen / Mistral / etc — anything served via Vertex MaaS that
    isn't Anthropic (Anthropic has its own client via call_vertex_anthropic).

    Token refresh is automatic each call (ADC tokens expire after ~1h).
    """
    from openai import OpenAI
    from google.auth import default
    from google.auth.transport.requests import Request

    creds, _ = default()
    creds.refresh(Request())

    cache_key = (region, project_id)
    with _VERTEX_OAI_CLIENT_LOCK:
        # Always rebuild — token refresh needed
        base_url = f"https://{region}-aiplatform.googleapis.com/v1beta1/projects/{project_id}/locations/{region}/endpoints/openapi"
        _VERTEX_OAI_CLIENT_CACHE[cache_key] = OpenAI(api_key=creds.token, base_url=base_url)
        return _VERTEX_OAI_CLIENT_CACHE[cache_key]


def call_vertex_openai(prompt_filename: str, variables: Dict[str, str], model_name: str, use_api_defaults: bool = False) -> Tuple[str, str, str, str]:
    """Call a non-Anthropic partner MaaS model on Vertex AI via its OpenAI-compatible endpoint.
    `model_name` should be in '<publisher>/<model>' form, e.g. 'meta/llama-4-maverick-17b-128e-instruct-maas'.
    """
    project_id = os.getenv("GOOGLE_CLOUD_PROJECT") or os.getenv("GCP_PROJECT_ID")
    # Vertex partner MaaS is region-pinned per model — caller chooses via env or config.
    # Llama 4 lives in us-east5; Qwen in different region; us-central1 is most common default.
    region     = os.getenv("VERTEX_OAI_REGION") or "us-east5"
    if not project_id:
        raise RuntimeError("GOOGLE_CLOUD_PROJECT (or GCP_PROJECT_ID) not set")

    prompt = build_prompt(prompt_filename, variables)
    system_content, user_content = _split_prompt(prompt)

    client = _get_vertex_openai_client(region, project_id)

    messages = []
    if system_content:
        messages.append({"role": "system", "content": system_content})
    messages.append({"role": "user", "content": user_content})

    # Vertex partner MaaS models cap max_tokens (Llama 4 Maverick: 8192).
    # Stay under the lowest known cap on this engine.
    kwargs = dict(model=model_name, messages=messages, max_tokens=8000)
    if not use_api_defaults:
        kwargs["temperature"] = 0

    t0 = time.time()
    status = "ok"
    response = None
    text = ""
    try:
        response = client.chat.completions.create(**kwargs)
        text = response.choices[0].message.content or ""
    except Exception as e:
        status = f"error:{type(e).__name__}"
        raise
    finally:
        usage = getattr(response, "usage", None) if response is not None else None
        _append_usage_log({
            "ts": datetime.now(timezone.utc).isoformat(),
            "engine": "vertex_openai",
            "model": model_name,
            "prompt_filename": prompt_filename,
            "system_chars": len(system_content),
            "user_chars":   len(user_content),
            "output_chars": len(text),
            "input_tokens":  getattr(usage, "prompt_tokens", None) if usage else None,
            "output_tokens": getattr(usage, "completion_tokens", None) if usage else None,
            "total_tokens":  getattr(usage, "total_tokens", None) if usage else None,
            "latency_ms": int((time.time() - t0) * 1000),
            "status": status,
        })

    return text, model_name, system_content, user_content



# ============================================================================================================================
# TOGETHER
# ============================================================================================================================
def call_together(prompt_filename: str, variables: Dict[str, str], model_name: str, use_api_defaults: bool = False) -> Tuple[str, str, str, str]:
    """Together AI uses the OpenAI Chat Completions schema with a custom base_url."""
    try:
        from openai import OpenAI
    except ImportError:
        raise RuntimeError("openai package not installed")

    key = os.getenv("TOGETHER_API_KEY")
    if not key:
        raise RuntimeError("TOGETHER_API_KEY not set")

    prompt = build_prompt(prompt_filename, variables)
    system_content, user_content = _split_prompt(prompt)

    client = OpenAI(api_key=key, base_url="https://api.together.xyz/v1")
    messages = []
    if system_content:
        messages.append({"role": "system", "content": system_content})
    messages.append({"role": "user", "content": user_content})

    kwargs = dict(model=model_name, messages=messages, max_tokens=8000)
    if not use_api_defaults:
        kwargs["temperature"] = 0

    from openai import RateLimitError, APIStatusError

    t0 = time.time()
    status = "ok"
    response = None
    text = ""
    try:
        max_retries = 5
        for attempt in range(max_retries):
            try:
                response = client.chat.completions.create(**kwargs)
                text = response.choices[0].message.content or ""
                break
            except (RateLimitError, APIStatusError) as e:
                # Together throttles aggressively on some models (e.g. DeepSeek V4 Pro).
                # Retry with exponential backoff on 429/503/529.
                code = getattr(e, "status_code", None)
                if code in (429, 503, 529) and attempt < max_retries - 1:
                    wait = 2 ** attempt * 10  # 10, 20, 40, 80, 160s
                    print(f"  [Together {code}] attempt {attempt+1}/{max_retries}, retrying in {wait}s...")
                    time.sleep(wait)
                else:
                    status = f"error:{code or type(e).__name__}"
                    raise
    except Exception as e:
        if status == "ok":
            status = f"error:{type(e).__name__}"
        raise
    finally:
        usage = getattr(response, "usage", None) if response is not None else None
        _append_usage_log({
            "ts": datetime.now(timezone.utc).isoformat(),
            "engine": "together",
            "model": model_name,
            "prompt_filename": prompt_filename,
            "system_chars": len(system_content),
            "user_chars":   len(user_content),
            "output_chars": len(text),
            "input_tokens":  getattr(usage, "prompt_tokens", None) if usage else None,
            "output_tokens": getattr(usage, "completion_tokens", None) if usage else None,
            "total_tokens":  getattr(usage, "total_tokens", None) if usage else None,
            "latency_ms": int((time.time() - t0) * 1000),
            "status": status,
        })

    return text, model_name, system_content, user_content



# ============================================================================================================================
# LLM RESPONSE
# ============================================================================================================================
def generate_llm_response(
    prompt_filename: str,
    variables: Dict[str, str],
    model_type: str,
    model_name: str,
    use_api_defaults: bool = False,
) -> Optional[str]:
    """
    Call the specified LLM and return the response text.

    Args:
        prompt_filename: Name of the prompt template in prompts/
        variables: Placeholder substitutions for the template
        model_type: "openai" | "gemini" | "vertex" (Claude via GCP Vertex) | "vertex_openai" (Llama/etc via Vertex partner MaaS) | "anthropic" (Claude via AWS Bedrock) | "together"
        model_name: Exact provider-side model ID
        use_api_defaults: If True, use API defaults (no temperature/token overrides). Use for pipeline 5.
    """
    if model_type == "openai":
        text, resolved_model, system_content, user_content = call_openai(prompt_filename, variables, model_name, use_api_defaults)
        _log_call("openai", resolved_model, system_content, user_content)
        return text

    if model_type == "gemini":
        text, resolved_model, system_content, user_content = call_gemini(prompt_filename, variables, model_name, use_api_defaults)
        _log_call("gemini", resolved_model, system_content, user_content)
        return text

    if model_type == "vertex":
        text, resolved_model, system_content, user_content = call_vertex_anthropic(prompt_filename, variables, model_name, use_api_defaults)
        _log_call("vertex", resolved_model, system_content, user_content)
        return text

    if model_type == "vertex_openai":
        text, resolved_model, system_content, user_content = call_vertex_openai(prompt_filename, variables, model_name, use_api_defaults)
        _log_call("vertex_openai", resolved_model, system_content, user_content)
        return text

    if model_type == "anthropic":
        text, resolved_model, system_content, user_content = call_anthropic_bedrock(prompt_filename, variables, model_name, use_api_defaults)
        _log_call("bedrock", resolved_model, system_content, user_content)
        return text

    if model_type == "together":
        text, resolved_model, system_content, user_content = call_together(prompt_filename, variables, model_name, use_api_defaults)
        _log_call("together", resolved_model, system_content, user_content)
        return text

    raise RuntimeError(f"Unsupported model_type: '{model_type}'. Use 'openai', 'gemini', 'vertex', 'vertex_openai', 'anthropic', or 'together'.")