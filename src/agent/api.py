"""
VLM API integration for OpenAI-compatible providers and Claude.

Supports:
- OpenAI (GPT-4o/GPT-4.1/o-series)
- OpenRouter (OpenAI-compatible endpoint)
- vLLM OpenAI-compatible server (local/remote)
- Claude 3.5 Sonnet (Anthropic endpoint)
"""

import os
import time
from typing import List, Dict, Any, Optional
from openai import OpenAI

# Try to import Anthropic (optional)
try:
    from anthropic import Anthropic
    ANTHROPIC_AVAILABLE = True
except ImportError:
    ANTHROPIC_AVAILABLE = False


# Model type detection
REASONING_MODELS = ["o1", "o3", "gpt-5"]  # o1-preview, o1-mini, o3-mini, gpt-5*
VISION_MODELS = ["gpt-4o", "gpt-4o-mini", "gpt-4.1"]
CLAUDE_MODELS = ["claude-3-5-sonnet-20241022", "claude-3-opus", "claude-3-sonnet"]

LAST_CALL_META: Dict[str, Any] = {}
CALL_HISTORY: List[Dict[str, Any]] = []
_CALL_SEQ: int = 0


def get_last_call_meta() -> Dict[str, Any]:
    """Return metadata from the most recent provider call."""
    return dict(LAST_CALL_META)


def get_call_history() -> List[Dict[str, Any]]:
    """Return metadata for all successful model calls in current process."""
    return [dict(x) for x in CALL_HISTORY]


def reset_call_tracking() -> None:
    """Clear call tracking buffers (last call + history)."""
    global LAST_CALL_META, CALL_HISTORY, _CALL_SEQ
    LAST_CALL_META = {}
    CALL_HISTORY = []
    _CALL_SEQ = 0


def _as_int(value: Any) -> int:
    try:
        if value is None:
            return 0
        return int(value)
    except Exception:
        return 0


def _get_nested(obj: Any, *keys: str, default: Any = None) -> Any:
    cur = obj
    for key in keys:
        if cur is None:
            return default
        if isinstance(cur, dict):
            cur = cur.get(key)
        else:
            cur = getattr(cur, key, None)
    return default if cur is None else cur


def _extract_usage(endpoint: str, response: Any) -> Dict[str, int]:
    usage = getattr(response, "usage", None)
    if usage is None and hasattr(response, "model_dump"):
        try:
            usage = response.model_dump().get("usage")
        except Exception:
            usage = None

    if endpoint == "chat.completions":
        input_tokens = _as_int(_get_nested(usage, "prompt_tokens"))
        output_tokens = _as_int(_get_nested(usage, "completion_tokens"))
        cached_input_tokens = _as_int(
            _get_nested(usage, "prompt_tokens_details", "cached_tokens")
        )
        reasoning_output_tokens = _as_int(
            _get_nested(usage, "completion_tokens_details", "reasoning_tokens")
        )
    else:
        input_tokens = _as_int(_get_nested(usage, "input_tokens"))
        output_tokens = _as_int(_get_nested(usage, "output_tokens"))
        cached_input_tokens = _as_int(
            _get_nested(usage, "input_tokens_details", "cached_tokens")
        )
        reasoning_output_tokens = _as_int(
            _get_nested(usage, "output_tokens_details", "reasoning_tokens")
        )

    total_tokens = _as_int(_get_nested(usage, "total_tokens"))
    if total_tokens <= 0:
        total_tokens = input_tokens + output_tokens

    uncached_input_tokens = max(input_tokens - cached_input_tokens, 0)
    return {
        "input_tokens": input_tokens,
        "cached_input_tokens": cached_input_tokens,
        "uncached_input_tokens": uncached_input_tokens,
        "output_tokens": output_tokens,
        "reasoning_output_tokens": reasoning_output_tokens,
        "total_tokens": total_tokens,
    }


def _record_call_meta(meta: Dict[str, Any]) -> Dict[str, Any]:
    global LAST_CALL_META, CALL_HISTORY, _CALL_SEQ
    _CALL_SEQ += 1
    record = dict(meta)
    record["call_index"] = _CALL_SEQ
    record["timestamp_unix"] = time.time()
    LAST_CALL_META = record
    CALL_HISTORY.append(record)
    return record


def is_reasoning_model(model: str) -> bool:
    """Check if model is a reasoning model (o1/o3 series)."""
    return any(rm in model for rm in REASONING_MODELS)


def call_gpt4o(
    prompt: str,
    images: List[str],
    model: str = "gpt-4o",
    max_tokens: int = 1000,
    temperature: float = 0.0,
    reasoning_effort: Optional[str] = None,
    response_schema: Optional[Dict[str, Any]] = None,
    extra_body: Optional[Dict[str, Any]] = None,
) -> str:
    """
    Call OpenAI models (GPT-4o or reasoning models) with text prompt and images.
    
    Args:
        prompt: Text prompt
        images: List of base64-encoded PNG images
        model: Model name (gpt-4o, gpt-4o-mini, o1-preview, o1-mini, o3-mini)
        max_tokens: Maximum response tokens
        temperature: Sampling temperature (ignored for reasoning models)
        reasoning_effort: For reasoning models: "minimal", "low", "medium", "high"
            (default: None = model decides)
        
    Returns:
        Raw text response from model
    """
    # Get API key from environment
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise ValueError("OPENAI_API_KEY environment variable not set")
    client = OpenAI(api_key=api_key)
    
    # Check if this is a reasoning model
    use_reasoning = is_reasoning_model(model)
    
    if use_reasoning:
        # Reasoning models use responses API
        return _call_reasoning_model(
            client=client,
            prompt=prompt,
            images=images,
            model=model,
            max_tokens=max_tokens,
            reasoning_effort=reasoning_effort,
            response_schema=response_schema,
            extra_body=extra_body,
            provider_name="openai",
        )
    else:
        # Standard vision models use chat completions
        return _call_vision_model(
            client=client,
            prompt=prompt,
            images=images,
            model=model,
            max_tokens=max_tokens,
            temperature=temperature,
            provider_name="openai",
            response_schema=response_schema,
            extra_body=extra_body,
        )


def _build_openrouter_client() -> OpenAI:
    api_key = os.getenv("OPENROUTER_API_KEY")
    if not api_key:
        raise ValueError("OPENROUTER_API_KEY environment variable not set")
    return OpenAI(api_key=api_key, base_url="https://openrouter.ai/api/v1")


def _build_vllm_client() -> OpenAI:
    api_key = os.getenv("VLLM_API_KEY", "EMPTY")
    base_url = os.getenv("VLLM_BASE_URL", "http://localhost:8000/v1")
    return OpenAI(api_key=api_key, base_url=base_url)


def call_openrouter(
    prompt: str,
    images: List[str],
    model: str,
    max_tokens: int = 1000,
    temperature: float = 0.0,
    response_schema: Optional[Dict[str, Any]] = None,
    extra_body: Optional[Dict[str, Any]] = None,
) -> str:
    """Call OpenRouter via OpenAI-compatible chat completions."""
    client = _build_openrouter_client()
    return _call_vision_model(
        client=client,
        prompt=prompt,
        images=images,
        model=model,
        max_tokens=max_tokens,
        temperature=temperature,
        provider_name="openrouter",
        response_schema=response_schema,
        extra_body=extra_body,
    )


def call_vllm(
    prompt: str,
    images: List[str],
    model: str,
    max_tokens: int = 1000,
    temperature: float = 0.0,
    response_schema: Optional[Dict[str, Any]] = None,
    extra_body: Optional[Dict[str, Any]] = None,
) -> str:
    """Call a vLLM OpenAI-compatible server."""
    client = _build_vllm_client()
    return _call_vision_model(
        client=client,
        prompt=prompt,
        images=images,
        model=model,
        max_tokens=max_tokens,
        temperature=temperature,
        provider_name="vllm",
        response_schema=response_schema,
        extra_body=extra_body,
    )


def _call_vision_model(
    client: OpenAI,
    prompt: str,
    images: List[str],
    model: str,
    max_tokens: int,
    temperature: float,
    provider_name: str = "openai",
    response_schema: Optional[Dict[str, Any]] = None,
    extra_body: Optional[Dict[str, Any]] = None,
) -> str:
    """Call standard vision models (GPT-4o, GPT-4o-mini) via chat completions."""
    # Build message content
    content = [{"type": "text", "text": prompt}]
    
    for img_b64 in images:
        content.append({
            "type": "image_url",
            "image_url": {
                "url": f"data:image/png;base64,{img_b64}",
                "detail": "high"  # high detail for waveform analysis
            }
        })
    
    # Call API
    try:
        request = {
            "model": model,
            "messages": [
                {
                    "role": "user",
                    "content": content
                }
            ],
            "temperature": temperature,
        }
        if "gpt-5" in model:
            request["max_completion_tokens"] = max_tokens
        else:
            request["max_tokens"] = max_tokens
        provider_extra_body: Dict[str, Any] = {}
        if extra_body:
            provider_extra_body.update(extra_body)

        if provider_name == "vllm":
            # Default Gemma thinking off for deterministic action-eval style calls.
            # Can still be overridden via VLM_EXTRA_BODY_JSON.
            model_name = str(model).lower()
            has_chat_kwargs = isinstance(provider_extra_body.get("chat_template_kwargs"), dict)
            if "gemma-4" in model_name and not has_chat_kwargs:
                provider_extra_body["chat_template_kwargs"] = {"enable_thinking": False}
            # vLLM structured outputs are commonly provided via guided decoding params.
            if response_schema is not None:
                provider_extra_body["guided_json"] = response_schema
        else:
            if response_schema is not None:
                request["response_format"] = {
                    "type": "json_schema",
                    "json_schema": {
                        "name": "spike_sorting_decision",
                        "schema": response_schema,
                        "strict": True,
                    },
                }
            elif "Output only in JSON schema" in prompt or "Output JSON:" in prompt:
                request["response_format"] = {"type": "json_object"}

        if provider_extra_body:
            request["extra_body"] = provider_extra_body

        if provider_name == "openrouter":
            referer = os.getenv("OPENROUTER_HTTP_REFERER")
            title = os.getenv("OPENROUTER_APP_TITLE", "SpikeSorting")
            headers: Dict[str, str] = {"X-Title": title}
            if referer:
                headers["HTTP-Referer"] = referer
            request["extra_headers"] = headers

        try:
            response = client.chat.completions.create(**request)
        except Exception as e:
            # Fallback for providers/models that do not support json_schema response_format
            if (
                provider_name != "vllm"
                and "response_format" in request
                and response_schema is not None
            ):
                fallback = dict(request)
                fallback["response_format"] = {"type": "json_object"}
                response = client.chat.completions.create(**fallback)
            else:
                raise e
        _record_call_meta(
            {
                "provider": provider_name,
                "endpoint": "chat.completions",
                "requested_model": model,
                "actual_model": getattr(response, "model", None),
                "usage": _extract_usage("chat.completions", response),
            }
        )
        
        return response.choices[0].message.content
    
    except Exception as e:
        print(f"[{provider_name} Vision Error] {e}")
        raise


def _call_reasoning_model(
    client: OpenAI,
    prompt: str,
    images: List[str],
    model: str,
    max_tokens: int,
    reasoning_effort: Optional[str],
    response_schema: Optional[Dict[str, Any]] = None,
    extra_body: Optional[Dict[str, Any]] = None,
    provider_name: str = "openai",
) -> str:
    """Call reasoning models (o1, o3) via responses API."""
    # Build input content
    content = [{"type": "input_text", "text": prompt}]
    
    for img_b64 in images:
        content.append({
            "type": "input_image",
            "image_url": f"data:image/png;base64,{img_b64}",
            "detail": "high"
        })
    
    # Build reasoning config
    reasoning_config = {}
    if reasoning_effort is not None:
        reasoning_config["effort"] = reasoning_effort
    
    # Call API
    try:
        request: Dict[str, Any] = {
            "model": model,
            "input": [{
                "role": "user",
                "content": content
            }],
            "max_output_tokens": max_tokens,
            "reasoning": reasoning_config if reasoning_config else None,
        }
        if response_schema is not None:
            request["text"] = {
                "format": {
                    "type": "json_schema",
                    "name": "spike_sorting_decision",
                    "schema": response_schema,
                    "strict": True,
                }
            }
        if extra_body:
            request["extra_body"] = extra_body

        response = client.responses.create(**request)
        _record_call_meta(
            {
                "provider": provider_name,
                "endpoint": "responses",
                "requested_model": model,
                "actual_model": getattr(response, "model", None),
                "usage": _extract_usage("responses", response),
            }
        )
        
        return response.output_text
    
    except Exception as e:
        print(f"[OpenAI Reasoning Error] {e}")
        raise


def call_claude(
    prompt: str,
    images: List[str],
    model: str = "claude-3-5-sonnet-20241022",
    max_tokens: int = 1000,
    temperature: float = 0.0,
) -> str:
    """
    Call Claude 3.5 Sonnet with text prompt and base64-encoded images.
    
    Args:
        prompt: Text prompt
        images: List of base64-encoded PNG images
        model: Model name
        max_tokens: Maximum response tokens
        temperature: Sampling temperature
        
    Returns:
        Raw text response from model
    """
    if not ANTHROPIC_AVAILABLE:
        raise ImportError("anthropic package not installed. Install with: pip install anthropic")
    
    # Get API key from environment
    api_key = os.getenv("ANTHROPIC_API_KEY")
    if not api_key:
        raise ValueError("ANTHROPIC_API_KEY environment variable not set")
    
    client = Anthropic(api_key=api_key)
    
    # Build message content
    content = [{"type": "text", "text": prompt}]
    
    for img_b64 in images:
        content.append({
            "type": "image",
            "source": {
                "type": "base64",
                "media_type": "image/png",
                "data": img_b64,
            }
        })
    
    # Call API
    try:
        response = client.messages.create(
            model=model,
            max_tokens=max_tokens,
            temperature=temperature,
            messages=[
                {
                    "role": "user",
                    "content": content
                }
            ]
        )
        input_tokens = _as_int(_get_nested(response, "usage", "input_tokens"))
        output_tokens = _as_int(_get_nested(response, "usage", "output_tokens"))
        _record_call_meta(
            {
                "provider": "anthropic",
                "endpoint": "messages",
                "requested_model": model,
                "actual_model": getattr(response, "model", None),
                "usage": {
                    "input_tokens": input_tokens,
                    "cached_input_tokens": 0,
                    "uncached_input_tokens": input_tokens,
                    "output_tokens": output_tokens,
                    "reasoning_output_tokens": 0,
                    "total_tokens": input_tokens + output_tokens,
                },
            }
        )
        
        return response.content[0].text
    
    except Exception as e:
        print(f"[Claude Error] {e}")
        raise


def call_vlm(
    prompt: str,
    images: List[str],
    provider: str = "gpt4o",
    model: Optional[str] = None,
    max_tokens: int = 1000,
    temperature: float = 0.0,
    reasoning_effort: Optional[str] = None,
    response_schema: Optional[Dict[str, Any]] = None,
    extra_body: Optional[Dict[str, Any]] = None,
) -> str:
    """
    Unified VLM API call - routes to GPT-4o/reasoning models or Claude.
    
    Args:
        prompt: Text prompt
        images: List of base64-encoded PNG images
        provider: "gpt4o" (includes reasoning models) or "claude"
        model: Optional model override
            - For gpt4o: "gpt-4o", "gpt-4o-mini", "o1-preview", "o1-mini", "o3-mini"
            - For claude: "claude-3-5-sonnet-20241022"
        max_tokens: Maximum response tokens
        temperature: Sampling temperature (ignored for reasoning models)
        reasoning_effort: For reasoning models only: "minimal", "low", "medium", "high"
            - None: model decides effort automatically
            - "low": faster, less thorough
            - "medium": balanced
            - "high": slower, more thorough
        
    Returns:
        Raw text response from model
        
    Examples:
        # Standard GPT-4o with temperature
        call_vlm(prompt, images, provider="gpt4o", model="gpt-4o", temperature=0.0)
        
        # Reasoning model with effort control
        call_vlm(prompt, images, provider="gpt4o", model="o1-preview", reasoning_effort="medium")
        
        # Claude
        call_vlm(prompt, images, provider="claude")
    """
    if provider == "gpt4o":
        model = model or "gpt-4o"
        return call_gpt4o(
            prompt,
            images,
            model,
            max_tokens,
            temperature,
            reasoning_effort,
            response_schema=response_schema,
            extra_body=extra_body,
        )

    elif provider == "openrouter":
        model = model or "qwen/qwen3.5-vl-4b-instruct"
        return call_openrouter(
            prompt=prompt,
            images=images,
            model=model,
            max_tokens=max_tokens,
            temperature=temperature,
            response_schema=response_schema,
            extra_body=extra_body,
        )

    elif provider == "vllm":
        model = model or "Qwen/Qwen3.5-VL-4B-Instruct"
        return call_vllm(
            prompt=prompt,
            images=images,
            model=model,
            max_tokens=max_tokens,
            temperature=temperature,
            response_schema=response_schema,
            extra_body=extra_body,
        )

    elif provider == "claude":
        model = model or "claude-3-5-sonnet-20241022"
        return call_claude(prompt, images, model, max_tokens, temperature)

    else:
        raise ValueError(
            f"Unknown provider: {provider}. Use 'gpt4o', 'openrouter', 'vllm', or 'claude'"
        )
