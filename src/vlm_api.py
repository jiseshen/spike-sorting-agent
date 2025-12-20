"""
VLM API integration for OpenAI (GPT-4o, o1, o3) and Claude 3.5 Sonnet.

Supports:
- GPT-4o/GPT-4o-mini: Standard vision models with temperature control
- o1-preview/o1-mini/o3-mini: Reasoning models with effort control
- Claude 3.5 Sonnet: Anthropic's vision model
"""

import os
import base64
import json
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
REASONING_MODELS = ["o1", "o3", "gpt-5.1"]  # o1-preview, o1-mini, o3-mini, etc.
VISION_MODELS = ["gpt-4o", "gpt-4o-mini", "gpt-4.1"]
CLAUDE_MODELS = ["claude-3-5-sonnet-20241022", "claude-3-opus", "claude-3-sonnet"]


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
) -> str:
    """
    Call OpenAI models (GPT-4o or reasoning models) with text prompt and images.
    
    Args:
        prompt: Text prompt
        images: List of base64-encoded PNG images
        model: Model name (gpt-4o, gpt-4o-mini, o1-preview, o1-mini, o3-mini)
        max_tokens: Maximum response tokens
        temperature: Sampling temperature (ignored for reasoning models)
        reasoning_effort: For reasoning models: "low", "medium", "high" (default: None = model decides)
        
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
            client, prompt, images, model, max_tokens, reasoning_effort
        )
    else:
        # Standard vision models use chat completions
        return _call_vision_model(
            client, prompt, images, model, max_tokens, temperature
        )


def _call_vision_model(
    client: OpenAI,
    prompt: str,
    images: List[str],
    model: str,
    max_tokens: int,
    temperature: float,
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
        response = client.chat.completions.create(
            model=model,
            messages=[
                {
                    "role": "user",
                    "content": content
                }
            ],
            max_tokens=max_tokens,
            temperature=temperature,
        )
        
        return response.choices[0].message.content
    
    except Exception as e:
        print(f"[OpenAI Vision Error] {e}")
        raise


def _call_reasoning_model(
    client: OpenAI,
    prompt: str,
    images: List[str],
    model: str,
    max_tokens: int,
    reasoning_effort: Optional[str],
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
        response = client.responses.create(
            model=model,
            input=[{
                "role": "user",
                "content": content
            }],
            max_output_tokens=max_tokens,
            reasoning=reasoning_config if reasoning_config else None,
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
        reasoning_effort: For reasoning models only: "low", "medium", "high"
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
            prompt, images, model, max_tokens, temperature, reasoning_effort
        )
    
    elif provider == "claude":
        model = model or "claude-3-5-sonnet-20241022"
        return call_claude(prompt, images, model, max_tokens, temperature)
    
    else:
        raise ValueError(f"Unknown provider: {provider}. Use 'gpt4o' or 'claude'")
