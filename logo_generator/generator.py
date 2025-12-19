from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

from google import genai
from google.genai import types

from enterprise_auth import get_enterprise_context, init_enterprise_auth
from prompt import ColorPalette, LogoComposition, LogoStyleDimension, build_logo_prompt


@dataclass(frozen=True)
class LogoInputs:
    brand_name: Optional[str]
    logo_prompt: str
    logo_composition: LogoComposition
    logo_style_dimension: LogoStyleDimension
    reference_image: Optional[str]
    color_palette: ColorPalette


@dataclass(frozen=True)
class GeneratedLogo:
    prompt: str
    image_bytes: bytes


_DEFAULT_LOCATION = "us-central1"
_NANO_BANANA_MODEL_NAME = "gemini-3-pro-image-preview"


def generate_logo(
    inputs: LogoInputs,
    *,
    project_id: Optional[str] = None,
    location: str = _DEFAULT_LOCATION,
    model_name: str = _NANO_BANANA_MODEL_NAME,
) -> GeneratedLogo:
    """
    Generates exactly one logo image using Vertex AI image generation.

    Auth rule: this function is the ONLY entry point that initializes enterprise auth.
    """
    init_enterprise_auth(project_id=project_id, location=location)
    resolved_project_id, resolved_location = get_enterprise_context()

    prompt = build_logo_prompt(
        brand_name=inputs.brand_name,
        logo_prompt=inputs.logo_prompt,
        logo_composition=inputs.logo_composition,
        logo_style_dimension=inputs.logo_style_dimension,
        reference_image=inputs.reference_image,
        color_palette=inputs.color_palette,
        max_chars=2000,
    )

    client = genai.Client(
        vertexai=True,
        project=resolved_project_id,
        location=resolved_location,
    )

    contents = [
        types.Content(
            role="user",
            parts=[types.Part(text=prompt)],
        )
    ]

    generate_content_config = types.GenerateContentConfig(
        temperature=1,
        top_p=0.95,
        max_output_tokens=32768,
        response_modalities=["TEXT", "IMAGE"],
        safety_settings=[
            types.SafetySetting(category="HARM_CATEGORY_HATE_SPEECH", threshold="OFF"),
            types.SafetySetting(category="HARM_CATEGORY_DANGEROUS_CONTENT", threshold="OFF"),
            types.SafetySetting(category="HARM_CATEGORY_SEXUALLY_EXPLICIT", threshold="OFF"),
            types.SafetySetting(category="HARM_CATEGORY_HARASSMENT", threshold="OFF"),
        ],
        image_config=types.ImageConfig(
            aspect_ratio="1:1",
            image_size="1K",
            output_mime_type="image/png",
        ),
    )

    image_bytes = _first_image_from_stream(
        client=client,
        model=model_name,
        contents=contents,
        config=generate_content_config,
    )
    return GeneratedLogo(prompt=prompt, image_bytes=image_bytes)


def _first_image_from_stream(
    *,
    client: genai.Client,
    model: str,
    contents: list[types.Content],
    config: types.GenerateContentConfig,
) -> bytes:
    for chunk in client.models.generate_content_stream(
        model=model,
        contents=contents,
        config=config,
    ):
        image = _extract_image_bytes_from_chunk(chunk)
        if image is not None:
            return image
    raise RuntimeError("Model stream ended without returning an image.")


def _extract_image_bytes_from_chunk(chunk) -> Optional[bytes]:
    candidates = getattr(chunk, "candidates", None) or []
    for cand in candidates:
        content = getattr(cand, "content", None)
        parts = getattr(content, "parts", None) or []
        for part in parts:
            inline_data = getattr(part, "inline_data", None)
            if inline_data is None:
                continue
            data = getattr(inline_data, "data", None)
            if isinstance(data, (bytes, bytearray)):
                return bytes(data)
    return None


