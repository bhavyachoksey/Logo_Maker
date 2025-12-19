from __future__ import annotations

from typing import Literal, Optional

from langchain.prompts import PromptTemplate

LogoComposition = Literal["icon_only", "typography_only", "icon_and_typography"]
LogoStyleDimension = Literal["icon", "2d", "3d"]
ColorPalette = Literal["pastel", "neutral", "bright"]

_PROMPT_TEMPLATE = PromptTemplate(
    input_variables=[
        "brand_name",
        "logo_prompt",
        "logo_composition",
        "logo_style_dimension",
        "reference_image",
        "color_palette",
        "composition_rule",
        "style_rule",
        "color_rule",
        "reference_rule",
        "brand_rule",
    ],
    template=(
        "Design a centered logo. Background: plain transparent. No mockups, no UI, no watermarks, "
        "no people, no photography.\n"
        "Composition: {logo_composition}. {composition_rule}\n"
        "Style: {logo_style_dimension}. {style_rule}\n"
        "Colors: {color_palette}. {color_rule}\n"
        "{reference_rule}"
        "Brand: {brand_rule}\n"
        "Brief: {logo_prompt}\n"
        "Output: 1 clean logo image."
    ),
)


def build_logo_prompt(
    *,
    brand_name: Optional[str],
    logo_prompt: str,
    logo_composition: LogoComposition,
    logo_style_dimension: LogoStyleDimension,
    reference_image: Optional[str],
    color_palette: ColorPalette,
    max_chars: int = 2000,
) -> str:
    if not logo_prompt or not logo_prompt.strip():
        raise ValueError("logo_prompt is required.")

    composition_rule = {
        "icon_only": "No text.",
        "typography_only": "Brand name dominant; no icon.",
        "icon_and_typography": "Icon + brand name.",
    }[logo_composition]

    style_rule = {
        "icon": "Flat, symbolic, minimal shapes.",
        "2d": "Clean vector, crisp edges, simple shading if any.",
        "3d": "Depth, lighting, subtle reflections, dynamic but readable.",
    }[logo_style_dimension]

    color_rule = {
        "pastel": "Soft tones, low saturation.",
        "neutral": "Grayscale / beige, restrained contrast.",
        "bright": "Vivid colors, high clarity, controlled contrast.",
    }[color_palette]

    reference_rule = (
        f"Reference image: {reference_image}. Use for inspiration only; do not copy.\n"
        if reference_image
        else ""
    )

    brand_rule = (
        (brand_name.strip() if brand_name and brand_name.strip() else "None")
        if logo_composition != "icon_only"
        else "None (icon only; no text)"
    )

    rendered = _PROMPT_TEMPLATE.format(
        brand_name=brand_name or "",
        logo_prompt=_compact_ws(logo_prompt),
        logo_composition=logo_composition,
        logo_style_dimension=logo_style_dimension,
        reference_image=reference_image or "",
        color_palette=color_palette,
        composition_rule=composition_rule,
        style_rule=style_rule,
        color_rule=color_rule,
        reference_rule=reference_rule,
        brand_rule=brand_rule,
    )

    rendered = _compact_ws(rendered)
    if len(rendered) <= max_chars:
        return rendered

    return _shrink_to_limit(
        rendered=rendered,
        logo_prompt=_compact_ws(logo_prompt),
        max_chars=max_chars,
    )


def _compact_ws(s: str) -> str:
    return " ".join(s.replace("\r\n", "\n").replace("\n", " ").split())


def _shrink_to_limit(*, rendered: str, logo_prompt: str, max_chars: int) -> str:
    marker = "Brief: "
    idx = rendered.find(marker)
    if idx == -1:
        return rendered[:max_chars]

    prefix = rendered[: idx + len(marker)]
    suffix = rendered[idx + len(marker) :]

    tail = ""
    if " Output:" in suffix:
        brief_and_tail = suffix
        brief_text, tail = brief_and_tail.split(" Output:", 1)
        tail = " Output:" + tail
    else:
        brief_text = suffix

    brief_text = _compact_ws(brief_text)
    tail = _compact_ws(tail)

    budget = max_chars - len(prefix) - len(tail)
    if budget <= 0:
        return (prefix + tail)[:max_chars]

    shortened_brief = logo_prompt[:budget]
    final = _compact_ws(prefix + shortened_brief + tail)
    return final[:max_chars]


