from __future__ import annotations

from pathlib import Path

from generator import LogoInputs, generate_logo


def main() -> None:
    inputs = LogoInputs(
        brand_name="Chora Ganga Kinare Wala",
        logo_prompt="make a typography only 3d logo for a big indian bollywood movie named Chora Ganga Kinare Wala the film theme is Karan Johar types. Bright colours and instantly attractive. be creative with the words and make it look good.",
        logo_composition="typography_only",
        logo_style_dimension="3d",
        reference_image=None,
        color_palette="bright",
    )

    result = generate_logo(
        inputs,
        project_id="logo-maker-481509",
        location="us-central1",
        model_name="gemini-3-pro-image-preview",
    )

    out_path = Path("jcl_logo.png")
    out_path.write_bytes(result.image_bytes)
    print(f"Saved: {out_path.resolve()}")
    print(f"Prompt chars: {len(result.prompt)}")


if __name__ == "__main__":
    main()


