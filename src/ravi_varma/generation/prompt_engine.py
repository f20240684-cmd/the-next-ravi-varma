"""Prompt expansion: turns a plain-language scene description into a
generation-ready, structured prompt, injecting the Ravi Varma style trigger
token without altering the user's actual story.

Two modes:
  * "local"  -- deterministic templates, no network/API required. This is
    the default and the guaranteed-available fallback.
  * "llm"    -- an optional external LLM (OpenAI or Anthropic) fleshes out
    visual details (lighting/pose/composition) around the user's scene.
    Requires an API key in `.env`; if unavailable, unset, or the call fails
    for any reason, automatically falls back to "local" mode and logs why.

In both modes the user's subject/story text is preserved verbatim in the
final prompt -- the engine only *adds* descriptive/style tokens, never
rewrites or replaces what the user asked for.
"""
from __future__ import annotations

import json
import os
import re
from dataclasses import asdict, dataclass, field
from typing import Optional

from ravi_varma.utils.logging import get_logger

logger = get_logger(__name__)

DEFAULT_NEGATIVE_PROMPT = (
    "malformed anatomy, extra fingers, missing fingers, fused fingers, "
    "duplicated people, distorted faces, disfigured, mutated, text, "
    "watermark, signature, logo, low quality, worst quality, blurry image, "
    "out of focus, deformed hands, unnatural limbs, cropped subjects, "
    "extra limbs, bad proportions, lowres"
)

# Deterministic, hand-authored visual vocabulary reflecting Ravi Varma's
# documented stylistic tendencies (academic realism, oil-painting technique,
# theatrical/mythological staging, soft directional lighting). These are
# generic style descriptors, not claims about any specific painting.
_STYLE_LIGHTING = "soft directional chiaroscuro lighting, warm golden highlights, painterly shadow modeling"
_STYLE_COMPOSITION = "classical academic composition, balanced framing, theatrical staging reminiscent of 19th-century Indian oleographs"
_STYLE_CLOTHING = "traditional Indian attire rendered with fine textile detail, ornate jewelry"
_STYLE_VISUAL = "oil-painting texture, rich color palette, realistic anatomy, fine brushwork"


@dataclass
class ExpandedPrompt:
    subject: str
    characters: str = ""
    pose: str = "unspecified, left to the model's default interpretation"
    environment: str = "unspecified, left to the model's default interpretation"
    lighting: str = _STYLE_LIGHTING
    composition: str = _STYLE_COMPOSITION
    clothing: str = _STYLE_CLOTHING
    style: str = _STYLE_VISUAL
    negative_prompt: str = DEFAULT_NEGATIVE_PROMPT
    trigger_token: Optional[str] = None
    mode_used: str = "local"

    def to_dict(self) -> dict:
        return asdict(self)

    def final_prompt(self) -> str:
        """Assemble the final diffusion prompt. The user's `subject` text is
        placed first and verbatim so their intent always dominates."""
        parts = []
        if self.trigger_token:
            parts.append(self.trigger_token)
        parts.append(self.subject.strip())
        for extra in (self.clothing, self.pose, self.environment, self.lighting, self.composition, self.style):
            if extra and "unspecified" not in extra:
                parts.append(extra)
        return ", ".join(p for p in parts if p)


# Lightweight keyword-based extraction so "local" mode can still populate
# characters/environment more specifically without an LLM, while never
# inventing facts the user didn't state.
_ENV_KEYWORDS = {
    "battlefield": "an epic battlefield setting",
    "palace": "a palatial interior",
    "forest": "a dense forest setting",
    "river": "a riverside setting",
    "temple": "a temple setting",
    "court": "a royal court setting",
    "garden": "a palace garden",
    "sea": "a coastal/seaside setting",
    "mountain": "a mountainous landscape",
    "throne": "a throne room",
}


def _extract_environment(text: str) -> str:
    lowered = text.lower()
    for kw, phrase in _ENV_KEYWORDS.items():
        if kw in lowered:
            return phrase
    return "unspecified, left to the model's default interpretation"


def _extract_characters(text: str) -> str:
    # Very conservative: pull out capitalized words that look like proper
    # names (e.g. "Arjuna", "Krishna") without asserting anything about them
    # beyond "characters mentioned in the scene".
    names = re.findall(r"\b[A-Z][a-z]+\b", text)
    # Filter out the first word of the sentence if it's just capitalized
    # because it starts the sentence and isn't repeated elsewhere -- keep it
    # simple and inclusive rather than risk dropping a real name.
    unique_names = list(dict.fromkeys(names))
    return ", ".join(unique_names) if unique_names else ""


class LocalPromptEngine:
    """Deterministic, offline prompt expansion. Always available."""

    def expand(self, scene_description: str, trigger_token: Optional[str] = None, negative_prompt: Optional[str] = None) -> ExpandedPrompt:
        scene_description = scene_description.strip()
        if not scene_description:
            raise ValueError("scene_description must not be empty.")

        return ExpandedPrompt(
            subject=scene_description,
            characters=_extract_characters(scene_description),
            environment=_extract_environment(scene_description),
            negative_prompt=negative_prompt or DEFAULT_NEGATIVE_PROMPT,
            trigger_token=trigger_token,
            mode_used="local",
        )


class LLMPromptEngine:
    """Optional LLM-backed prompt expansion. Supports OpenAI or Anthropic,
    selected via `PROMPT_EXPANSION_PROVIDER` in `.env`. Never invents plot
    details -- the system prompt explicitly instructs the LLM to preserve
    the user's story and only add *visual* descriptors."""

    SYSTEM_PROMPT = (
        "You expand short scene descriptions into structured visual prompts for an "
        "image generation model. You must NEVER change the user's story, characters, "
        "actions, or intent. Only ADD visual descriptors (pose, environment, lighting, "
        "composition, clothing) that are consistent with what the user wrote. "
        "Respond with ONLY a JSON object with these exact keys: "
        "subject, characters, pose, environment, lighting, composition, clothing, style, negative_prompt. "
        "The 'subject' value must contain the user's original scene text, verbatim, at minimum."
    )

    def __init__(self, provider: str, model: Optional[str] = None):
        self.provider = provider
        self.model = model

    def _call_openai(self, scene_description: str) -> dict:
        from openai import OpenAI

        client = OpenAI(api_key=os.environ["OPENAI_API_KEY"])
        response = client.chat.completions.create(
            model=self.model or "gpt-4o-mini",
            messages=[
                {"role": "system", "content": self.SYSTEM_PROMPT},
                {"role": "user", "content": scene_description},
            ],
            response_format={"type": "json_object"},
            temperature=0.4,
        )
        return json.loads(response.choices[0].message.content)

    def _call_anthropic(self, scene_description: str) -> dict:
        import anthropic

        client = anthropic.Anthropic(api_key=os.environ["ANTHROPIC_API_KEY"])
        response = client.messages.create(
            model=self.model or "claude-haiku-4-5-20251001",
            max_tokens=600,
            system=self.SYSTEM_PROMPT,
            messages=[{"role": "user", "content": scene_description}],
        )
        text = "".join(block.text for block in response.content if block.type == "text")
        # Defensive: strip markdown code fences if the model added them anyway.
        text = re.sub(r"^```json|```$", "", text.strip(), flags=re.MULTILINE).strip()
        return json.loads(text)

    def expand(self, scene_description: str, trigger_token: Optional[str] = None, negative_prompt: Optional[str] = None) -> ExpandedPrompt:
        if self.provider == "openai" and os.environ.get("OPENAI_API_KEY"):
            data = self._call_openai(scene_description)
        elif self.provider == "anthropic" and os.environ.get("ANTHROPIC_API_KEY"):
            data = self._call_anthropic(scene_description)
        else:
            raise RuntimeError(f"No valid API key configured for provider '{self.provider}'.")

        if scene_description.strip() not in data.get("subject", ""):
            # The model failed to preserve the user's story -- refuse the
            # LLM output rather than silently rewriting their scene.
            raise ValueError("LLM output did not preserve the original scene text; discarding.")

        return ExpandedPrompt(
            subject=data.get("subject", scene_description),
            characters=data.get("characters", ""),
            pose=data.get("pose", ExpandedPrompt.__dataclass_fields__["pose"].default),
            environment=data.get("environment", ExpandedPrompt.__dataclass_fields__["environment"].default),
            lighting=data.get("lighting", _STYLE_LIGHTING),
            composition=data.get("composition", _STYLE_COMPOSITION),
            clothing=data.get("clothing", _STYLE_CLOTHING),
            style=data.get("style", _STYLE_VISUAL),
            negative_prompt=data.get("negative_prompt") or negative_prompt or DEFAULT_NEGATIVE_PROMPT,
            trigger_token=trigger_token,
            mode_used="llm",
        )


class PromptExpansionEngine:
    """Public entry point used by scripts/generate.py and the Gradio app.
    Tries the configured LLM provider (if any), and transparently falls
    back to the local deterministic engine on any failure."""

    def __init__(self, trigger_token: str = "<rvvarma>", negative_prompt: Optional[str] = None):
        self.trigger_token = trigger_token
        self.negative_prompt = negative_prompt
        self.local_engine = LocalPromptEngine()

        provider = os.environ.get("PROMPT_EXPANSION_PROVIDER", "none").lower()
        self.llm_engine: Optional[LLMPromptEngine] = (
            LLMPromptEngine(provider, os.environ.get("PROMPT_EXPANSION_MODEL"))
            if provider in ("openai", "anthropic")
            else None
        )

    def expand(self, scene_description: str, enable_style: bool = True, force_local: bool = False) -> ExpandedPrompt:
        trigger = self.trigger_token if enable_style else None

        if not force_local and self.llm_engine is not None:
            try:
                return self.llm_engine.expand(scene_description, trigger, self.negative_prompt)
            except Exception as e:
                logger.warning("LLM prompt expansion failed (%s); falling back to local engine.", e)

        return self.local_engine.expand(scene_description, trigger, self.negative_prompt)
