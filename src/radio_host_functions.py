"""
Synthetic Radio Host Generator - Core Functions
"""
__version__ = "2.0.1"

import os
import re
import io
import logging
from functools import reduce
from operator import add
from typing import List

import google.generativeai as genai
from pydub import AudioSegment
import wikipediaapi

logger = logging.getLogger(__name__)

# ======================
# Configuration
# ======================

CONFIG = {
    "WIKIPEDIA_MAX_CHARS": 2500,
    "SCRIPT_TARGET_WORDS": (300, 400),  # Increased for 2+ minutes at ~150 words/min
    "SCRIPT_TARGET_TURNS": (20, 25),  # More turns for longer conversation

    "GEMINI_MODEL": "gemini-3-flash-preview",
    "GEMINI_TTS_MODEL": "gemini-2.5-flash-preview-tts",
    "GEMINI_TEMPERATURE": 0.9,

    "SPEAKER_A_NAME": "Vijay",
    "SPEAKER_B_NAME": "Neha",

    "OUTPUT_FILENAME": "synthetic_radio_host.mp3",
}

# ======================
# Wikipedia
# ======================

def fetch_wikipedia_article(title: str) -> str:
    wiki = wikipediaapi.Wikipedia(
        user_agent="SyntheticRadioHost/2.0.1",
        language="en",
    )

    page = wiki.page(title.strip())
    if not page.exists():
        raise ValueError("Wikipedia page not found")

    text = page.text[:CONFIG["WIKIPEDIA_MAX_CHARS"]]
    if len(text) < 300:
        raise ValueError("Wikipedia article too short")

    return text


# ======================
# Prompt + Script
# ======================

def generate_script_prompt(wiki_text: str) -> str:
    a = CONFIG["SPEAKER_A_NAME"]
    b = CONFIG["SPEAKER_B_NAME"]
    wmin, wmax = CONFIG["SCRIPT_TARGET_WORDS"]
    tmin, tmax = CONFIG["SCRIPT_TARGET_TURNS"]

    return f"""
You are a professional Indian FM radio script writer.

Generate a natural Hinglish conversation
between two radio hosts: {a} and {b}.

MANDATORY:
- Start with: "{a}: Hi hello dosto, main {a} hoon..."
- End with friendly goodbyes from both hosts
- {wmin}-{wmax} words (MUST be at least 300 words for 2+ minute duration)
- {tmin}-{tmax} turns
- Spoken Hinglish only
- The conversation should be engaging and last at least 2 minutes when spoken

FORMAT (STRICT):
{a}: ...
{b}: ...

TOPIC:
{wiki_text}
""".strip()


def generate_script(prompt: str, model: genai.GenerativeModel) -> str:
    system_instruction = "You write natural Hinglish radio conversations."
    full_prompt = f"{system_instruction}\n\n{prompt}"
    
    response = model.generate_content(
        full_prompt,
        generation_config=genai.types.GenerationConfig(
            temperature=CONFIG["GEMINI_TEMPERATURE"],
            max_output_tokens=2000,  # Increased for longer scripts (2+ minutes)
        )
    )

    script = response.text.strip()
    if not script:
        raise RuntimeError("Empty script")

    return script


# ======================
# TTS Helpers
# ======================

def clean_for_tts(line: str) -> str:
    a = CONFIG["SPEAKER_A_NAME"]
    b = CONFIG["SPEAKER_B_NAME"]

    line = re.sub(rf"^({a}|{b}):\s*", "", line)
    line = re.sub(r"\(.*?\)", "", line)
    line = re.sub(r"\s+", " ", line)

    return line.strip()


# ======================
# Audio Generation (SMOOTH)
# ======================

def generate_audio_segments(script: str) -> List[AudioSegment]:
    # Check for Gemini API key
    api_key = os.environ.get("GEMINI_API_KEY")
    if not api_key:
        raise RuntimeError(
            "GEMINI_API_KEY not set in environment. "
            "Please set your Gemini API key as an environment variable."
        )

    # Initialize Gemini TTS model
    tts_model = genai.GenerativeModel(CONFIG["GEMINI_TTS_MODEL"])

    a = CONFIG["SPEAKER_A_NAME"]
    b = CONFIG["SPEAKER_B_NAME"]

    segments: List[AudioSegment] = []

    for line in script.splitlines():
        if not (line.startswith(f"{a}:") or line.startswith(f"{b}:")):
            continue

        text = clean_for_tts(line)
        if not text:
            continue

        try:
            # Generate speech using Gemini TTS
            # TTS models automatically return audio, no need for response_mime_type
            response = tts_model.generate_content(text)
            
            # Get audio content from response
            # Gemini TTS returns audio in response.parts[0].inline_data.data
            audio_content = None
            
            # Check response parts for inline_data (most common for TTS)
            if hasattr(response, 'parts') and response.parts:
                for part in response.parts:
                    # Check for inline_data (standard way TTS models return audio)
                    if hasattr(part, 'inline_data') and part.inline_data:
                        if hasattr(part.inline_data, 'data'):
                            audio_content = part.inline_data.data
                            break
                        elif hasattr(part.inline_data, 'mime_type'):
                            # Sometimes data is in a different attribute
                            logger.debug(f"Found inline_data with mime_type: {part.inline_data.mime_type}")
                    
                    # Fallback: check for direct audio attributes
                    if hasattr(part, 'audio') and part.audio:
                        audio_content = part.audio
                        break
                    elif hasattr(part, 'audio_content'):
                        audio_content = part.audio_content
                        break
            
            # Additional fallback checks at response level
            if not audio_content:
                if hasattr(response, 'audio') and response.audio:
                    audio_content = response.audio
                elif hasattr(response, 'audio_content'):
                    audio_content = response.audio_content
                elif hasattr(response, 'candidates') and response.candidates:
                    # Check candidates if they exist
                    for candidate in response.candidates:
                        if hasattr(candidate, 'content') and hasattr(candidate.content, 'parts'):
                            for part in candidate.content.parts:
                                if hasattr(part, 'inline_data') and part.inline_data:
                                    if hasattr(part.inline_data, 'data'):
                                        audio_content = part.inline_data.data
                                        break
            
            if not audio_content:
                logger.warning(
                    f"No audio content in response for: {text[:50]}...\n"
                    f"Response type: {type(response)}, "
                    f"Has parts: {hasattr(response, 'parts')}, "
                    f"Parts count: {len(response.parts) if hasattr(response, 'parts') else 0}"
                )
                continue

            # Convert audio content to AudioSegment
            # Try MP3 first, then let pydub auto-detect
            try:
                segment = AudioSegment.from_file(
                    io.BytesIO(audio_content),
                    format="mp3"
                )
            except Exception as format_error:
                # If MP3 fails, let pydub auto-detect the format
                try:
                    segment = AudioSegment.from_file(io.BytesIO(audio_content))
                except Exception as auto_error:
                    logger.error(
                        f"Failed to parse audio format. MP3 error: {format_error}, "
                        f"Auto-detect error: {auto_error}"
                    )
                    continue

            segments.append(segment + AudioSegment.silent(220))
            logger.debug(f"Generated audio segment: {len(segment)}ms for text: {text[:30]}...")

        except Exception as e:
            logger.error(f"Failed to generate audio for line: {text[:50]}... Error: {e}")
            import traceback
            logger.debug(traceback.format_exc())
            continue

    if not segments:
        raise RuntimeError("No audio generated")

    return segments


# ======================
# Export
# ======================

def combine_and_export_audio(audio_segments: List[AudioSegment], output_file: str) -> None:
    final_audio = reduce(add, audio_segments)
    final_audio = final_audio.normalize(headroom=1.5)

    final_audio.export(output_file, format="mp3", bitrate="192k")

    logger.info(
        f"Exported {output_file} | "
        f"{len(final_audio) / 1000:.1f}s | "
        f"{os.path.getsize(output_file) / (1024 * 1024):.2f} MB"
    )