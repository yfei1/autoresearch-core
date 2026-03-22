"""
autoresearch_core.util — LLM output parsing and overlap detection utilities.

Generic helpers for extracting structured data from LLM outputs and detecting
content overlaps across files.
"""

import json
import re
from dataclasses import dataclass


# ---------------------------------------------------------------------------
# LLM output parsing
# ---------------------------------------------------------------------------

def strip_markdown_fences(text: str) -> str:
    """Remove markdown code fences wrapping LLM output."""
    stripped = text.strip()
    if stripped.startswith("```"):
        lines = stripped.split("\n")
        lines = [l for l in lines if not l.strip().startswith("```")]
        return "\n".join(lines).strip()
    return stripped


def extract_json_object(output: str) -> dict | None:
    """Extract the outermost JSON object from LLM output.

    Handles markdown fences and surrounding text.
    """
    cleaned = strip_markdown_fences(output)
    first = cleaned.find('{')
    last = cleaned.rfind('}')
    if first == -1 or last == -1 or last <= first:
        return None
    try:
        return json.loads(cleaned[first:last + 1])
    except json.JSONDecodeError:
        return None


def extract_json_array(output: str) -> list | None:
    """Extract the outermost JSON array from LLM output.

    Handles markdown fences and surrounding text.
    """
    cleaned = strip_markdown_fences(output)
    first = cleaned.find('[')
    last = cleaned.rfind(']')
    if first == -1 or last == -1 or last <= first:
        return None
    try:
        result = json.loads(cleaned[first:last + 1])
        return result if isinstance(result, list) else None
    except json.JSONDecodeError:
        return None


# ---------------------------------------------------------------------------
# Overlap detection
# ---------------------------------------------------------------------------

def find_paragraph_overlaps(
    source_content: str,
    target_contents: dict[str, str],
    threshold: float = 0.7,
    min_paragraph_len: int = 100,
) -> list[tuple[str, float, str, str]]:
    """Find paragraph-level word-set overlaps between source and target files.

    Args:
        source_content: content of the file being checked.
        target_contents: {path: content} for files to compare against.
        threshold: minimum word-set overlap ratio to flag (0-1).
        min_paragraph_len: minimum paragraph length in chars.

    Returns:
        List of (target_path, overlap_ratio, source_preview, target_preview).
        At most one overlap per target file.
    """
    paragraphs = re.split(r'\n\s*\n', source_content)
    paragraphs = [p.strip() for p in paragraphs if len(p.strip()) > min_paragraph_len]
    if not paragraphs:
        return []

    overlaps = []
    seen = set()

    for target_path, target_content in target_contents.items():
        if target_path in seen:
            continue
        target_sentences = re.split(r'[.!?]\s+', target_content)

        for para in paragraphs:
            if target_path in seen:
                break
            para_words = set(para.lower().split())
            if len(para_words) < 10:
                continue
            for i in range(len(target_sentences) - 2):
                chunk = ' '.join(target_sentences[i:i + 3])
                chunk_words = set(chunk.lower().split())
                if len(chunk_words) < 10:
                    continue
                ratio = len(para_words & chunk_words) / min(len(para_words), len(chunk_words))
                if ratio > threshold:
                    overlaps.append((
                        target_path,
                        ratio,
                        para.replace('\n', ' ')[:150],
                        chunk.replace('\n', ' ')[:150],
                    ))
                    seen.add(target_path)
                    break

    return overlaps


@dataclass
class Overlap:
    """A detected content overlap between two files."""
    source_path: str           # file with the duplicate content
    canonical_path: str        # file that should be the canonical home
    source_preview: str        # preview of the overlapping paragraph in source
    canonical_preview: str     # preview of the matching content in canonical
    overlap_ratio: float       # 0-1, how much overlap


def detect_overlaps(
    file_contents: dict[str, str],
    threshold: float = 0.7,
    min_paragraph_len: int = 100,
) -> list[Overlap]:
    """Detect paragraph-level content overlaps across all files.

    Pairwise comparison using find_paragraph_overlaps, returning typed Overlap objects
    sorted by overlap ratio descending.
    """
    overlaps = []
    paths = sorted(file_contents.keys())

    for i, source_path in enumerate(paths):
        target_contents = {p: file_contents[p] for p in paths[i + 1:]}
        raw = find_paragraph_overlaps(
            file_contents[source_path], target_contents,
            threshold=threshold, min_paragraph_len=min_paragraph_len,
        )
        for target_path, ratio, src_preview, tgt_preview in raw:
            overlaps.append(Overlap(
                source_path=source_path,
                canonical_path=target_path,
                source_preview=src_preview,
                canonical_preview=tgt_preview,
                overlap_ratio=ratio,
            ))

    overlaps.sort(key=lambda o: o.overlap_ratio, reverse=True)
    return overlaps
