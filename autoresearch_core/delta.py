"""
autoresearch_core.delta — Op-based Delta for evolution engines.

An Op is a generic action (edit file, create file, append text).
A Delta is a list of Ops that are applied atomically — all succeed or all fail.
New Op kinds can be added by extending Op.execute() without touching the loop.
"""

import difflib
import time
import uuid
from dataclasses import dataclass, field


# ---------------------------------------------------------------------------
# EditOp (search-replace pair, used inside edit_file Ops)
# ---------------------------------------------------------------------------

@dataclass
class EditOp:
    """A single search-replace edit within a file."""
    search: str
    replace: str
    description: str = ""

    def validate(self) -> str | None:
        """Return error message if invalid, None if valid."""
        if not self.search or not self.search.strip():
            return "search string is empty"
        if not self.replace.strip():
            return "replace string is empty (would delete content)"
        if self.search == self.replace:
            return "search and replace are identical (no-op)"
        return None


# ---------------------------------------------------------------------------
# Op — the universal action
# ---------------------------------------------------------------------------

@dataclass
class Op:
    """A generic action the LLM can take.

    Supported kinds:
        edit_file   — search-replace edits on an existing file
        create_file — create a new file with given content
        append_file — append text to an existing file (e.g., reverse links)
        delete_file — remove a file from the vault
        rename_file — move a file to a new path (content preserved)

    Extensible: add new kinds by extending execute(). No loop changes needed.
    """
    kind: str              # "edit_file", "create_file", "append_file", "delete_file", "rename_file"
    path: str = ""         # target path relative to repo root
    edits: list[EditOp] = field(default_factory=list)    # for edit_file
    content: str = ""      # for create_file
    text: str = ""         # for append_file
    # delete_file uses only path — no other fields needed
    new_path: str = ""     # for rename_file (destination path)

    def execute(self, current_content: str | None = None) -> tuple[str | None, str | None]:
        """Execute this op. Returns (result_content, error). Dispatches by kind.

        For delete_file, returns (None, None) — the None content signals deletion.
        For rename_file, returns (content, None) — content is copied to new_path;
            the caller must also delete the old path from file_contents.
        """

        if self.kind == "rename_file":
            if current_content is None:
                return None, f"Cannot rename non-existent file: {self.path}"
            if not self.new_path.strip():
                return None, "rename_file: new_path is empty"
            # Return content unchanged — caller handles the move
            return current_content, None

        if self.kind == "delete_file":
            if current_content is None:
                return None, f"Cannot delete non-existent file: {self.path}"
            return None, None  # sentinel: this file is deleted

        if self.kind == "create_file":
            if not self.content.strip():
                return "", "create_file with empty content"
            return self.content, None

        if self.kind == "append_file":
            if current_content is None:
                return "", f"Cannot append to non-existent file: {self.path}"
            if not self.text.strip():
                return current_content, None  # no-op append is harmless
            return current_content.rstrip('\n') + '\n' + self.text.strip() + '\n', None

        if self.kind == "edit_file":
            if current_content is None:
                return "", f"Cannot edit non-existent file: {self.path}"
            result = current_content
            for i, edit in enumerate(self.edits):
                err = edit.validate()
                if err:
                    return current_content, f"Edit {i} on {self.path}: {err}"
                if edit.search not in result:
                    return current_content, (
                        f"Edit {i} on {self.path}: search text not found. "
                        f"First 80 chars: {edit.search[:80]!r}"
                    )
                count = result.count(edit.search)
                if count > 1:
                    return current_content, (
                        f"Edit {i} on {self.path}: search text matches {count} "
                        f"locations (must be unique)"
                    )
                result = result.replace(edit.search, edit.replace, 1)
            return result, None

        return "", f"Unknown op kind: {self.kind}"

    def to_dict(self) -> dict:
        """Serialize for logging."""
        d = {"kind": self.kind, "path": self.path}
        if self.kind == "edit_file":
            d["num_edits"] = len(self.edits)
        elif self.kind == "create_file":
            d["content_length"] = len(self.content)
        elif self.kind == "append_file":
            d["text_length"] = len(self.text)
        elif self.kind == "rename_file":
            d["new_path"] = self.new_path
        # delete_file has no extra fields
        return d


# ---------------------------------------------------------------------------
# Delta — a bundle of Ops
# ---------------------------------------------------------------------------

@dataclass
class Delta:
    """A proposed change — a list of Ops across any number of files.

    The loop is strategy-agnostic: it calls execute_all(), gates the results,
    and writes atomically. It never needs to know whether this is a densify,
    split, dedup, or cross-link — just ops.
    """
    id: str = field(default_factory=lambda: uuid.uuid4().hex[:12])
    generation: int = 0
    strategy: str = ""
    intent: str = ""
    ops: list[Op] = field(default_factory=list)

    # Post-ranking metadata
    rank: int = 0
    advantage: float = 0.0
    adopted: bool = False
    vetoed: bool = False
    veto_reason: str = ""
    judge_rankings: dict = field(default_factory=dict)
    timestamp: float = field(default_factory=time.time)

    def primary_target(self) -> str:
        """The main file being modified (first op's path)."""
        return self.ops[0].path if self.ops else ""

    def affected_paths(self) -> list[str]:
        """All unique file paths this delta touches, in order."""
        paths = []
        for op in self.ops:
            if op.path:
                paths.append(op.path)
            if op.kind == "rename_file" and op.new_path:
                paths.append(op.new_path)
        return list(dict.fromkeys(paths))

    def execute_all(self, file_contents: dict[str, str]) -> tuple[dict[str, str | None], str | None]:
        """Execute all ops atomically. Returns (new_contents, error).

        On error, returns original contents unchanged (all-or-nothing).
        Values of None in the result indicate deleted files.
        """
        results: dict[str, str | None] = dict(file_contents)
        for op in self.ops:
            current = results.get(op.path)
            # A deleted file (None) can be overwritten by create_file
            if current is None and op.kind not in ("create_file", "delete_file", "rename_file"):
                return file_contents, f"Cannot {op.kind} on deleted file: {op.path}"
            new_content, err = op.execute(current)
            if err:
                return file_contents, err
            if op.kind == "rename_file":
                # Copy content to new_path, delete old path
                results[op.new_path] = new_content
                results[op.path] = None  # mark old path as deleted
            else:
                results[op.path] = new_content
        return results, None

    def render_for_ranking(self, file_contents: dict[str, str]) -> str:
        """Render combined diff across all affected files for GRPO judges."""
        new_contents, err = self.execute_all(file_contents)
        if err:
            return f"(failed to apply: {err})"

        parts = []
        for path in self.affected_paths():
            old = file_contents.get(path, "")
            new = new_contents.get(path)
            if new is None:
                # File deleted
                parts.append(f"--- DELETED: {path} ---\n(file removed from vault)")
                continue
            if old != new:
                diff_lines = list(difflib.unified_diff(
                    old.splitlines(keepends=True),
                    new.splitlines(keepends=True),
                    fromfile=f"current/{path}",
                    tofile=f"proposed/{path}",
                    n=3,
                ))
                if diff_lines:
                    parts.append("".join(diff_lines))

        return "\n\n".join(parts) if parts else "(no changes)"

    def to_log_entry(self) -> dict:
        """Serialize for history logging."""
        return {
            "id": self.id,
            "generation": self.generation,
            "strategy": self.strategy,
            "intent": self.intent,
            "ops": [op.to_dict() for op in self.ops],
            "affected_paths": (paths := self.affected_paths()),
            "num_files": len(paths),
            "rank": self.rank,
            "advantage": self.advantage,
            "adopted": self.adopted,
            "vetoed": self.vetoed,
            "veto_reason": self.veto_reason,
            "timestamp": self.timestamp,
        }
