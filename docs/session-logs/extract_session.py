#!/usr/bin/env python3
"""Extract a readable, text-only log from a Claude Code session .jsonl.

Skips image blobs and truncates long tool output so the result is something a
human (or docs-agent) can actually read.
"""
import json
import sys

SRC = sys.argv[1]
OUT = sys.argv[2]
TOOL_OUT_LIMIT = 1200


def text_of(content):
    """Flatten a message's content blocks into (text, tool_notes)."""
    if isinstance(content, str):
        return content, []
    parts, notes = [], []
    for block in content if isinstance(content, list) else []:
        if not isinstance(block, dict):
            continue
        btype = block.get("type")
        if btype == "text":
            parts.append(block.get("text", ""))
        elif btype == "thinking":
            pass  # skip reasoning
        elif btype == "image":
            notes.append("[image omitted]")
        elif btype == "tool_use":
            name = block.get("name", "?")
            inp = block.get("input", {})
            summary = ""
            for key in ("command", "file_path", "pattern", "prompt", "description"):
                if key in inp:
                    summary = str(inp[key])[:300]
                    break
            notes.append(f"[tool: {name}] {summary}")
        elif btype == "tool_result":
            body = block.get("content")
            if isinstance(body, list):
                chunks = []
                for sub in body:
                    if isinstance(sub, dict) and sub.get("type") == "text":
                        chunks.append(sub.get("text", ""))
                    elif isinstance(sub, dict) and sub.get("type") == "image":
                        chunks.append("[image omitted]")
                body = "\n".join(chunks)
            body = str(body or "")
            if len(body) > TOOL_OUT_LIMIT:
                body = body[:TOOL_OUT_LIMIT] + f"\n... [truncated, {len(body)} chars total]"
            notes.append(f"[tool result]\n{body}")
    return "\n".join(p for p in parts if p.strip()), notes


def main():
    lines = []
    with open(SRC, encoding="utf-8") as fh:
        for raw in fh:
            raw = raw.strip()
            if not raw:
                continue
            try:
                rec = json.loads(raw)
            except json.JSONDecodeError:
                continue

            msg = rec.get("message")
            if not isinstance(msg, dict):
                continue
            role = msg.get("role")
            if role not in ("user", "assistant"):
                continue

            body, notes = text_of(msg.get("content"))
            if not body and not notes:
                continue

            ts = rec.get("timestamp", "")[:19]
            if body.strip():
                lines.append(f"\n### {role.upper()}  ({ts})\n")
                lines.append(body.strip())
            for note in notes:
                lines.append(f"\n<{role} {ts}> {note}")

    with open(OUT, "w", encoding="utf-8") as fh:
        fh.write("# Session transcript — GuardianEye Phase 4 risk engine\n")
        fh.write(f"\nExtracted from `{SRC}` (text only; images omitted, tool output truncated).\n")
        fh.write("\n".join(lines))
    print(f"wrote {OUT}")


if __name__ == "__main__":
    main()
