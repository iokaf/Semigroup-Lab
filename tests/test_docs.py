"""docs.md must only reference functions and sections that exist."""
import re
from pathlib import Path

from engine import sourcemap

ROOT = Path(__file__).resolve().parents[1]
DOCS = (ROOT / "docs.md").read_text(encoding="utf-8")


def test_code_references_resolve():
    keys = re.findall(r"<!-- code: ([A-Za-z_]+) -->", DOCS)
    assert len(keys) >= 30
    for k in keys:
        info = sourcemap.resolve(k)
        assert info["source"].lstrip().startswith(("def ", "class ")), k


def test_internal_links_point_to_sections():
    sections = set(re.findall(r"<!-- section: ([a-z0-9-]+) -->", DOCS))
    assert len(sections) == 16
    assert set(re.findall(r"\]\(#([a-z0-9-]+)\)", DOCS)) <= sections
    analyse = (ROOT / "views" / "analyse.py").read_text(encoding="utf-8")
    used = set(re.findall(r"how\(\"([a-z0-9-]+)\"", analyse)) | set(re.findall(r"how\('([a-z0-9-]+)'", analyse))
    assert used and used <= sections, used - sections


def test_every_section_has_a_header_line():
    lines = DOCS.splitlines()
    for i, line in enumerate(lines):
        if line.startswith("<!-- section:"):
            nxt = next(l for l in lines[i + 1:] if l.strip())
            assert nxt.startswith("## "), f"section marker at line {i + 1} is not followed by a '## ' header"
