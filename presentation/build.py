"""Build deck.html from deck.src.html by inlining the screenshots as data URIs.

The published deck runs under a CSP that blocks every external host, so the images have to
live in the file itself. Edit deck.src.html (which keeps {{IMG:name|caption}} placeholders
so it stays readable) and re-run:

    python presentation/build.py
"""
import base64
import re
from pathlib import Path

HERE = Path(__file__).resolve().parent
SRC = HERE / "deck.src.html"
OUT = HERE / "deck.html"
SHOTS = HERE / "screenshots"

used = []


def repl(m):
    name, caption = m.group(1), m.group(2)
    png = SHOTS / f"{name}.png"
    if not png.exists():
        raise SystemExit(f"missing screenshot: {png}")
    data = base64.b64encode(png.read_bytes()).decode()
    used.append((name, len(data)))
    return (f'<figure><img src="data:image/png;base64,{data}" alt="{caption}">'
            f'<figcaption>{caption}</figcaption></figure>')


html = re.sub(r"\{\{IMG:([a-z0-9-]+)\|([^}]+)\}\}", repl, SRC.read_text(encoding="utf-8"))
if "{{IMG" in html:
    raise SystemExit("unreplaced image placeholder — check the {{IMG:name|caption}} syntax")

OUT.write_text(html, encoding="utf-8")
for name, size in used:
    print(f"  {name:24} {size // 1024:>5}KB base64")
print(f"\n{OUT.name}: {OUT.stat().st_size / 1024 / 1024:.2f}MB, {len(used)} images")
