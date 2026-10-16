"""Write the two pages of the playground from one body (play/page_body.html), one stylesheet (play/page.css) and one script (static/playground.js):
the local page play/static/index.html (served by the robot at /) and the site page ayushdeshmukh/projects/robot-playground/index.html.
    python3 play/make_pages.py            (run from the repo root)"""
import re
import shutil
from pathlib import Path

HERE = Path(__file__).parent
SITE = HERE.parent.parent / "ayushdeshmukh" / "projects" / "robot-playground"
body, css = (HERE / "page_body.html").read_text(), (HERE / "page.css").read_text()

plain = f"""<!doctype html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1">
<meta name="robots" content="noindex, nofollow, noarchive"><title>Robot Playground</title>
<style>
:root{{--bg:#f6f1e7;--fg:#211d19;--muted:#726657;--accent-warm:#d9480f;--accent-hi:#e8bd2a;--card:#fffdf8;--shadow:rgba(33,29,25,.9);--accent-bold:#1e4fa3}}
@media(prefers-color-scheme:dark){{:root{{--bg:#18140f;--fg:#efe6d6;--muted:#a89b89;--accent-warm:#ff8a50;--accent-hi:#f2cf4e;--card:#241e17;--shadow:rgba(0,0,0,.9);--accent-bold:#8fb6ff}}}}
body{{margin:0;background:var(--bg);color:var(--fg);font:15px/1.5 system-ui,sans-serif}}main{{max-width:1040px;margin:0 auto;padding:20px 16px 40px}}h1{{margin:0 0 4px}}
{css}
</style></head><body><main><h1>Robot Playground</h1>
{body}
</main><script src="/playground.js"></script></body></html>
"""
(HERE / "static" / "index.html").write_text(plain)

page = (SITE / "index.html").read_text()
page = re.sub(r"/\* PLAYGROUND-CSS \*/.*?/\* END-PLAYGROUND-CSS \*/", "/* PLAYGROUND-CSS */\n" + css + "/* END-PLAYGROUND-CSS */", page, flags=re.S)
page = re.sub(r"<!-- PLAYGROUND-BODY -->.*?<!-- END-PLAYGROUND-BODY -->", "<!-- PLAYGROUND-BODY -->\n" + body + "<!-- END-PLAYGROUND-BODY -->", page, flags=re.S)
page = re.sub(r'<script src="playground.js"></script>', '<script src="playground.js"></script>', page)
(SITE / "index.html").write_text(page)
shutil.copy(HERE / "static" / "playground.js", SITE / "playground.js")
print("pages written")
