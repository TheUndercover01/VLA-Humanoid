"""16 Oct: the project page is the README, word for word, in the site's style, with videos instead of GIFs.
    python3 scripts/readme_to_page.py [--readme README.md] [--page ../ayushdeshmukh/projects/never-shown/index.html]
Keeps the page's <head> (fonts, site CSS, its local styles) and the site bar; everything inside <main> is rebuilt from the README.
Every results/site_media/<name>.gif becomes videos/<map>.mp4 (+ .jpg poster) on the page; phone_clips.gif becomes the four clip videos."""
import argparse
import re
from pathlib import Path

import markdown

VID = {"real_clip_to_template": "clip_to_template"}            # gif name -> mp4 name on the page, where they differ
PLAY = "https://theundercover01.github.io/ayushdeshmukh/projects/play/"
EXTRA_CSS = """
  /* 16 Oct: README-generated page */
  .paper-sub i{display:block;margin-top:.35rem}
  .paper video.v{display:inline-block;vertical-align:top;background:#000;border:3px solid var(--fg);border-radius:4px;box-shadow:4px 4px 0 var(--shadow);margin:.25rem .3%}
  .paper .center{text-align:center;margin:0 0 .6rem}
  .paper .center sub{display:block;font-size:.82rem;line-height:1.45;color:var(--muted);max-width:46rem;margin:0 auto}
  .paper .center sub b{color:var(--fg)}
  .paper .tbl{overflow-x:auto;margin:0 0 1.4rem}
  .paper .tbl table{border-collapse:collapse;width:100%;font-size:.86rem;background:var(--card);border:3px solid var(--fg);box-shadow:5px 5px 0 var(--shadow)}
  .paper .tbl th,.paper .tbl td{padding:.45rem .6rem;border-bottom:1px solid color-mix(in oklch, var(--fg) 15%, transparent);vertical-align:top}
  .paper .tbl th{font:700 .8rem "Space Grotesk",sans-serif;text-align:left;border-bottom:2px solid var(--fg)}
  .paper .tbl sub{font-size:.78rem;color:var(--muted)}
  .paper table.gallery{width:100%;border-collapse:separate;border-spacing:.6rem;margin:0 -.6rem 1rem}
  .paper table.gallery td{vertical-align:top;font-size:.8rem;line-height:1.4;color:var(--muted)}
  .paper table.gallery video.v{width:100%;margin:0 0 .3rem}
  .paper blockquote{margin:0 0 1rem;padding:.6rem .9rem;border-left:5px solid var(--accent-warm);background:var(--card);font:400 .85rem/1.7 "JetBrains Mono",monospace}
  .paper pre{overflow-x:auto;background:var(--card);border:2px solid var(--fg);border-radius:4px;padding:.7rem .9rem;font-size:.8rem}
  .paper h3{font:700 1.15rem/1.3 "Space Grotesk",sans-serif;margin:1.6rem 0 .5rem}
  .paper ul,.paper ol{line-height:1.6}
  .paper .bigplay{text-align:center;margin:1.4rem 0 2rem}
  .paper .bigplay a{display:inline-block;text-decoration:none;font:800 1.15rem "Space Grotesk",sans-serif;color:var(--fg);background:var(--card);
    border:3px solid var(--fg);border-radius:6px;box-shadow:5px 5px 0 var(--shadow);padding:.75rem 1.6rem}
  .paper .bigplay a:hover{background:var(--accent-hi);color:#211d19}
  .paper .clips{margin:0 0 .4rem}
"""


def slug(text):
    """GitHub's heading anchors: lower case, drop punctuation except - and spaces, spaces -> -."""
    t = re.sub(r"<[^>]+>", "", text).strip().lower()
    t = re.sub(r"[^\w\- ]", "", t)
    return t.replace(" ", "-")


def video(name, width):
    name = VID.get(name, name)
    return (f'<video class="v" src="videos/{name}.mp4" poster="videos/{name}.jpg" style="width:{width}" '
            'data-autoplay autoplay muted loop controls playsinline preload="metadata"></video>')


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--readme", default="README.md")
    ap.add_argument("--page", default="../ayushdeshmukh/projects/never-shown/index.html")
    a = ap.parse_args()
    md = Path(a.readme).read_text()
    page = Path(a.page).read_text()

    # title block: <h1>, subtitle, question -> the page header
    title = re.search(r"<h1[^>]*>(.*?)</h1>", md).group(1)
    sub = re.search(r'<p align="center"><b>(.*?)</b></p>', md).group(1)
    q = re.search(r'<p align="center"><i>(.*?)</i></p>', md).group(1)
    body = md[md.index("<p align=\"center\">\n  <img"):]          # from the hero video on

    # the two README buttons -> one big "play" button (the page itself is the project page)
    body = re.sub(r'<p align="center"><b>Click the buttons below.*?</p>\s*<p align="center">.*?</p>\s*<p align="center">.*?</p>',
                  f'<div class="bigplay"><a href="{PLAY}">▶&nbsp; Play with the robot live</a></div>', body, flags=re.S)

    body = re.sub(r"~~(.+?)~~", r"<del>\1</del>", body)
    html = markdown.markdown(body, extensions=["tables", "fenced_code", "sane_lists"])

    # GIFs -> videos
    html = re.sub(r'<img src="results/site_media/phone_clips\.gif"[^>]*>',
                  '<div class="clips">' + "".join(
                      f'<figure>{video(n, "100%")}</figure>'
                      for n, c in [("clip_push", "Push"), ("clip_lift", "Pick up"), ("clip_drop", "Put down"),
                                   ("clip_stack", "Put on the other object")]) + "</div>", html)
    html = re.sub(r'<img src="results/site_media/([\w]+)\.gif"[^>]*?width="([\d%]+)"[^>]*>',
                  lambda m: video(m.group(1), m.group(2)), html)
    html = html.replace('src="docs/pipeline.svg"', 'src="img/pipeline.svg"')
    html = re.sub(r'<p align="center">', '<div class="center">', html)
    html = re.sub(r'(<div class="center">.*?)</p>', r"\1</div>", html, flags=re.S)
    html = html.replace("<table>\n<tr><td", '<table class="gallery">\n<tr><td')

    # markdown tables -> styled; headings -> anchors like GitHub's; plain paragraphs -> the site's text class
    html = re.sub(r"<table>\s*<thead>", '<div class="tbl"><table>\n<thead>', html)
    html = re.sub(r"(<div class=\"tbl\"><table>.*?</table>)", r"\1</div>", html, flags=re.S)
    html = re.sub(r"<h2>(.*?)</h2>", lambda m: f'<h2 class="paper-h" id="{slug(m.group(1))}">{m.group(1)}</h2>', html)
    html = re.sub(r"<h3>(.*?)</h3>", lambda m: f'<h3 id="{slug(m.group(1))}">{m.group(1)}</h3>', html)
    html = html.replace("<p>", '<p class="paper-text">')
    parts = re.split(r'(?=<h2 class="paper-h")', html)
    html = parts[0] + "".join(f"<section>\n{p}\n</section>\n" for p in parts[1:])

    head = page[:page.index("</style>")]
    head = head.split("  /* 16 Oct: README-generated page */")[0] + EXTRA_CSS + "</style>"
    rest = page[page.index("</style>") + len("</style>"):]
    bar = rest[:rest.index('<main class="paper">')]
    foot = rest[rest.index('  <footer class="site-footer paper-foot">'):]
    header = (f'<main class="paper">\n  <header class="paper-head">\n    <h1 class="paper-title">{title}</h1>\n'
              f'    <p class="paper-sub">{sub}<i>{q}</i></p>\n    <p class="paper-authors"><b>Ayush Deshmukh</b></p>\n  </header>\n\n')
    Path(a.page).write_text(head + bar + header + html + "\n" + foot)
    print(f"wrote {a.page}: {len(re.findall('<video', html))} videos, {len(re.findall('<h2', html))} sections")


if __name__ == "__main__":
    main()
