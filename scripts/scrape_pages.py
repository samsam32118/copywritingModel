#!/usr/bin/env python3
"""Generic landing-page scraper for the copywriting fine-tune pipeline.

Two input modes:

  --urls-json yc_all.json --source yc --out raw_yc.jsonl
      Reads a YC-style JSON list of company records. Only records with
      status in {Active, Public, Acquired} and a non-empty website are
      scraped. YC metadata is attached to each output record.

  --urls-txt urls.txt --source gallery --out raw_gallery.jsonl
      Reads a plain text file of one URL per line. No metadata beyond what
      is scraped from the page itself is attached.

Output: one JSON object per line (JSONL) written to --out, plus a
compact error record per failed URL written to errors_<source>.jsonl in
the same directory as --out.

Resumable: URLs already present (by "url" key) in --out or in the
errors file are skipped on a re-run.
"""

import argparse
import json
import os
import re
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from urllib.parse import urlparse

import requests
from bs4 import BeautifulSoup

try:
    from langdetect import detect, DetectorFactory, LangDetectException
    DetectorFactory.seed = 0
except Exception:  # pragma: no cover - guarded import
    detect = None
    LangDetectException = Exception


# --------------------------------------------------------------------------
# Config / constants
# --------------------------------------------------------------------------

USER_AGENT = (
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
    "(KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36"
)
ACCEPT_LANGUAGE = "en-US,en;q=0.9"
TIMEOUT_S = 12
MAX_BODY_BYTES = int(2.5 * 1024 * 1024)  # 2.5MB cap
DEFAULT_WORKERS = 24
PROGRESS_EVERY = 200
YC_ALLOWED_STATUS = {"Active", "Public", "Acquired"}

STRIP_TAGS = {
    "script", "style", "noscript", "svg", "iframe", "template",
    "input", "select", "textarea",  # form inputs
    "nav", "footer",
}
JUNK_CLASS_ID_RE = re.compile(
    r"(?i)cookie|consent|gdpr|navbar|nav-|menu|footer|breadcrumb|sr-only|visually-hidden"
)
CTA_RE = re.compile(r"(?i)btn|button|cta")
WALK_TAGS = ["h1", "h2", "h3", "h4", "p", "button", "a"]
CODE_START_RE = re.compile(
    r"^(import|from|class|def|func|function|const|let|var|public|private|"
    r"override|package|return|struct|enum|interface|protocol)\b"
)
FUNC_CALL_RE = re.compile(r"^[A-Za-z_][\w.]*\([^\n]*\)$")
BUTTON_JUNK_RE = re.compile(
    r"(?i)^("
    r"log ?in|sign ?in|menu|close|dismiss|"
    r"(close|dismiss) (notification|menu|dialog|modal|banner|popup|announcement|alert|nav|cookie).*|"
    r"accept|decline|ok|got it|×|"
    r"cookie.*|privacy.*|terms.*|english|toggle.*|skip.*|back to top|"
    r"open menu.*|next|previous|play|pause|"
    r"(go ?to) (slide|chapter|step|page)\s*\d*|"
    r"(previous|next) (slide|chapter|step|page)"
    r")$"
)

_domain_locks = {}
_domain_locks_guard = threading.Lock()

_print_lock = threading.Lock()


# --------------------------------------------------------------------------
# Small helpers
# --------------------------------------------------------------------------

def normalize_ws(text):
    return re.sub(r"\s+", " ", text or "").strip()


def get_domain_lock(domain):
    with _domain_locks_guard:
        lock = _domain_locks.get(domain)
        if lock is None:
            lock = threading.Lock()
            _domain_locks[domain] = lock
        return lock


def ensure_scheme(url):
    url = (url or "").strip()
    if not url:
        return url
    if not re.match(r"^https?://", url, re.I):
        url = "https://" + url
    return url


def build_session():
    s = requests.Session()
    s.headers.update({
        "User-Agent": USER_AGENT,
        "Accept-Language": ACCEPT_LANGUAGE,
        "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
    })
    adapter = requests.adapters.HTTPAdapter(pool_connections=64, pool_maxsize=64, max_retries=0)
    s.mount("http://", adapter)
    s.mount("https://", adapter)
    return s


# --------------------------------------------------------------------------
# Fetching
# --------------------------------------------------------------------------

def fetch_once(session, url):
    """Single attempt. Returns (content_bytes, error_str, status, final_url)."""
    resp = None
    try:
        resp = session.get(url, timeout=TIMEOUT_S, stream=True, allow_redirects=True)
        ctype = resp.headers.get("Content-Type", "")
        if ctype and "text/html" not in ctype.lower():
            return None, f"non-html content-type: {ctype}", resp.status_code, resp.url
        content = bytearray()
        for chunk in resp.iter_content(chunk_size=65536):
            if not chunk:
                continue
            content.extend(chunk)
            if len(content) >= MAX_BODY_BYTES:
                break
        status, final_url = resp.status_code, resp.url
        return bytes(content), None, status, final_url
    finally:
        if resp is not None:
            resp.close()


def fetch(session, url):
    """Fetch with one retry on connection-level errors. One in-flight request per domain."""
    domain = urlparse(url).netloc.lower()
    lock = get_domain_lock(domain)
    with lock:
        last_err = None
        for attempt in range(2):
            try:
                return fetch_once(session, url)
            except (requests.exceptions.ConnectionError, requests.exceptions.Timeout) as e:
                last_err = f"{type(e).__name__}: {e}"
                continue  # retry once
            except requests.exceptions.RequestException as e:
                return None, f"{type(e).__name__}: {e}", None, None
            except Exception as e:
                return None, f"unexpected {type(e).__name__}: {e}", None, None
        return None, last_err or "connection error after retry", None, None


# --------------------------------------------------------------------------
# Parsing
# --------------------------------------------------------------------------

def strip_junk(soup):
    for tag in soup.find_all(STRIP_TAGS):
        if tag.decomposed:
            continue
        tag.decompose()
    # Second pass: anything whose id/class looks like chrome/consent noise.
    # find_all(True) is materialized up front, so a tag whose ancestor we
    # already decomposed in this same loop can still appear here -- skip it,
    # since decompose() clears the tag's __dict__ (attrs becomes None).
    for tag in soup.find_all(True):
        if tag.decomposed:
            continue
        id_val = tag.get("id") or ""
        classes = tag.get("class") or []
        class_str = " ".join(classes) if isinstance(classes, (list, tuple)) else str(classes)
        combined = f"{id_val} {class_str}"
        if JUNK_CLASS_ID_RE.search(combined):
            tag.decompose()


def dedupe_adjacent_repeats(text):
    """Collapse an immediately-repeated word block (2+ words) to one copy.

    Some sites implement animated/rotating headline text as two DOM copies
    of the same phrase (one visible, one for a CSS transition) with no
    class/id hint we can strip on. get_text() then yields e.g. "Earn Cash
    Back Earn Cash Back when you when you". Collapse blocks of >=2 words
    that repeat back-to-back; single-word repeats are left alone since
    those are often intentional emphasis ("no no", "go go go").
    """
    words = text.split()
    n = len(words)
    out = []
    i = 0
    while i < n:
        collapsed = False
        max_k = (n - i) // 2
        for k in range(max_k, 1, -1):
            if words[i:i + k] == words[i + k:i + 2 * k]:
                out.extend(words[i:i + k])
                i += 2 * k
                collapsed = True
                break
        if not collapsed:
            out.append(words[i])
            i += 1
    return " ".join(out)


def get_element_text(tag):
    """Prefer a non-empty aria-label (often the clean, complete copy behind
    a decorative/animated headline) over get_text(), then dedupe any
    remaining immediate phrase repeats."""
    aria = normalize_ws(tag.get("aria-label") or "")
    text = aria if aria else normalize_ws(tag.get_text(" "))
    return dedupe_adjacent_repeats(text)


def looks_like_code(text):
    """Heuristic: is this <p> actually a line of an embedded SDK code sample?

    Many devtool landing pages (heavily represented in YC) render quickstart
    code blocks as one <p>/line instead of semantic <pre>/<code>, so our
    generic walk would otherwise capture "import Foo", "class Bar {", etc.
    as paragraph copy. Matching is deliberately case-sensitive on keywords
    so a normal, capitalized English sentence ("Import your contacts...")
    is never caught.
    """
    t = text.strip()
    if not t:
        return False
    if t.startswith("//"):
        return True
    if CODE_START_RE.match(t):
        return True
    if t.endswith("{") or t.endswith("};") or t.endswith(");"):
        return True
    if FUNC_CALL_RE.match(t):
        return True
    return False


def is_cta_anchor(tag):
    classes = tag.get("class") or []
    class_str = " ".join(classes) if isinstance(classes, (list, tuple)) else str(classes)
    role = tag.get("role") or ""
    if role.strip().lower() == "button":
        return True
    if CTA_RE.search(class_str):
        return True
    if CTA_RE.search(role):
        return True
    return False


def extract_elements(soup):
    elements = []
    seen_texts = set()
    root = soup.body or soup
    for tag in root.find_all(WALK_TAGS):
        name = tag.name
        if name == "a":
            if not is_cta_anchor(tag):
                continue
            out_tag = "button"
        elif name == "button":
            out_tag = "button"
        elif name == "p":
            out_tag = "p"
        else:
            out_tag = name  # h1..h4

        text = get_element_text(tag)
        if not text:
            continue

        words = text.split()
        wc = len(words)
        if out_tag == "p" and (wc < 2 or wc > 90):
            continue
        if out_tag == "p" and looks_like_code(text):
            continue
        if out_tag == "button":
            if wc < 1 or wc > 7:
                continue
            if BUTTON_JUNK_RE.match(text):
                continue

        if not any(ch.isalpha() for ch in text):
            continue  # symbols/numbers only

        if text in seen_texts:
            continue
        seen_texts.add(text)

        elements.append({"tag": out_tag, "text": text})
    return elements


def get_title(soup):
    if soup.title:
        return normalize_ws(soup.title.get_text(" "))
    return ""


def get_meta_content(soup, attr_name, value):
    tag = soup.find("meta", attrs={attr_name: re.compile("^%s$" % re.escape(value), re.I)})
    if tag and tag.get("content"):
        return normalize_ws(tag.get("content"))
    return ""


def get_meta_description(soup):
    return get_meta_content(soup, "name", "description")


def get_og(soup, key):
    val = get_meta_content(soup, "property", key)
    if val:
        return val
    return get_meta_content(soup, "name", key)


def get_html_lang(soup):
    if soup.html and soup.html.get("lang"):
        return normalize_ws(soup.html.get("lang"))
    return ""


def detect_lang(text):
    text = (text or "").strip()
    if len(text) < 20 or detect is None:
        return None
    try:
        return detect(text)
    except LangDetectException:
        return None
    except Exception:
        return None


def parse_page(content_bytes):
    soup = BeautifulSoup(content_bytes, "lxml")
    strip_junk(soup)
    elements = extract_elements(soup)
    title = get_title(soup)
    meta_description = get_meta_description(soup)
    og_title = get_og(soup, "og:title")
    og_description = get_og(soup, "og:description")
    html_lang = get_html_lang(soup)

    lang_source = " ".join(e["text"] for e in elements)
    if not lang_source.strip():
        lang_source = normalize_ws(soup.get_text(" "))[:2000]
    lang = detect_lang(lang_source)

    return {
        "title": title,
        "meta_description": meta_description,
        "og_title": og_title,
        "og_description": og_description,
        "html_lang": html_lang,
        "lang": lang,
        "elements": elements,
    }


# --------------------------------------------------------------------------
# Task loading
# --------------------------------------------------------------------------

def load_yc_tasks(path):
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    tasks = []
    seen = set()
    for r in data:
        if r.get("status") not in YC_ALLOWED_STATUS:
            continue
        website = ensure_scheme(r.get("website") or "")
        if not website:
            continue
        if website in seen:
            continue
        seen.add(website)
        yc_meta = {
            "one_liner": r.get("one_liner"),
            "long_description": r.get("long_description"),
            "industry": r.get("industry"),
            "subindustry": r.get("subindustry"),
            "industries": r.get("industries"),
            "tags": r.get("tags"),
            "batch": r.get("batch"),
            "status": r.get("status"),
        }
        tasks.append({"url": website, "name": r.get("name"), "yc": yc_meta})
    return tasks


def load_txt_tasks(path):
    tasks = []
    seen = set()
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            u = ensure_scheme(line.strip())
            if not u or u.startswith("#"):
                continue
            if u in seen:
                continue
            seen.add(u)
            tasks.append({"url": u, "name": None, "yc": None})
    return tasks


def load_done_urls(*paths):
    done = set()
    for p in paths:
        if p and os.path.exists(p):
            with open(p, "r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        rec = json.loads(line)
                    except json.JSONDecodeError:
                        continue
                    u = rec.get("url")
                    if u:
                        done.add(u)
    return done


# --------------------------------------------------------------------------
# Worker
# --------------------------------------------------------------------------

def process_task(session, source, task):
    url = task["url"]
    content, err, status, final_url = fetch(session, url)
    if err or content is None:
        return False, {"url": url, "error": err or "empty response body"}
    try:
        parsed = parse_page(content)
    except Exception as e:
        return False, {"url": url, "error": f"parse error: {type(e).__name__}: {e}"}

    record = {
        "source": source,
        "name": task.get("name"),
        "url": url,
        "final_url": final_url,
        "status": status,
        "title": parsed["title"],
        "meta_description": parsed["meta_description"],
        "og_title": parsed["og_title"],
        "og_description": parsed["og_description"],
        "html_lang": parsed["html_lang"],
        "lang": parsed["lang"],
        "yc": task.get("yc"),
        "elements": parsed["elements"],
    }
    return True, record


# --------------------------------------------------------------------------
# Main
# --------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--urls-json", help="YC-style JSON list of company records")
    ap.add_argument("--urls-txt", help="Plain text file, one URL per line")
    ap.add_argument("--source", required=True, help="Source tag, e.g. yc or gallery")
    ap.add_argument("--out", required=True, help="Output JSONL path")
    ap.add_argument("--workers", type=int, default=DEFAULT_WORKERS)
    ap.add_argument("--limit", type=int, default=None, help="Only process first N tasks (debug)")
    args = ap.parse_args()

    if not args.urls_json and not args.urls_txt:
        ap.error("one of --urls-json or --urls-txt is required")

    tasks = []
    if args.urls_json:
        tasks.extend(load_yc_tasks(args.urls_json))
    if args.urls_txt:
        tasks.extend(load_txt_tasks(args.urls_txt))

    out_path = args.out
    out_dir = os.path.dirname(out_path) or "."
    os.makedirs(out_dir, exist_ok=True)
    err_path = os.path.join(out_dir, f"errors_{args.source}.jsonl")

    done = load_done_urls(out_path, err_path)
    todo = [t for t in tasks if t["url"] not in done]
    if args.limit is not None:
        todo = todo[: args.limit]

    print(
        f"[{args.source}] total candidates={len(tasks)} already-done={len(done)} "
        f"to-fetch={len(todo)} workers={args.workers}",
        flush=True,
    )

    if not todo:
        print(f"[{args.source}] nothing to do.", flush=True)
        return

    session = build_session()
    out_f = open(out_path, "a", encoding="utf-8", buffering=1)
    err_f = open(err_path, "a", encoding="utf-8", buffering=1)

    processed = 0
    ok_count = 0
    err_count = 0
    start = time.time()

    try:
        with ThreadPoolExecutor(max_workers=args.workers) as pool:
            futures = {pool.submit(process_task, session, args.source, t): t for t in todo}
            for fut in as_completed(futures):
                task = futures[fut]
                try:
                    ok, record = fut.result()
                except Exception as e:
                    ok, record = False, {"url": task["url"], "error": f"worker crash: {type(e).__name__}: {e}"}

                if ok:
                    out_f.write(json.dumps(record, ensure_ascii=False) + "\n")
                    ok_count += 1
                else:
                    err_f.write(json.dumps(record, ensure_ascii=False) + "\n")
                    err_count += 1

                processed += 1
                if processed % PROGRESS_EVERY == 0 or processed == len(todo):
                    elapsed = time.time() - start
                    rate = processed / elapsed if elapsed > 0 else 0.0
                    with _print_lock:
                        print(
                            f"[{args.source}] {processed}/{len(todo)} processed "
                            f"(ok={ok_count} err={err_count}) "
                            f"elapsed={elapsed:.0f}s rate={rate:.1f}/s",
                            flush=True,
                        )
    finally:
        out_f.close()
        err_f.close()

    print(
        f"[{args.source}] done. ok={ok_count} err={err_count} out={out_path} errors={err_path}",
        flush=True,
    )


if __name__ == "__main__":
    main()
