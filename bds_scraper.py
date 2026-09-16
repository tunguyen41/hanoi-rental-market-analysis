"""
Scraper for batdongsan.com.vn - Hanoi rental listing pages.

Stage 1: download search result pages (saved raw to disk, resumable).
Stage 2: parse listing cards from the saved pages into a CSV.

The site sits behind Cloudflare's JS challenge, so plain HTTP requests
(even TLS-impersonated ones) get a "Just a moment..." page instead of
real content. Stage 1 therefore drives a real Chromium browser via
Playwright, which executes the challenge like a normal browser would.

Install:  pip install curl_cffi beautifulsoup4 lxml pandas playwright
          playwright install chromium
Run:      python bds_scraper.py --pages 3      # small test first!
"""
import re
import time
import random
import argparse
from datetime import date
from pathlib import Path

import pandas as pd
from bs4 import BeautifulSoup

BASE = "https://batdongsan.com.vn"
# Residential categories only (open each once in a browser to confirm the URL works)
CATEGORIES = {
    "can_ho_chung_cu": "/cho-thue-can-ho-chung-cu-ha-noi",
    "chung_cu_mini":   "/cho-thue-can-ho-chung-cu-mini-ha-noi",
    "nha_rieng":       "/cho-thue-nha-rieng-ha-noi",
    "nha_tro":         "/cho-thue-nha-tro-phong-tro-ha-noi",
}
RAW_DIR = Path("data/raw_pages")
OUT_CSV = Path("data/interim/listings.csv")
PW_PROFILE_DIR = Path("pw_profile")   # persists Cloudflare clearance cookies between runs

# ---------- fetching (plain HTTP, kept for reference - Cloudflare blocks this) ----------
try:
    from curl_cffi import requests as http          # mimics Chrome, better vs Cloudflare
    SESSION = http.Session(impersonate="chrome")
except ImportError:
    import requests as http
    SESSION = http.Session()
    SESSION.headers["User-Agent"] = ("Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                                     "AppleWebKit/537.36 Chrome/124.0 Safari/537.36")


def fetch_http(url):
    for attempt in range(3):
        try:
            r = SESSION.get(url, timeout=30)
            if r.status_code == 200:
                return r.text
            print(f"  HTTP {r.status_code}: {url}")
            if r.status_code in (403, 429):
                time.sleep(30 * (attempt + 1))       # back off hard if blocked
        except Exception as e:
            print(f"  error {e}")
            time.sleep(10)
    return None


# ---------- fetching (real browser via Playwright - passes the Cloudflare JS challenge) ----------
_pw = None
_browser_ctx = None
_page = None


def get_browser_page(headless=False):
    global _pw, _browser_ctx, _page
    if _page is not None:
        return _page
    from playwright.sync_api import sync_playwright
    _pw = sync_playwright().start()
    _browser_ctx = _pw.chromium.launch_persistent_context(
        user_data_dir=str(PW_PROFILE_DIR.resolve()),
        headless=headless,
        viewport={"width": 1366, "height": 900},
    )
    _page = _browser_ctx.new_page()
    return _page


def close_browser():
    global _pw, _browser_ctx, _page
    if _browser_ctx:
        _browser_ctx.close()
    if _pw:
        _pw.stop()
    _pw = _browser_ctx = _page = None


def fetch_browser(url, page):
    try:
        page.goto(url, timeout=45000, wait_until="domcontentloaded")
    except Exception as e:
        print(f"  browser nav error: {e}")
        return None

    deadline = time.time() + 20
    while "just a moment" in page.title().lower() and time.time() < deadline:
        time.sleep(1)

    if "just a moment" in page.title().lower():
        print("  Cloudflare challenge still showing - waiting up to 60s "
              "(solve it by hand in the browser window if it needs a click)...")
        try:
            page.wait_for_function(
                "!document.title.toLowerCase().includes('just a moment')", timeout=60000
            )
        except Exception:
            print("  gave up waiting for the challenge to clear")
            return None

    try:
        page.wait_for_selector('a[href*="-pr"]', timeout=15000)
    except Exception:
        pass  # page may genuinely have no listings; still return what loaded

    return page.content()


def download_pages(cat, path, max_pages, fetch_fn):
    folder = RAW_DIR / cat
    folder.mkdir(parents=True, exist_ok=True)
    for p in range(1, max_pages + 1):
        f = folder / f"p{p}.html"
        if f.exists():
            continue                                  # resumable
        url = BASE + path + ("" if p == 1 else f"/p{p}")
        html = fetch_fn(url)
        if not html:
            print(f"  stopping {cat} at page {p}")
            break
        f.write_text(html, encoding="utf-8")
        print(f"  saved {cat} p{p}")
        time.sleep(random.uniform(10, 20))            # be polite / avoid tripping Cloudflare


# ---------- parsing ----------
ID_RE = re.compile(r"-pr(\d+)$")
PRICE_RE = re.compile(r"(Giá thỏa thuận|[\d.,]+\s*(?:tỷ|triệu|nghìn)(?:/(?:tháng|m²))?)")
AREA_RE = re.compile(r"([\d.,]+)\s*m²")
LOC_RE = re.compile(r"((?:Q\.|H\.|TX\.)[^·]*?\([^)]*\))")


def first_text(card, selector):
    el = card.select_one(selector)
    return el.get_text(" ", strip=True) if el else None


def parse_page(html, cat, page):
    soup = BeautifulSoup(html, "lxml")
    rows, seen = [], set()
    for a in soup.select('a[href*="-pr"]'):
        href = a.get("href", "").split("?")[0]
        m = ID_RE.search(href)
        if not m or m.group(1) in seen:
            continue
        seen.add(m.group(1))
        text = a.get_text(" ", strip=True)
        # Class selectors first (verify in DevTools), regex on card text as fallback
        price = first_text(a, ".re__card-config-price") or \
            (PRICE_RE.search(text).group(1) if PRICE_RE.search(text) else None)
        area = first_text(a, ".re__card-config-area") or \
            (AREA_RE.search(text).group(0) if AREA_RE.search(text) else None)
        loc = first_text(a, ".re__card-location") or \
            (LOC_RE.search(text).group(1) if LOC_RE.search(text) else None)
        rows.append({
            "listing_id": m.group(1),
            "category": cat,
            "url": href if href.startswith("http") else BASE + href,
            "title": a.get("title") or first_text(a, ".re__card-title"),
            "price_raw": price,
            "area_raw": area,
            "bedrooms_raw": first_text(a, ".re__card-config-bedroom"),
            "toilets_raw": first_text(a, ".re__card-config-toilet"),
            "location_raw": loc,
            "posted_raw": first_text(a, ".re__card-published-info-published-at"),
            "card_text": text[:500],                  # keep for debugging/cleaning
            "page": page,
            "scraped_at": date.today().isoformat(),
        })
    return rows


def parse_all():
    rows = []
    for f in sorted(RAW_DIR.glob("*/p*.html")):
        rows += parse_page(f.read_text(encoding="utf-8"), f.parent.name, int(f.stem[1:]))
    df = pd.DataFrame(rows).drop_duplicates("listing_id")
    OUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUT_CSV, index=False, encoding="utf-8-sig")   # utf-8-sig opens fine in Excel
    print(f"Parsed {len(df)} unique listings -> {OUT_CSV}")
    print(df[["category", "price_raw", "area_raw", "location_raw"]].head(10))
    print("\nMissing values per column:\n", df.isna().sum())


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--pages", type=int, default=3, help="max pages per category")
    ap.add_argument("--parse-only", action="store_true")
    ap.add_argument("--engine", choices=["browser", "http"], default="browser",
                     help="browser (default) drives real Chromium to pass Cloudflare; "
                          "http is the old plain-request path, which Cloudflare blocks")
    ap.add_argument("--headless", action="store_true",
                     help="run the browser hidden (more likely to be blocked by Cloudflare)")
    args = ap.parse_args()
    if not args.parse_only:
        if args.engine == "browser":
            page = get_browser_page(headless=args.headless)
            fetch_fn = lambda u: fetch_browser(u, page)
        else:
            fetch_fn = fetch_http
        try:
            for cat, path in CATEGORIES.items():
                print(f"== {cat}")
                download_pages(cat, path, args.pages, fetch_fn=fetch_fn)
        finally:
            if args.engine == "browser":
                close_browser()
    parse_all()
