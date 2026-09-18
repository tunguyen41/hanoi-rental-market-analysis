"""
Scraper for alonhadat.com.vn - Hanoi rental listings.

Like nhatot.com, alonhadat has no Cloudflare-style bot protection - plain
HTTP works fine, no browser/Playwright needed. Its listing cards are also
marked up with schema.org microdata (itemprop attributes), so price, area,
address and post date come from structured attributes instead of regexed
free text (price even ships a raw VND integer in the `content` attribute).

Note: the site's address itemprops (streetAddress/addressLocality/
addressRegion) reflect Vietnam's post-2025 two-tier admin structure (ward +
province, no district), so cards also carry an `old-address` block with the
pre-reform "Quận/Huyện" text for anyone matching against older data.

Stage 1: download search-result HTML pages (saved raw to disk, resumable).
Stage 2: parse listing cards from the saved pages into a CSV.

Install:  pip install curl_cffi beautifulsoup4 lxml pandas
Run:      python alonhadat_scraper.py --pages 3      # small test first!
"""
import re
import time
import random
import argparse
from datetime import date
from pathlib import Path

import pandas as pd
from bs4 import BeautifulSoup

try:
    from curl_cffi import requests as http          # mimics Chrome, better vs bot checks
    SESSION = http.Session(impersonate="chrome")
except ImportError:
    import requests as http
    SESSION = http.Session()
    SESSION.headers["User-Agent"] = ("Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                                     "AppleWebKit/537.36 Chrome/124.0 Safari/537.36")

BASE = "https://alonhadat.com.vn"
# Residential categories only (open each once in a browser to confirm the URL works).
# Paths use "cho-thue-*" (allowed by robots.txt), not the disallowed "can-thue*".
CATEGORIES = {
    "can_ho_chung_cu": "/cho-thue-can-ho-chung-cu/ha-noi",
    "nha_rieng":       "/cho-thue-nha/ha-noi",
    "nha_tro":         "/cho-thue-phong-tro-nha-tro/ha-noi",
}
RAW_DIR = Path("data/raw_pages_alonhadat")
OUT_CSV = Path("data/interim/listings_alonhadat.csv")

ID_RE = re.compile(r"-(\d+)\.html$")


def fetch_page(url):
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


def page_url(path, page):
    return BASE + path + ("" if page == 1 else f"/trang-{page}")


def download_pages(cat, path, max_pages):
    folder = RAW_DIR / cat
    folder.mkdir(parents=True, exist_ok=True)
    for p in range(1, max_pages + 1):
        f = folder / f"p{p}.html"
        if f.exists():
            continue                                  # resumable
        html = fetch_page(page_url(path, p))
        if not html:
            print(f"  stopping {cat} at page {p}")
            break
        n_items = len(BeautifulSoup(html, "lxml").select("article.property-item"))
        if n_items == 0:
            print(f"  reached the end of {cat} at page {p}")
            break
        f.write_text(html, encoding="utf-8")
        print(f"  saved {cat} p{p} ({n_items} listings)")
        time.sleep(random.uniform(6, 12))             # be polite - site rate-limits (429) if hit too fast


# ---------- parsing ----------
def first_text(card, selector):
    el = card.select_one(selector)
    return el.get_text(" ", strip=True) if el else None


def parse_page(html, cat, page):
    soup = BeautifulSoup(html, "lxml")
    rows = []
    for card in soup.select("article.property-item"):
        link = card.select_one("a.link")
        href = link.get("href", "") if link else ""
        m = ID_RE.search(href)
        if not m:
            continue

        price_el = card.select_one(".price [itemprop=price]")
        price_vnd = (int(price_el["content"])
                     if price_el and price_el.get("content", "").isdigit() else None)
        time_el = card.select_one("time.created-date")

        rows.append({
            "listing_id": m.group(1),
            "category": cat,
            "url": href if href.startswith("http") else BASE + href,
            "title": first_text(card, "h3.property-title"),
            "price_vnd": price_vnd,
            "price_raw": first_text(card, ".price"),
            "area_raw": first_text(card, ".area [itemprop=value]"),
            "bedrooms_raw": first_text(card, ".bedroom [itemprop=value]"),
            "street": first_text(card, "[itemprop=streetAddress]"),
            "ward": first_text(card, "[itemprop=addressLocality]"),
            "city": first_text(card, "[itemprop=addressRegion]"),
            "old_address_raw": first_text(card, "p.old-address span"),
            "posted_raw": time_el.get("datetime") if time_el else None,
            "body": (first_text(card, "p.brief") or "")[:500],
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
    print(df[["category", "price_raw", "area_raw", "ward"]].head(10))
    print("\nMissing values per column:\n", df.isna().sum())


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--pages", type=int, default=3, help="max pages per category")
    ap.add_argument("--parse-only", action="store_true")
    args = ap.parse_args()
    if not args.parse_only:
        for cat, path in CATEGORIES.items():
            print(f"== {cat}")
            download_pages(cat, path, args.pages)
    parse_all()
