"""
Scraper for nhatot.com - Hanoi rental listings, via its public JSON API.

nhatot's search-result HTML pages are behind a bot block (plain HTTP and
even TLS-impersonated requests get a 403), but the JSON API that its own
frontend calls (gateway.chotot.com) has no such protection - so no
browser/Playwright is needed here, just plain HTTP.

The API mixes "for sale" and "for rent" ads under the same category id;
each ad has a `type` field ("s" = sale, "u" = rent), so rent listings are
filtered out client-side after fetching.

Stage 1: download raw JSON pages (saved to disk, resumable).
Stage 2: parse ads from the saved pages into a CSV, keeping rent ads only.

Install:  pip install curl_cffi pandas
Run:      python nhatot_scraper.py --pages 3      # small test first!
"""
import json
import time
import random
import argparse
from datetime import date, datetime
from pathlib import Path

import pandas as pd

try:
    from curl_cffi import requests as http
    SESSION = http.Session(impersonate="chrome")
except ImportError:
    import requests as http
    SESSION = http.Session()
    SESSION.headers["User-Agent"] = ("Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                                     "AppleWebKit/537.36 Chrome/124.0 Safari/537.36")

API_URL = "https://gateway.chotot.com/v1/public/ad-listing"
REGION_HA_NOI = 12000
# Residential categories only (cg = Cho Tot's property-type category id)
CATEGORIES = {
    "can_ho_chung_cu": 1010,   # Can ho/Chung cu (apartment)
    "nha_rieng":       1020,   # Nha o (house)
    "nha_tro":         1050,   # Phong tro (room); NOT 1000, which is the parent "Bat dong san" category (mixes offices, houses, apartments)
}
RENT_TYPE = "u"                # ad["type"]: "s" = for sale, "u" = for rent
# District name (as it appears in processed/listings.csv) -> area_v2 id, read
# off the area_v2/area_name fields of already-downloaded ads.
DISTRICTS = {
    "Hoàn Kiếm": 12073, "Ba Đình": 12074, "Đống Đa": 12075, "Hai Bà Trưng": 12076,
    "Thanh Xuân": 12077, "Tây Hồ": 12078, "Cầu Giấy": 12079, "Hoàng Mai": 12080,
    "Long Biên": 12081, "Đông Anh": 12082, "Sóc Sơn": 12083, "Thanh Trì": 12084,
    "Hà Đông": 12086, "Đan Phượng": 12088, "Hoài Đức": 12089, "Quốc Oai": 12090,
    "Thạch Thất": 12091, "Chương Mỹ": 12092, "Thường Tín": 12093,
    "Nam Từ Liêm": 12121, "Ba Vì": 12122, "Gia Lâm": 12123, "Mê Linh": 12124,
    "Mỹ Đức": 12125, "Bắc Từ Liêm": 12129,
}
# Furnishing code -> label for RENT ads, confirmed from each ad's own
# feature_params.seo_structure (which pairs code with display label):
#   1 = Nội thất cao cấp, 2 = Nội thất đầy đủ, 3 = Nhà trống.
# Apartments/houses store it in ad["furnishing_sell"], rooms (cg=1050) in
# ad["furnishing_rent"]. Do not reuse sale-ad labels (3 = "Hoàn thiện cơ bản",
# 4 = "Bàn giao thô"): they don't apply to rentals.
# Codes 1 and 2 are merged into "Đầy đủ".
FURNISHING_LABELS = {
    1: "Đầy đủ",
    2: "Đầy đủ",
    3: "Trống",
}
RAW_DIR = Path("data/raw_pages_nhatot")
OUT_CSV = Path("data/interim/listings_nhatot.csv")


def fetch_page(cg, offset, limit, region=REGION_HA_NOI, area=None):
    params = {"cg": cg, "region_v2": region, "o": offset, "limit": limit}
    if area:
        # st=u,h asks the API for rent ads only, so a thin district isn't
        # drowned out by sale ads (the client-side type filter still applies)
        params.update(area_v2=area, st="u,h")
    for attempt in range(3):
        try:
            r = SESSION.get(API_URL, params=params, timeout=30)
            if r.status_code == 200:
                return r.text
            print(f"  HTTP {r.status_code}: {r.url}")
            if r.status_code in (403, 429):
                time.sleep(30 * (attempt + 1))
        except Exception as e:
            print(f"  error {e}")
            time.sleep(10)
    return None


def download_pages(cat, cg, max_pages, limit, area=None):
    folder = RAW_DIR / cat / (f"a{area}" if area else "")
    folder.mkdir(parents=True, exist_ok=True)
    for p in range(max_pages):
        offset = p * limit
        f = folder / f"o{offset}.json"
        if f.exists():
            continue                                  # resumable
        raw = fetch_page(cg, offset, limit, area=area)
        if not raw:
            print(f"  stopping {cat} at offset {offset}")
            break
        data = json.loads(raw)
        ads = data.get("ads", [])
        f.write_text(raw, encoding="utf-8")
        print(f"  saved {folder.relative_to(RAW_DIR)} offset={offset} ({len(ads)} ads, total={data.get('total')})")
        if not ads or offset + limit >= (data.get("total") or 0):
            break                                      # reached the end
        time.sleep(random.uniform(1, 3))                # be polite


# ---------- parsing ----------
def parse_page(raw_json, cat, offset):
    data = json.loads(raw_json)
    rows = []
    for ad in data.get("ads", []):
        if ad.get("type") != RENT_TYPE:
            continue                                  # skip for-sale ads
        list_time = ad.get("list_time")
        posted = (datetime.fromtimestamp(list_time / 1000).date().isoformat()
                  if list_time else None)
        rows.append({
            "listing_id": ad.get("list_id"),
            "category": cat,
            "url": f"https://www.nhatot.com/{ad.get('list_id')}.htm",
            "title": ad.get("subject"),
            "price_raw": ad.get("price_string"),
            "price": ad.get("price"),
            "area_raw": ad.get("size"),
            "bedrooms_raw": ad.get("rooms"),
            "toilets_raw": ad.get("toilets"),
            "furniture": FURNISHING_LABELS.get(
                ad.get("furnishing_sell") or ad.get("furnishing_rent")),
            "location_raw": " - ".join(filter(None, [
                ad.get("street_name"), ad.get("ward_name"), ad.get("area_name"),
            ])),
            "region_name": ad.get("region_name"),
            "posted_raw": posted,
            "body": (ad.get("body") or "")[:500],     # keep for debugging/cleaning
            "page_offset": offset,
            "scraped_at": date.today().isoformat(),
        })
    return rows


def parse_all():
    rows = []
    # <cat>/o*.json (city-wide) and <cat>/a<area_v2>/o*.json (per district)
    for f in sorted(RAW_DIR.rglob("o*.json")):
        cat = f.relative_to(RAW_DIR).parts[0]
        rows += parse_page(f.read_text(encoding="utf-8"), cat, int(f.stem[1:]))
    df = pd.DataFrame(rows).drop_duplicates("listing_id")
    OUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUT_CSV, index=False, encoding="utf-8-sig")   # utf-8-sig opens fine in Excel
    print(f"Parsed {len(df)} unique rent listings -> {OUT_CSV}")
    print(df[["category", "price_raw", "area_raw", "location_raw"]].head(10))
    print("\nMissing values per column:\n", df.isna().sum())


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--pages", type=int, default=3, help="max pages per category")
    ap.add_argument("--limit", type=int, default=50, help="ads per page (site caps at 50)")
    ap.add_argument("--categories", default=",".join(CATEGORIES),
                    help="comma-separated subset of: " + ", ".join(CATEGORIES))
    ap.add_argument("--districts", default="",
                    help="comma-separated district names (e.g. 'Hoàn Kiếm,Đông Anh'): "
                         "crawl only those districts' rent ads instead of the whole city")
    ap.add_argument("--parse-only", action="store_true")
    args = ap.parse_args()
    if not args.parse_only:
        districts = [d.strip() for d in args.districts.split(",") if d.strip()]
        for cat in args.categories.split(","):
            if not districts:
                print(f"== {cat}")
                download_pages(cat, CATEGORIES[cat], args.pages, args.limit)
            for d in districts:
                print(f"== {cat} / {d}")
                download_pages(cat, CATEGORIES[cat], args.pages, args.limit, area=DISTRICTS[d])
    parse_all()
