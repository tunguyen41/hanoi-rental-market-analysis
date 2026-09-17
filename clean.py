"""
Transform + Load stage of the scraping pipeline.

Reads each source's interim CSV (produced by that source's *_scraper.py's
Extract stage), maps it into a common schema via a per-source adapter, then
concatenates everything and writes the unified, analysis-ready table to
data/processed/listings.csv.

Adding a new source later: write one adapt_<source>(df) function that maps
its raw columns to COMMON_COLUMNS, and register it in SOURCES. No changes
needed to the shared helpers or the other adapters.

Run:  python clean.py
"""
import re
from pathlib import Path

import pandas as pd

INTERIM_DIR = Path("data/interim")
OUT_CSV = Path("data/processed/listings.csv")

COMMON_COLUMNS = [
    "source", "listing_id", "category", "url", "title",
    "price_vnd", "price_raw", "area_m2", "area_raw",
    "bedrooms", "toilets", "furniture", "city", "district", "ward", "location_raw",
    "posted_date", "scraped_at",
]

# ---------- shared normalization helpers (used by every adapter) ----------
UNIT_MULTIPLIER = {"tỷ": 1_000_000_000, "triệu": 1_000_000, "nghìn": 1_000}
PRICE_RE = re.compile(r"([\d.,]+)\s*(tỷ|triệu|nghìn)", re.IGNORECASE)
RELATIVE_DAYS_RE = re.compile(r"(\d+)\s*ngày trước")
ADMIN_PREFIX_RE = re.compile(r"^(Q\.|H\.|TX\.|P\.|X\.|Quận|Huyện|Thị xã|Phường|Xã)\s*")


def parse_price_vnd(price_raw):
    """'10 triệu/tháng' -> 10_000_000.0. Unparseable (e.g. 'Giá thỏa thuận') -> NaN."""
    if pd.isna(price_raw):
        return pd.NA
    m = PRICE_RE.search(str(price_raw))
    if not m:
        return pd.NA
    number = float(m.group(1).replace(".", "").replace(",", "."))
    return number * UNIT_MULTIPLIER[m.group(2).lower()]


def parse_area_m2(area_raw):
    """Handles both '60 m²' (batdongsan) and a bare number (nhatot)."""
    if pd.isna(area_raw):
        return pd.NA
    m = re.search(r"[\d.,]+", str(area_raw))
    return float(m.group(0).replace(".", "").replace(",", ".")) if m else pd.NA


def strip_admin_prefix(text):
    """'Q. Nam Từ Liêm' / 'Quận Ba Đình' / 'P. Tây Mỗ mới' -> just the name."""
    if pd.isna(text):
        return pd.NA
    return ADMIN_PREFIX_RE.sub("", text.strip()) or pd.NA


def parse_relative_posted(posted_raw, scraped_at):
    """batdongsan gives relative Vietnamese text ('Đăng hôm nay', 'Đăng hôm qua',
    'Đăng N ngày trước') anchored to the scrape date, not an absolute date."""
    if pd.isna(posted_raw):
        return pd.NaT
    text = str(posted_raw).lower()
    anchor = pd.to_datetime(scraped_at)
    if "hôm nay" in text:
        return anchor
    if "hôm qua" in text:
        return anchor - pd.Timedelta(days=1)
    m = RELATIVE_DAYS_RE.search(text)
    if m:
        return anchor - pd.Timedelta(days=int(m.group(1)))
    return pd.NaT


# ---------- per-source adapters ----------
def adapt_batdongsan(df):
    loc = df["location_raw"].fillna("")
    district = loc.str.extract(r"((?:Q\.|H\.|TX\.)[^(]*)")[0].str.strip()
    ward = loc.str.extract(r"\(([^)]*)\)")[0].str.strip()
    return pd.DataFrame({
        "source": "batdongsan",
        "listing_id": "bds_" + df["listing_id"].astype(str),
        "category": df["category"],
        "url": df["url"],
        "title": df["title"],
        "price_vnd": df["price_raw"].map(parse_price_vnd),
        "price_raw": df["price_raw"],
        "area_m2": df["area_raw"].map(parse_area_m2),
        "area_raw": df["area_raw"],
        "bedrooms": df["bedrooms_raw"],
        "toilets": df["toilets_raw"],
        "furniture": pd.NA,        # not available as a structured field on batdongsan
        "city": "Hà Nội",
        "district": district.map(strip_admin_prefix),
        "ward": ward.map(strip_admin_prefix),
        "location_raw": df["location_raw"],
        "posted_date": [parse_relative_posted(p, s)
                         for p, s in zip(df["posted_raw"], df["scraped_at"])],
        "scraped_at": df["scraped_at"],
    })


def adapt_nhatot(df):
    parts = df["location_raw"].fillna("").str.split(" - ")
    ward = parts.map(lambda p: p[-2] if len(p) >= 2 else pd.NA)
    district = parts.map(lambda p: p[-1] if len(p) >= 1 and p[-1] else pd.NA)
    return pd.DataFrame({
        "source": "nhatot",
        "listing_id": "nt_" + df["listing_id"].astype(str),
        "category": df["category"],
        "url": df["url"],
        "title": df["title"],
        "price_vnd": df["price"],
        "price_raw": df["price_raw"],
        "area_m2": pd.to_numeric(df["area_raw"], errors="coerce"),  # already a clean number, not Vietnamese-grouped text
        "area_raw": df["area_raw"],
        "bedrooms": df["bedrooms_raw"],
        "toilets": df["toilets_raw"],
        "furniture": df["furniture"],
        "city": df["region_name"],
        "district": district.map(strip_admin_prefix),
        "ward": ward.map(strip_admin_prefix),
        "location_raw": df["location_raw"],
        "posted_date": pd.to_datetime(df["posted_raw"], errors="coerce"),
        "scraped_at": df["scraped_at"],
    })


# name -> (interim csv filename, adapter function)
SOURCES = {
    "batdongsan": ("listings.csv", adapt_batdongsan),
    "nhatot": ("listings_nhatot.csv", adapt_nhatot),
}


def main():
    frames = []
    for name, (filename, adapt) in SOURCES.items():
        path = INTERIM_DIR / filename
        if not path.exists():
            print(f"  skipping {name}: {path} not found")
            continue
        raw = pd.read_csv(path)
        df = adapt(raw)[COMMON_COLUMNS]
        print(f"  {name}: {len(df)} rows")
        frames.append(df)

    combined = pd.concat(frames, ignore_index=True).drop_duplicates("listing_id")

    OUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    combined.to_csv(OUT_CSV, index=False, encoding="utf-8-sig")
    print(f"\nWrote {len(combined)} listings -> {OUT_CSV}")
    print(combined["source"].value_counts())
    print("\nMissing values per column:\n", combined.isna().sum())


if __name__ == "__main__":
    main()
