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
ADMIN_PREFIX_RE = re.compile(r"^(Q\.|H\.|TX\.|P\.|X\.|Quận|Huyện|Thị xã|Phường|Xã)\s*")
OLD_DISTRICT_RE = re.compile(r"((?:Quận|Huyện|Thị xã)\s+[^,]+)")


def strip_admin_prefix(text):
    """'Q. Nam Từ Liêm' / 'Quận Ba Đình' / 'P. Tây Mỗ mới' -> just the name."""
    if pd.isna(text):
        return pd.NA
    return ADMIN_PREFIX_RE.sub("", text.strip()) or pd.NA


# ---------- per-source adapters ----------
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


def adapt_alonhadat(df):
    # alonhadat's own address fields (street/ward/city) reflect the post-2025
    # ward+province structure with no district; old_address_raw carries the
    # pre-reform "...,Quận X,..." text, which is where district comes from.
    district = df["old_address_raw"].fillna("").str.extract(OLD_DISTRICT_RE)[0]
    location_raw = df[["street", "ward", "city"]].apply(
        lambda r: " - ".join(v for v in r if pd.notna(v)), axis=1
    )
    return pd.DataFrame({
        "source": "alonhadat",
        "listing_id": "alo_" + df["listing_id"].astype(str),
        "category": df["category"],
        "url": df["url"],
        "title": df["title"],
        "price_vnd": df["price_vnd"],
        "price_raw": df["price_raw"],
        "area_m2": pd.to_numeric(df["area_raw"], errors="coerce"),  # already a clean number, not Vietnamese-grouped text
        "area_raw": df["area_raw"],
        "bedrooms": df["bedrooms_raw"],
        "toilets": pd.NA,          # not available as a structured field on alonhadat
        "furniture": pd.NA,        # not available as a structured field on alonhadat
        "city": df["city"],
        "district": district.map(strip_admin_prefix),
        "ward": df["ward"].map(strip_admin_prefix),
        "location_raw": location_raw,
        "posted_date": pd.to_datetime(df["posted_raw"], errors="coerce"),
        "scraped_at": df["scraped_at"],
    })


# name -> (interim csv filename, adapter function)
SOURCES = {
    "nhatot": ("listings_nhatot.csv", adapt_nhatot),
    "alonhadat": ("listings_alonhadat.csv", adapt_alonhadat),
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
