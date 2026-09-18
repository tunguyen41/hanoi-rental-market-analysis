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


TOILET_RE = re.compile(r"(?<![\w.,])(\d{1,2})\s*(?:phòng\s+|nhà\s+)?(?:wc|vs|toilet|vệ sinh)\b",
                       re.IGNORECASE)


def extract_toilets(text):
    """'2PN-2WC' / '01 vệ sinh' -> 2 / 1. Ambiguous (differing counts) or absent -> NaN."""
    if pd.isna(text):
        return pd.NA
    counts = {int(n) for n in TOILET_RE.findall(str(text))}
    counts = {c for c in counts if 1 <= c <= 10}
    return counts.pop() if len(counts) == 1 else pd.NA


FURNITURE_FULL_RE = re.compile(
    r"full nội thất|full đồ|full nt|đầy đủ nội thất|nội thất đầy đủ|đủ nội thất|đủ đồ"
    r"|nội thất cao cấp|nội thất sang trọng", re.IGNORECASE)
FURNITURE_NOT_FULL_RE = re.compile(
    r"không nội thất|ko nội thất|nhà trống|nguyên bản|bàn giao thô|chưa có nội thất"
    r"|nội thất cơ bản|đồ cơ bản|hoàn thiện cơ bản", re.IGNORECASE)


def extract_furniture_full(text):
    """'Đầy đủ' when the text says full furniture and mentions no other level, else NaN.
    Validated on 937 labeled nhatot rentals: 100% precision (479/479), 57% recall. Deriving
    'Trống' from text was only ~24% precise, so it is deliberately never derived.
    NaN means unknown, not unfurnished."""
    if pd.isna(text):
        return pd.NA
    t = str(text)
    return "Đầy đủ" if FURNITURE_FULL_RE.search(t) and not FURNITURE_NOT_FULL_RE.search(t) else pd.NA


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
        "toilets": (df["title"].fillna("") + " " + df["body"].fillna("")).map(extract_toilets),  # derived from text; no structured field
        "furniture": (df["title"].fillna("") + " " + df["body"].fillna("")).map(extract_furniture_full),  # derived from text; no structured field
        "city": df["city"],
        "district": district.map(strip_admin_prefix),
        "ward": df["ward"].map(strip_admin_prefix),
        "location_raw": location_raw,
        "posted_date": pd.to_datetime(df["posted_raw"], errors="coerce"),
        "scraped_at": df["scraped_at"],
    })


def drop_implausible(df):
    """Drop rows that are data errors, not real market observations: areas <= 5 m2
    (typos) and rent > 1M VND/m2 outside nha_rieng (price typos, whole-building leases)."""
    ppm2 = df["price_vnd"] / df["area_m2"]
    bad = (df["area_m2"] <= 5) | ((ppm2 > 1_000_000) & (df["category"] != "nha_rieng"))
    if bad.any():
        print(f"\n  dropping {bad.sum()} implausible rows (area <= 5 m2, or rent > 1M VND/m2 outside nha_rieng):")
        print(df.loc[bad, ["listing_id", "category", "price_vnd", "area_m2"]].to_string(index=False))
    return df[~bad]


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

    combined = drop_implausible(pd.concat(frames, ignore_index=True).drop_duplicates("listing_id"))

    OUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    combined.to_csv(OUT_CSV, index=False, encoding="utf-8-sig")
    print(f"\nWrote {len(combined)} listings -> {OUT_CSV}")
    print(combined["source"].value_counts())
    print("\nMissing values per column:\n", combined.isna().sum())


if __name__ == "__main__":
    main()
