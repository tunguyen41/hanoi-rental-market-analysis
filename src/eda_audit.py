"""Reproducible audit; preserves input rows and keeps review decisions in review_queue.csv."""
from pathlib import Path
import re
import unicodedata

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
CATEGORIES = ['can_ho_chung_cu', 'nha_rieng', 'nha_tro']
SOURCES = ['nhatot', 'alonhadat']
LABELS = dict(zip(CATEGORIES, ['Apartments', 'Private houses', 'Rental rooms']))
SEED = 42
AREA_EDGES = {
    'can_ho_chung_cu': [0, 40, 60, 80, 100, 150, np.inf],
    'nha_rieng': [0, 30, 50, 80, 120, 200, np.inf],
    'nha_tro': [0, 20, 30, 40, 60, np.inf],
}
SCOPE_RE = re.compile(r't(?:òa|oà) nhà|khách sạn|mặt bằng|văn phòng|kinh doanh|cả t(?:òa|oà)|sàn|\b(?:mbkd|mkbd|kd|vp)\b|mọi mô hình|thuê toà căn hộ|thuê tòa căn hộ', re.I)
DECISIONS = ['scope_decision', 'price_decision', 'area_decision', 'review_note', 'reviewer']


def normalize(value):
    if pd.isna(value):
        return ''
    return re.sub(r'\s+', ' ', unicodedata.normalize('NFC', str(value))).strip()


def number_vn(text):
    text = text.replace(' ', '')
    if ',' in text:
        return float(text.replace('.', '').replace(',', '.'))
    if re.fullmatch(r'\d{1,3}(?:\.\d{3})+', text):
        return float(text.replace('.', ''))
    return float(text)


def parse_monthly_price(raw, numeric, area):
    """Validate against the displayed unit; never multiply an already total price twice."""
    raw = normalize(raw).lower()
    if not np.isfinite(numeric) or numeric <= 0:
        return np.nan, 'invalid_numeric'
    if 'tháng' not in raw or 'thỏa thuận' in raw or 'thoả thuận' in raw:
        return np.nan, 'unknown_unit'
    match = re.search(r'(\d[\d.,]*)\s*(triệu|ngàn|nghìn|đồng|đ)', raw)
    if not match:
        return np.nan, 'unknown_amount'
    amount = number_vn(match[1]) * {'triệu': 1e6, 'ngàn': 1e3, 'nghìn': 1e3, 'đồng': 1, 'đ': 1}[match[2]]
    per_m2 = bool(re.search(r'/\s*m[²2]', raw))
    if per_m2:
        if not np.isfinite(area) or area <= 0:
            return np.nan, 'per_m2_missing_area'
        expected = amount * area
        if np.isclose(numeric, expected, rtol=.005, atol=1):
            return float(numeric), 'per_m2_total_verified'
        if np.isclose(numeric, amount, rtol=.005, atol=1):
            return float(expected), 'per_m2_converted'
        return np.nan, 'raw_numeric_mismatch'
    if not np.isclose(numeric, amount, rtol=.005, atol=1):
        return np.nan, 'raw_numeric_mismatch'
    return float(numeric), 'monthly_verified'


def duplicate_groups(df):
    """Conservative candidate groups, independent of price; not verified properties."""
    parent = list(range(len(df)))
    def find(i):
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i
    def union(a, b):
        parent[find(b)] = find(a)
    keys = []
    for _, r in df.iterrows():
        title = re.sub(r'[^\w\s]', ' ', normalize(r.title).lower())
        title = re.sub(r'\s+', ' ', title).strip()
        keys.append((r.category, normalize(r.district).lower(), normalize(r.ward).lower(), title))
    for values in [df.listing_id.fillna('').tolist(), df.url.fillna('').tolist(), keys]:
        seen = {}
        for i, value in enumerate(values):
            if not value or (isinstance(value, tuple) and (len(value[-1]) < 25 or not value[1] or not value[2])):
                continue
            if value in seen:
                union(seen[value], i)
            else:
                seen[value] = i
    groups = {}
    for i in range(len(df)):
        groups.setdefault(find(i), []).append(i)
    ids = [''] * len(df)
    for members in groups.values():
        if len(members) > 1:
            group = 'candidate_' + min(df.iloc[members].listing_id.astype(str))
            for i in members:
                ids[i] = group
    return pd.Series(ids, index=df.index, dtype='str')


def read_reviews(path, df):
    if not path.exists():
        return pd.DataFrame(columns=['listing_id', *DECISIONS])
    reviews = pd.read_csv(path, keep_default_na=False)
    if not set(['listing_id', *DECISIONS]).issubset(reviews):
        raise ValueError('review_queue.csv is missing decision columns; the existing file will not be overwritten.')
    if reviews.listing_id.duplicated().any():
        raise ValueError('Review queue contains duplicate listing IDs.')
    for col in DECISIONS[:3]:
        if not reviews[col].isin(['', 'keep', 'exclude', 'unresolved']).all():
            raise ValueError(f'{col}: use keep/exclude/unresolved or leave blank.')
    decided = reviews[DECISIONS[:3]].isin(['keep', 'exclude']).any(axis=1)
    if (decided & (reviews.review_note.eq('') | reviews.reviewer.eq(''))).any():
        raise ValueError('Review decisions require review_note and reviewer.')
    # Fail safely if a reviewed listing has changed: decisions must refer to this snapshot.
    current = df.set_index('listing_id')
    for _, row in reviews.loc[decided].iterrows():
        if row.listing_id not in current.index:
            continue
        r = current.loc[row.listing_id]
        if isinstance(r, pd.DataFrame):
            raise ValueError('Cannot apply a review to a duplicated input listing ID.')
        for c in ['title', 'price_raw', 'area_raw']:
            if c not in row:
                raise ValueError(f'Review queue is missing context field {c}.')
            if c == 'area_raw':
                old, new = pd.to_numeric(row[c], errors='coerce'), pd.to_numeric(r[c], errors='coerce')
                same = (pd.isna(old) and pd.isna(new)) or old == new
            else:
                same = normalize(row[c]) == normalize(r[c])
            if not same:
                raise ValueError(f'Saved review no longer matches {row.listing_id}/{c}; reassess the decision.')
    return reviews[['listing_id', *DECISIONS]]


def prepare_analysis(raw, reviews=None):
    required = {'source', 'listing_id', 'category', 'url', 'title', 'price_vnd', 'price_raw',
                'area_m2', 'area_raw', 'bedrooms', 'toilets', 'furniture', 'city', 'district',
                'ward', 'location_raw', 'posted_date', 'scraped_at'}
    if not required.issubset(raw):
        raise ValueError(f'Missing columns: {sorted(required - set(raw))}')
    df = raw.copy().reset_index(drop=True)
    # Original fields stay intact; analysis fields carry coercion/normalization.
    price = pd.to_numeric(df.price_vnd, errors='coerce').astype(float)
    area = pd.to_numeric(df.area_m2, errors='coerce').astype(float)
    parsed = [parse_monthly_price(r, p, a) for r, p, a in zip(df.price_raw, price, area)]
    df['price_month_vnd'] = [p[0] for p in parsed]
    df['price_unit_status'] = [p[1] for p in parsed]
    df['district_analysis'] = df.district.map(normalize).replace('', np.nan)
    title = df.title.map(normalize).str.lower()
    shared = title.str.contains(r'ở ghép|phòng ghép|ký túc|giường tầng|/\s*người')
    category_conflict = (df.category.eq('nha_rieng') & title.str.contains(r'phòng trọ|cho thuê phòng'))
    category_conflict |= df.category.eq('nha_tro') & title.str.contains(r'nhà \d+ tầng \d+ phòng ngủ')
    df['scope_flag'] = np.where(title.str.contains(SCOPE_RE) | shared | category_conflict, 'suspicious', 'normal')
    df['duplicate_group_id'] = duplicate_groups(df)
    flags = [[] for _ in range(len(df))]
    def flag(mask, name):
        for idx in df.index[pd.Series(mask, index=df.index).fillna(False)]:
            flags[idx].append(name)
    flag(df.scope_flag.eq('suspicious'), 'scope_suspicious')
    flag(shared, 'shared_or_per_person')
    flag(category_conflict, 'category_title_conflict')
    flag(df.price_month_vnd.isna(), 'price_unit_unverified')
    flag(~np.isfinite(area) | area.le(5), 'area_missing_or_invalid')
    flag(df.duplicate_group_id.ne(''), 'duplicate_candidate')
    flag(df.listing_id.isna() | df.listing_id.duplicated(False), 'invalid_or_duplicate_id')
    flag(~df.source.isin(SOURCES) | ~df.category.isin(CATEGORIES) | ~df.city.map(normalize).eq('Hà Nội'), 'schema_or_city')
    flag(df.district_analysis.isna(), 'district_missing')
    for date in ['posted_date', 'scraped_at']:
        flag(pd.to_datetime(df[date], errors='coerce').isna(), f'{date}_invalid')
    flag(pd.to_datetime(df.posted_date, errors='coerce') > pd.to_datetime(df.scraped_at, errors='coerce'), 'posted_after_parse_date')
    ambiguous_price = title.str.contains(r'/\s*ngày|theo ngày|th[oỏ]a thuận|thoả thuận') | price.lt(1e6)
    flag(ambiguous_price, 'price_needs_review')
    area_question = area.gt(1000)
    flag(area_question, 'area_needs_review')
    df['area_analysis_m2'] = area.where(np.isfinite(area) & area.gt(5))
    for col in ['bedrooms', 'toilets']:
        value = pd.to_numeric(df[col], errors='coerce')
        valid = np.isfinite(value) & value.ge(0) & value.mod(1).eq(0)
        # Counts may describe all rooms in a building instead of the advertised unit.
        questionable = value.gt(10) | (df.category.eq('nha_tro') & value.gt(5))
        flag(value.notna() & (~valid | questionable), f'{col}_unit_ambiguous')
        df[f'{col}_analysis'] = value.where(valid & ~questionable)
    for category in CATEGORIES:
        m = df.category.eq(category) & df.price_month_vnd.notna()
        if m.sum() >= 30:
            low, high = df.loc[m, 'price_month_vnd'].quantile([.01, .99])
            flag(m & ((df.price_month_vnd < low) | (df.price_month_vnd > high)), 'price_tail_p01_p99')
    df['quality_flags'] = [';'.join(f) for f in flags]
    if reviews is not None and len(reviews):
        df = df.merge(reviews, on='listing_id', how='left', validate='many_to_one')
    for col in DECISIONS:
        if col not in df:
            df[col] = ''
        df[col] = df[col].fillna('')
    # Explicit reviewer uncertainty also applies to records missed by keyword rules.
    manual_scope = df.scope_decision.eq('unresolved') & df.scope_flag.eq('normal')
    df.loc[manual_scope, 'scope_flag'] = 'suspicious'
    df.loc[manual_scope, 'quality_flags'] = (df.loc[manual_scope, 'quality_flags'] + ';manual_scope_review').str.strip(';')
    pending_scope = df.scope_flag.eq('suspicious') & ~df.scope_decision.isin(['keep', 'exclude'])
    pending_price = (ambiguous_price | df.price_decision.eq('unresolved')) & ~df.price_decision.isin(['keep', 'exclude'])
    pending_area = (area_question | df.area_decision.eq('unresolved')) & ~df.area_decision.isin(['keep', 'exclude'])
    df.loc[pending_area | df.area_decision.eq('exclude'), 'area_analysis_m2'] = np.nan
    df.loc[df.price_decision.eq('exclude'), 'price_unit_status'] = 'review_excluded'
    df.loc[pending_price | df.price_decision.eq('exclude'), 'price_month_vnd'] = np.nan
    df['review_status'] = np.select(
        [pending_scope | pending_price | pending_area, df[DECISIONS[:3]].isin(['keep', 'exclude']).any(axis=1)],
        ['pending', 'reviewed'], default='not_required')
    reasons = [[] for _ in range(len(df))]
    def exclude(mask, name):
        for idx in df.index[mask]:
            reasons[idx].append(name)
    exclude(df.price_month_vnd.isna(), 'price_unverified_or_excluded')
    exclude(pending_scope, 'scope_pending')
    exclude(df.scope_decision.eq('exclude'), 'outside_residential_scope')
    exclude(df.quality_flags.str.contains('schema_or_city|invalid_or_duplicate_id'), 'schema_or_id')
    df['exclusion_reason'] = [';'.join(r) for r in reasons]
    df['include_price'] = df.exclusion_reason.eq('')
    df['include_ppm2'] = df.include_price & df.category.isin(['can_ho_chung_cu', 'nha_tro']) & df.area_analysis_m2.notna()
    df['price_million'] = df.price_month_vnd / 1e6
    df['price_per_m2'] = df.price_month_vnd / df.area_analysis_m2
    df['log_price'] = np.log(df.price_month_vnd)
    df['log_area'] = np.log(df.area_analysis_m2)
    df['furniture_analysis'] = df.furniture.replace({'Đầy đủ': 'Furnished', 'Trống': 'Unfurnished'}).fillna('Unknown')
    df['furniture_origin'] = np.where(df.source.eq('nhatot'), 'structured', 'text_derived')
    df['toilets_origin'] = df.furniture_origin
    df['area_band'] = 'Unknown'
    for category, edges in AREA_EDGES.items():
        labels = [f'[{a:g}, {b:g})' for a, b in zip(edges[:-1], edges[1:])]
        mask = df.category.eq(category)
        bands = pd.cut(df.loc[mask, 'area_analysis_m2'], edges, labels=labels, right=False)
        df.loc[mask, 'area_band'] = bands.astype('str').fillna('Unknown')
    df['bedroom_group'] = df.bedrooms_analysis.map(lambda x: 'Unknown' if pd.isna(x) else ('4+' if x >= 4 else str(int(x))))
    df['split_group'] = df.duplicate_group_id.where(df.duplicate_group_id.ne(''), 'listing_' + df.listing_id.astype(str))
    return df


def audit_summary(df):
    return pd.DataFrame({'stage': ['Input', 'Flagged (including missing area)', 'Review pending',
                                   'Excluded from price set', 'Final residential price set', 'Final primary price/m² set'],
                         'n': [len(df), df.quality_flags.ne('').sum(), df.review_status.eq('pending').sum(),
                               (~df.include_price).sum(), df.include_price.sum(), df.include_ppm2.sum()]})


def markdown_table(df, digits=2):
    """Small Markdown export without an extra tabulate dependency."""
    def fmt(v):
        if pd.isna(v):
            return '—'
        if isinstance(v, (float, np.floating)):
            return f'{v:,.{digits}f}'
        return LABELS.get(str(v), str(v)).replace('|', '/').replace('\n', ' ')
    rows = ['| ' + ' | '.join(map(str, df.columns)) + ' |', '| ' + ' | '.join(['---'] * len(df.columns)) + ' |']
    rows += ['| ' + ' | '.join(fmt(v) for v in row) + ' |' for row in df.itertuples(index=False, name=None)]
    return '\n'.join(rows)


def run_audit(root=ROOT):
    root = Path(root)
    raw = pd.read_csv(root / 'data/processed/listings.csv')
    dest = root / 'data/analysis'
    dest.mkdir(parents=True, exist_ok=True)
    reviews = read_reviews(dest / 'review_queue.csv', raw)
    df = prepare_analysis(raw, reviews)
    df.to_csv(dest / 'listings_analysis.csv', index=False, encoding='utf-8-sig')
    # Keep all substantive flags but missing area alone does not require manual review.
    needs_review = df.quality_flags.str.replace('area_missing_or_invalid', '', regex=False).str.strip(';').ne('')
    needs_review |= df[DECISIONS].ne('').any(axis=1)
    context = ['listing_id', 'source', 'category', 'title', 'url', 'price_raw', 'price_vnd', 'area_raw', 'area_m2',
               'bedrooms', 'quality_flags', 'scope_flag', 'duplicate_group_id', 'review_status', *DECISIONS]
    df.loc[needs_review, context].to_csv(dest / 'review_queue.csv', index=False, encoding='utf-8-sig')
    write_dictionary(df, root)
    return df


def write_dictionary(df, root):
    descriptions = {
        'source': 'Source website; used to diagnose sample composition.',
        'listing_id': 'Source-prefixed listing ID; never a predictor.',
        'category': 'Original source category; does not guarantee residential scope.',
        'url': 'Original listing URL for verification; never a predictor.',
        'title': 'Original Vietnamese title; may contain prices, excluded from baseline predictors.',
        'price_vnd': 'Numeric price from the existing pipeline, preserved unchanged.',
        'price_raw': 'Original price text for unit verification; never a predictor.',
        'area_m2': 'Original numeric area; its meaning may differ across property categories.',
        'area_raw': 'Original area field from the existing pipeline, retained for verification.',
        'bedrooms': 'Original bedroom count; may describe an entire building.',
        'toilets': 'Original bathroom count; structured on Nhatot, text-derived on Alonhadat.',
        'furniture': 'Original furnishing label; missing does not mean unfurnished.',
        'city': 'Original city name.',
        'district': 'Original district name; not verified against current administrative boundaries.',
        'ward': 'Original ward/commune name; do not join across sources using name alone.',
        'location_raw': 'Original location text.',
        'posted_date': 'Source posting date; does not measure vacancy duration.',
        'scraped_at': 'Parser execution date, NOT a verified download date.',
        'price_month_vnd': 'Verified monthly asking rent in VND; missing if price/unit is unresolved. Proposed target.',
        'price_unit_status': 'Raw/numeric price validation result; per_m2_total_verified must not be multiplied again.',
        'district_analysis': 'NFC and whitespace-normalized district; no administrative remapping.',
        'scope_flag': 'normal = not flagged by scope rules; suspicious = contextual review required.',
        'duplicate_group_id': 'Candidate group from ID/URL or normalized title + district/ward/category, independent of price. Not a verified unique property.',
        'area_analysis_m2': 'Finite area >5 m²; missing when review is pending or excludes area. Private-house area definitions can still differ.',
        'bedrooms_analysis': 'Nonnegative integer; values >10, or >5 for rooms, are missing because they may count an entire building.',
        'toilets_analysis': 'Nonnegative integer; values >10, or >5 for rooms, are missing because they may count an entire building.',
        'quality_flags': 'Semicolon-separated flags. A flag does not automatically exclude a row.',
        'scope_decision': 'keep/exclude/unresolved or blank; contextual scope review decision.',
        'price_decision': 'keep/exclude/unresolved or blank; keep does not override a parser validation failure.',
        'area_decision': 'keep/exclude/unresolved or blank; affects area, not monthly rent eligibility by itself.',
        'review_note': 'Review evidence, context and decision limitations.',
        'reviewer': 'Person/agent that read the context; does not imply website verification.',
        'review_status': 'pending for unresolved scope/price/area; reviewed when a decision exists; otherwise not_required. Duplicate/count flags may remain unverified.',
        'exclusion_reason': 'Reason for exclusion from the monthly price set; empty when included. Not a separate price/m² exclusion reason.',
        'include_price': 'Boolean membership of the residential monthly asking-rent set under the audit policy.',
        'include_ppm2': 'Boolean membership of the primary price/m² set: include_price, apartment/room category and usable area.',
        'price_million': 'price_month_vnd / 1e6; never a predictor.',
        'price_per_m2': 'price_month_vnd / area_analysis_m2; secondary metric, exploratory for houses; never a predictor.',
        'log_price': 'ln(price_month_vnd); alternative target, never a predictor.',
        'log_area': 'ln(area_analysis_m2).',
        'furniture_analysis': 'English furnishing label: Furnished, Unfurnished or Unknown; no unfurnished imputation.',
        'furniture_origin': 'structured/text_derived according to the source adapter.',
        'toilets_origin': 'structured/text_derived according to the source adapter.',
        'area_band': 'Fixed category-specific [left, right) interval; Unknown when missing.',
        'bedroom_group': '0/1/2/3/4+/Unknown derived from bedrooms_analysis.',
        'split_group': 'Candidate duplicate group or a unique group for a single listing; used for proposed group splitting.',
    }
    table = pd.DataFrame([{'field': c, 'dtype': str(df[c].dtype), 'description': descriptions[c]} for c in df])
    text = '# Data dictionary\n\nThe original 18 input columns are preserved; analysis fields are separate. Missing values are retained, with no imputation during EDA. Original Vietnamese listing text, category codes and place names remain unchanged; English labels are used for presentation.\n\n'
    text += markdown_table(table)
    text += '\n\n## Area-band boundaries\n\n' + '\n'.join(f'- {LABELS[k]} ({k}): {v}, left-closed/right-open intervals, m².' for k, v in AREA_EDGES.items())
    text += '\n\nThe review queue stores review decisions and the current review list. Edit *_decision, review_note and reviewer, then rerun the audit. Saved decisions are rejected if title/price_raw/area_raw changes. Do not delete rows from the input.\n'
    (root / 'reports').mkdir(exist_ok=True)
    (root / 'reports/data_dictionary.md').write_text(text, encoding='utf-8')
