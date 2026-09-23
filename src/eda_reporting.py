"""Question-driven summaries, standalone figures and a report from the frozen analysis CSV."""
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns

from .eda_audit import ROOT, CATEGORIES, SOURCES, LABELS, SEED, audit_summary, markdown_table

COLORS = {'nhatot': '#2678A6', 'alonhadat': '#D17C32'}
MISSING_COLS = ['area_m2', 'bedrooms', 'toilets', 'furniture']
FEATURES = ['category', 'district_analysis', 'area_analysis_m2', 'bedrooms_analysis', 'toilets_analysis', 'furniture', 'source']


def load_analysis(root=ROOT):
    df = pd.read_csv(Path(root) / 'data/analysis/listings_analysis.csv')
    for c in ['quality_flags', 'exclusion_reason', 'duplicate_group_id', 'review_note', 'scope_decision', 'price_decision', 'area_decision']:
        df[c] = df[c].fillna('')
    for c in ['include_price', 'include_ppm2']:
        if df[c].dtype != bool:
            raise ValueError(f'{c} must be Boolean in the analysis CSV; rerun the audit.')
    if not df.loc[df.include_price, 'price_month_vnd'].gt(0).all():
        raise ValueError('The price set contains invalid prices.')
    if not (df.include_ppm2 <= df.include_price).all():
        raise ValueError('The price/m² set must be a subset of the price set.')
    return df


def summarize(df, keys, value='price_million'):
    return df.groupby(keys, dropna=False, observed=True)[value].agg(
        n='count', median='median', q1=lambda x: x.quantile(.25), q3=lambda x: x.quantile(.75),
        p90=lambda x: x.quantile(.9)).reset_index()


def missing_summary(df):
    return df.groupby(['source', 'category'])[MISSING_COLS].agg(lambda x: 100*x.isna().mean()).round(1)


def feature_coverage(df):
    rows = []
    for label, group in [('ALL', df), *[(f'{s}/{c}', g) for (s, c), g in df.groupby(['source', 'category'])]]:
        for f in FEATURES:
            rows.append({'group': label, 'feature': f, 'n': len(group), 'available': group[f].notna().sum(), 'coverage_pct': group[f].notna().mean()*100})
    return pd.DataFrame(rows)


def area_missing_comparison(df):
    x = df.assign(area_available=df.area_analysis_m2.notna())
    return summarize(x, ['source', 'category', 'area_available'])


def source_comparison(price):
    usable = price[price.area_band.ne('Unknown') & price.district_analysis.notna()]
    keys = ['category', 'district_analysis', 'area_band']
    counts = usable.groupby(keys + ['source']).size().unstack('source', fill_value=0).reindex(columns=SOURCES, fill_value=0)
    counts['both_sources'] = counts[SOURCES].gt(0).all(axis=1)
    counts['eligible'] = counts[SOURCES].ge(10).all(axis=1)
    counts['main'] = counts[SOURCES].ge(30).all(axis=1)
    counts['n_total'] = counts[SOURCES].sum(axis=1)
    counts = counts.reset_index()
    qualified = usable.merge(counts.loc[counts.eligible, keys], on=keys, how='inner')
    return counts, summarize(qualified, keys + ['source']), qualified


def select_controlled(price, feature, nhatot_only=False):
    """Choose a well-covered stratum by counts alone, never by size of price difference."""
    x = price[price.area_band.ne('Unknown') & price.district_analysis.notna()].copy()
    if nhatot_only:
        x = x[x.source.eq('nhatot')]
    # Relax geography only when fully matched cells cannot support a comparison.
    for geography in [True, False]:
        keys = ['category'] + (['district_analysis'] if geography else []) + ['area_band'] + ([] if nhatot_only else ['source'])
        groups = []
        for key, g in x.groupby(keys):
            counts = g.loc[g[feature].ne('Unknown'), feature].value_counts()
            if counts.ge(10).sum() >= 2:
                groups.append((len(g), str(key), key, g))
        if groups:
            _, _, key, g = max(groups, key=lambda item: (item[0], item[1]))
            label = ' / '.join(LABELS.get(str(value), str(value)) for value in key) + (' / nhatot' if nhatot_only else '')
            if not geography:
                label += '\nExploratory: district not controlled (fully matched groups are too small)'
            return g, label
    return x.iloc[:0], 'No stratum has at least two known groups with ≥10 listings each.'


def sensitivity_sets(df):
    base = df[df.include_price].copy()
    # Add unresolved scope only; price/unit errors and explicit commercial exclusions stay out.
    pending = df[df.exclusion_reason.eq('scope_pending') & df.price_month_vnd.notna()]
    trimmed = []
    thresholds = []
    for cat, g in base.groupby('category'):
        if len(g) >= 30:
            lo, hi = g.price_month_vnd.quantile([.01, .99])
            keep = g.price_month_vnd.between(lo, hi)
            trimmed.append(g[keep])
            thresholds.append({'category': cat, 'p01_vnd': lo, 'p99_vnd': hi, 'removed': (~keep).sum()})
        else:
            trimmed.append(g)
    # One representative per candidate group is a sensitivity scenario, not a claim of true deduplication.
    candidates = base.sort_values('listing_id').drop_duplicates('split_group')
    sets = {'base': base, 'plus_pending_scope': pd.concat([base, pending]),
            'one_per_candidate_group': candidates, 'trim_p01_p99': pd.concat(trimmed) if trimmed else base,
            'nhatot_only': base[base.source.eq('nhatot')], 'alonhadat_only': base[base.source.eq('alonhadat')]}
    rows = []
    districts = []
    for scenario, g in sets.items():
        for cat, group in g.groupby('category'):
            pair = group.dropna(subset=['area_analysis_m2', 'price_month_vnd'])
            corr = pair.area_analysis_m2.corr(pair.price_month_vnd, method='spearman') if len(pair) >= 10 else np.nan
            rows.append({'scenario': scenario, 'category': cat, 'n': len(group), 'median': group.price_million.median(),
                         'q1': group.price_million.quantile(.25), 'q3': group.price_million.quantile(.75),
                         'area_pairs': len(pair), 'spearman': corr})
        t = summarize(g, ['category', 'district_analysis'])
        t['scenario'] = scenario
        districts.append(t)
    out = pd.DataFrame(rows)
    reference = out[out.scenario.eq('base')].set_index('category')['median']
    out['median_change_pct'] = (out['median'] / out.category.map(reference) - 1) * 100
    return out, pd.DataFrame(thresholds), pd.concat(districts, ignore_index=True), sets


def proposed_split_coverage(price):
    """Feasibility only; this snapshot has already been inspected and is not an untouched test."""
    groups = sorted(price.split_group.unique())
    rng = np.random.default_rng(SEED)
    rng.shuffle(groups)
    n_test = max(1, round(.2*len(groups))) if len(groups) > 1 else 0
    test_groups = set(groups[:n_test])
    x = price.assign(proposed_partition=np.where(price.split_group.isin(test_groups), 'test_candidate', 'train_candidate'))
    assert set(x.loc[x.proposed_partition.eq('test_candidate'), 'split_group']).isdisjoint(
        x.loc[x.proposed_partition.eq('train_candidate'), 'split_group'])
    return x.groupby(['proposed_partition', 'source', 'category']).agg(n=('listing_id', 'size'), groups=('split_group', 'nunique')).reset_index()


def setup():
    sns.set_theme(style='whitegrid', font='DejaVu Sans', context='notebook')
    plt.rcParams.update({'axes.spines.top': False, 'axes.spines.right': False, 'savefig.dpi': 160})


def save(fig, name, note, root):
    fig.text(.01, .01, note, fontsize=9, va='bottom', wrap=True)
    fig.tight_layout(rect=(0, .08, 1, .96))
    dest = Path(root) / 'reports/figures'
    dest.mkdir(parents=True, exist_ok=True)
    path = dest / f'{name}.png'
    fig.savefig(path, bbox_inches='tight')
    plt.close(fig)
    return path


def plot_composition(df, root):
    fig, ax = plt.subplots(1, 2, figsize=(15, 7), gridspec_kw={'width_ratios': [1, 1.5]})
    pd.crosstab(df.category, df.source).reindex(CATEGORIES)[SOURCES].rename(index=LABELS).plot.bar(ax=ax[0], color=[COLORS[s] for s in SOURCES], rot=0)
    ax[0].set(xlabel='', ylabel='Listings', title=f'Source × property-type composition (n={len(df):,})')
    for container in ax[0].containers:
        ax[0].bar_label(container, fontsize=9)
    df.district_analysis.fillna('Unknown').value_counts().sort_index().plot.barh(ax=ax[1], color='#467B74')
    ax[1].set(xlabel='Listings', ylabel='', title='District coverage — full input dataset')
    return save(fig, '01_composition', 'Collected listing counts, not market shares or total market supply.', root)


def plot_missing(df, root):
    table = missing_summary(df)
    table.index = [f'{s} / {LABELS[c]}' for s, c in table.index]
    fig, ax = plt.subplots(figsize=(11, 5))
    sns.heatmap(table, annot=True, fmt='.1f', vmin=0, vmax=100, cmap='YlOrBr', ax=ax, cbar_kws={'label': '% missing'})
    ax.set(title=f'Original-field missingness by source × property type (n={len(df):,})', ylabel='')
    return save(fig, '02_missingness', 'Missingness is calculated within each group; missing values are not replaced with zero.', root)


def plot_distributions(price, value, name, xlabel, root):
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.8))
    for cat, ax in zip(CATEGORIES, axes):
        g = price.loc[price.category.eq(cat)].dropna(subset=[value])
        for source in SOURCES:
            x = g.loc[g.source.eq(source), value]
            if len(x):
                sns.ecdfplot(x=x, ax=ax, color=COLORS[source], label=f'{source} (n={len(x)})')
        ax.set_xscale('log')
        ax.set(title=f'{LABELS[cat]} · n={len(g)}', xlabel=xlabel, ylabel='Cumulative proportion')
        ax.legend(fontsize=8)
    return save(fig, name, 'Eligible residential price set; missing axis values excluded separately. Log scale; distribution tails retained.', root)


def plot_district(price, ppm, root):
    g = price[price.include_ppm2].copy() if ppm else price.copy()
    value = 'price_per_m2' if ppm else 'price_million'
    if ppm:
        g[value] = g[value]/1000
    cats = ['can_ho_chung_cu', 'nha_tro'] if ppm else CATEGORIES
    fig, axes = plt.subplots(1, len(cats), figsize=(6*len(cats), 7), squeeze=False)
    for cat, ax in zip(cats, axes[0]):
        t = summarize(g[g.category.eq(cat) & g.district_analysis.notna()], ['district_analysis'], value).sort_values('district_analysis')
        labels = []
        for i, row in enumerate(t.itertuples()):
            labels.append(f'{row.district_analysis} (n={row.n})')
            if row.n < 10:
                continue
            color = '#2678A6' if row.n >= 30 else '#999999'
            ax.plot([row.q1, row.q3], [i, i], color=color, linewidth=2)
            ax.scatter(row.median, i, color=color, s=35)
        ax.set_yticks(range(len(t)), labels, fontsize=9)
        ax.set(title=LABELS[cat], xlabel='Thousand VND/m²/month' if ppm else 'Million VND/month')
        ax.invert_yaxis()
    return save(fig, '06_ppm2_district' if ppm else '05_price_district',
                'Median and Q1–Q3, not a CI. Blue: n≥30; gray: 10–29; n<10: counts only. Area/source composition is not controlled.', root)


def plot_area_price(price, root):
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    for cat, ax in zip(CATEGORIES, axes):
        g = price[price.category.eq(cat)].dropna(subset=['area_analysis_m2', 'price_million'])
        for source in SOURCES:
            s = g[g.source.eq(source)]
            ax.scatter(s.area_analysis_m2, s.price_million, s=16, alpha=.4, color=COLORS[source], label=f'{source} (n={len(s)})')
        ax.set(xscale='log', yscale='log', xlabel='Area (m², log)', ylabel='Million VND/month (log)', title=f'{LABELS[cat]} · n={len(g)}')
        ax.legend(fontsize=8)
    return save(fig, '07_area_price', 'Price set with usable area. Houses: exploratory because area definitions differ; no causal interpretation.', root)


def plot_controlled(price, feature, name, root, furniture=False):
    g, label = select_controlled(price, feature, nhatot_only=furniture)
    fig, ax = plt.subplots(figsize=(11, 5))
    if len(g):
        order = sorted(g[feature].unique())
        counts = g[feature].value_counts()
        plot_data = g[g[feature].map(counts).ge(10)]
        sns.boxplot(data=plot_data, x=feature, y='price_million', order=order, color='#81B5B0', ax=ax)
        ax.set_xticks(range(len(order)), [f'{x}\nn={counts[x]}' for x in order])
        ax.set(xlabel='Furnishing' if furniture else 'Bedrooms', ylabel='Million VND/month', title=label)
    else:
        ax.text(.5, .5, label, ha='center', va='center', transform=ax.transAxes)
        ax.set_axis_off()
    return save(fig, name, 'Stratum selected by sample size, not price differences. Controls shown in title; n<10: counts only; 10–29: exploratory.', root)


def plot_sources(price, root):
    counts, _, matched = source_comparison(price)
    keys = ['category', 'district_analysis', 'area_band']
    selected = counts[counts.eligible].sort_values(['n_total', *keys], ascending=[False, True, True, True]).head(4)
    nplots = max(1, len(selected))
    fig, axes = plt.subplots(1, nplots, figsize=(5*nplots, 5), squeeze=False)
    if selected.empty:
        axes[0, 0].text(.5, .5, 'No eligible groups with ≥10 listings per source.', ha='center')
    for (_, row), ax in zip(selected.iterrows(), axes[0]):
        m = np.ones(len(matched), dtype=bool)
        for key in keys:
            m &= matched[key].eq(row[key]).to_numpy()
        g = matched[m]
        for source in SOURCES:
            s = g[g.source.eq(source)]
            sns.ecdfplot(data=s, x='price_million', label=f'{source} n={len(s)}', color=COLORS[source], ax=ax)
        ax.set(title=f'{LABELS[row.category]} / {row.district_analysis}\n{row.area_band} m²', xlabel='Million VND/month', ylabel='Cumulative proportion')
        ax.legend(fontsize=8)
    return save(fig, '10_source_comparison', 'Largest strata within category + district + area band. Each source has ≥10 listings; residual property differences remain.', root)


def generate_figures(df, root=ROOT):
    setup()
    price = df[df.include_price]
    return [plot_composition(df, root), plot_missing(df, root),
            plot_distributions(price, 'price_million', '03_price_distribution', 'Million VND/month (log)', root),
            plot_distributions(price, 'area_analysis_m2', '04_area_distribution', 'Area in m² (log)', root),
            plot_district(price, False, root), plot_district(price, True, root), plot_area_price(price, root),
            plot_controlled(price, 'bedroom_group', '08_bedrooms_price', root),
            plot_controlled(price, 'furniture_analysis', '09_furniture_price', root, furniture=True), plot_sources(price, root)]


def write_report(df, figures, root=ROOT):
    price = df[df.include_price].copy()
    counts, source_stats, matched = source_comparison(price)
    sens, thresholds, district_sens, sets = sensitivity_sets(df)
    category_stats = summarize(price, ['category'])
    ppm_stats = summarize(df[df.include_ppm2], ['category'], 'price_per_m2')
    cover = feature_coverage(price)
    split = proposed_split_coverage(price)
    grouped = price.groupby(['source', 'category']).agg(n=('listing_id', 'size'), independent_candidates=('split_group', 'nunique')).reset_index()
    body = ['# EDA — Asking rents in the Hanoi listing sample',
            'Generated from `data/analysis/listings_analysis.csv`. The observation unit is a listing, not a transaction or a verified unique property. These findings describe a fully inspected snapshot, not performance on an untouched holdout. Original listing text, source category codes and place names are preserved.',
            '## 1. Data and scope',
            f'Input: {len(df):,} listings. Posting dates: {df.posted_date.min()} to {df.posted_date.max()}. `scraped_at`: {", ".join(sorted(df.scraped_at.unique()))}; this records parser execution, not a verified collection date.',
            markdown_table(audit_summary(df)),
            'Excluded rows remain in the analysis dataset. The review queue stores flags, decisions and evidence. Codex reviewed titles and CSV fields only; source websites were not verified. Unresolved scope/price cases are excluded from the main price set; unresolved area alone removes area from the relevant calculations. Percentile flags do not automatically exclude a listing.',
            markdown_table(df.groupby(['source','category']).agg(input_n=('listing_id','size'), price_n=('include_price','sum'), ppm2_n=('include_ppm2','sum')).reset_index()),
            '## 2. Key findings']
    composition = pd.crosstab(df.source, df.category).reindex(columns=CATEGORIES)
    shares = composition.div(composition.sum(axis=1), axis=0)*100
    body += [f'**1. Sample composition differs between sources.** Rental rooms account for {shares.loc["nhatot", "nha_tro"]:.1f}% of Nhatot listings and {shares.loc["alonhadat", "nha_tro"]:.1f}% of Alonhadat listings. An aggregate price comparison cannot separate this composition difference. See Figure 01.']
    rawmissing = df.groupby('source').area_m2.agg(lambda x: x.isna().mean()*100)
    body += [f'**2. Area availability differs between sources.** Original area missingness is {rawmissing.loc["nhatot"]:.1f}% on Nhatot and {rawmissing.loc["alonhadat"]:.1f}% on Alonhadat. The primary price/m² set contains {df.include_ppm2.sum():,} listings and excludes private houses; it does not represent the entire price set. See Figure 02 and the area-availability comparison below.']
    lines = [f'{LABELS[r.category]}: {r.median:.2f} million VND/month, Q1–Q3 {r.q1:.2f}–{r.q3:.2f}, n={r.n}' for r in category_stats.itertuples()]
    body += ['**3. Asking rents differ by property type.** ' + '; '.join(lines) + '. These are asking rents in the filtered sample, not transaction prices. See Figure 03.', markdown_table(category_stats)]
    district_example = summarize(price[price.category.eq('can_ho_chung_cu')], ['district_analysis'])
    district_example = district_example[district_example.n.ge(30)].sort_values(['n', 'district_analysis'], ascending=[False, True]).head(2)
    body += ['**4. District comparisons should stay within the same property type.** The two largest apartment groups meeting n≥30 are shown below, selected by sample size rather than price. Read median differences alongside Q1–Q3. Area and source composition still differ, so the gap cannot be attributed to location alone. This is not a market-wide ranking. See Figure 05.', markdown_table(district_example)]
    body += ['**5. The area–price relationship depends on property type.** The table uses valid price/area pairs; missing area may bias the subset. Private-house results remain exploratory because land area and floor area can differ. See Figure 07.', markdown_table(sens[sens.scenario.eq('base')][['category','area_pairs','spearman']])]
    qualified_n = int(counts.eligible.sum())
    body += [f'**6. Matched source comparisons have limited coverage.** There are {len(counts)} strata with usable area and district, {int(counts.both_sources.sum())} with both sources, {qualified_n} with ≥10 listings per source, and {int(counts.main.sum())} with ≥30 per source. The ≥10 groups cover {len(matched)}/{len(price)} listings in the price set; the remainder are outside this comparison. See Figure 10.']
    if qualified_n:
        top = counts[counts.eligible].sort_values(['n_total','category','district_analysis','area_band'], ascending=[False,True,True,True]).iloc[0]
        t = source_stats[(source_stats.category.eq(top.category)) & source_stats.district_analysis.eq(top.district_analysis) & source_stats.area_band.eq(top.area_band)]
        source_med = t.set_index('source')['median']
        pct = (source_med['nhatot']/source_med['alonhadat']-1)*100
        body += [f'Largest matched group, selected by n: **{LABELS[top.category]} / {top.district_analysis} / {top.area_band} m²**. The Nhatot median differs from Alonhadat by {pct:+.1f}%. This is a source-associated difference after partial matching. Furnishing, location and posting practices may still differ; it is not a causal website effect.', markdown_table(t)]
    furniture, stratum = select_controlled(price, 'furniture_analysis', True)
    if len(furniture):
        body += [f'**7. Furnishing comparisons require overlapping groups.** Stratum: {stratum}, n={len(furniture)}. Unknown is kept separate; property quality is not fully controlled. See Figure 09.', markdown_table(summarize(furniture, ['furniture_analysis']))]
    else:
        body += ['**7. Furnishing comparisons lack enough observations even after relaxing district matching, with ≥10 listings per known level.** Do not infer a furnishing premium from pooled data. Figure 09 records this limitation.']
    basic_sens = sens[sens.scenario.isin(['plus_pending_scope','one_per_candidate_group','trim_p01_p99'])]
    max_change = basic_sens.median_change_pct.abs().max() if len(basic_sens) else 0
    body += [f'**8. Stability depends on sample-selection decisions.** Across adding unresolved-scope listings, keeping one listing per candidate duplicate group, and trimming P1/P99, the largest absolute change in a category median is {max_change:.1f}%. Details follow. Stable medians do not guarantee stable distributions or district-level findings.']
    body += ['## 3. Missingness and secondary metrics',
             'Original-field missingness by source × property type (%):', markdown_table(missing_summary(df).reset_index()),
             'Price-set comparison by audited area availability (million VND/month). This describes sample differences, not the missingness mechanism:', markdown_table(area_missing_comparison(price)),
             'Apartment/room asking rent per m² (VND/m²/month); not used to rank private houses:', markdown_table(ppm_stats),
             '## 4. Sensitivity analysis',
             'Each scenario changes one decision. `one_per_candidate_group` retains the smallest listing ID in each candidate group; it does not establish complete deduplication. `plus_pending_scope` adds only unresolved-scope listings with valid prices, not price errors or confirmed out-of-scope commercial listings.',
             markdown_table(sens), 'Monthly-rent P1/P99 thresholds in the main set, VND:', markdown_table(thresholds)]
    district_base = district_sens[district_sens.scenario.eq('base')][['category','district_analysis','n','median']].rename(columns={'n':'base_n','median':'base_median'})
    comparisons = district_sens.merge(district_base, on=['category','district_analysis'])
    comparisons = comparisons[comparisons.n.ge(30) & comparisons.base_n.ge(30) & comparisons.scenario.isin(['plus_pending_scope','one_per_candidate_group','trim_p01_p99'])].copy()
    comparisons['change_pct'] = (comparisons['median']/comparisons.base_median-1)*100
    body += ['Relative district comparisons within category, restricted to cells with ≥30 listings in both base and scenario. Area/source composition is not controlled:', markdown_table(comparisons[['scenario','category','district_analysis','base_n','n','base_median','median','change_pct']])]
    body += ['## 5. Modeling readiness — no model trained',
             '- **Target:** `price_month_vnd`. Consider a log target during validation; convert predictions back to VND/month for evaluation. Do not choose using the test set.',
             '- **Proposed core features:** category, district_analysis and area_analysis_m2, with an area-missing indicator. Bedrooms are optional after coverage checks. Leave toilets out of the baseline because of extensive missingness. Keep Unknown furnishing and compare models with/without furnishing. Use source diagnostically and compare models with/without source.',
             '- **Leakage:** exclude all price fields and transformations from predictors: price_vnd, price_raw, price_million, price_per_m2 and log_price when not used as the target. Exclude title from the baseline; URL/ID are identifiers only. Price-dependent audit/review flags are not predictors either.',
             '- **Missing values:** no EDA imputation. During modeling, use Unknown for categorical features and fit numeric imputation/missing indicators only on training data. Missing rooms/bathrooms do not mean zero.',
             '- **Split:** use split_group to keep candidate duplicates together; seed=42, approximately 80/20 of groups. The table below checks feasibility, not an untouched test set. Title matching may miss reworded reposts and cross-source duplicates.',
             '- **Baseline:** training-set median by category, falling back to the overall training median. Primary metric: MAE in VND/month; secondary: RMSE, with source/category breakdowns. Do not trim test-target tails to improve scores.',
             '- **Final evaluation:** the entire snapshot has been explored; prefer a new unseen snapshot for final evaluation. Holding out one source is an additional transfer check requiring sufficient feature overlap.',
             'Feature coverage in the price set (%):', markdown_table(cover[cover.group.eq('ALL')]),
             'Coverage by source/category (%):', markdown_table(cover.pivot(index='group', columns='feature', values='coverage_pct').reset_index()),
             'Listing counts and candidate independent groups:', markdown_table(grouped),
             'Proposed split distribution (no training/evaluation):', markdown_table(split),
             '## 6. Limitations and unresolved checks',
             f'- {df.review_status.eq("pending").sum()} listings have pending scope/price/area decisions and are excluded from the corresponding calculations; see review_queue.csv. The normal label does not mean verified residential use.',
             f'- {df.quality_flags.str.contains("bedrooms_unit_ambiguous").sum()} listings have invalid bedroom counts or counts that may describe entire buildings; bedrooms_analysis is missing rather than guessed.',
             '- The main scope is whole residential units/rooms, not shared beds or per-person rents. Listings with unclear total room rent remain pending; shared floor area is not automatically used to compute price/m².',
             '- Addresses have not been reconciled across administrative versions. No ward maps or temporal trends are produced. Private-house area and price/m² relationships remain exploratory.',
             '- Raw/interim data are unavailable for recovering records excluded by the earlier pipeline. CSV-only review can miss errors. Do not generalize to the whole market or infer causality.',
             '## 7. Figures']
    for path in figures:
        body += [f'### {path.stem}', f'![{path.stem}](figures/{path.name})']
    text = '\n\n'.join(body) + '\n'
    path = Path(root) / 'reports/eda_report.md'
    path.write_text(text, encoding='utf-8')
    return path
