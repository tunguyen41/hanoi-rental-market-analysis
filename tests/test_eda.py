"""Regression checks for analytical errors that can change conclusions."""
from pathlib import Path
import tempfile
import unittest

import numpy as np
import pandas as pd

from src.eda_audit import parse_monthly_price, prepare_analysis, read_reviews, run_audit, DECISIONS
from src.eda_reporting import source_comparison, sensitivity_sets, proposed_split_coverage


def row(**kwargs):
    data = dict(source='nhatot', listing_id='nt_1', category='can_ho_chung_cu', url='https://example.test/1',
                title='Cho thuê căn hộ đầy đủ nội thất Cầu Giấy', price_vnd=5_000_000,
                price_raw='5 triệu/tháng', area_m2=30., area_raw=30., bedrooms=1., toilets=np.nan,
                furniture=np.nan, city='Hà Nội', district='Cầu Giấy', ward='Dịch Vọng',
                location_raw='Dịch Vọng - Cầu Giấy', posted_date='2026-09-18', scraped_at='2026-09-19')
    data.update(kwargs)
    return data


class PriceUnitTests(unittest.TestCase):
    def test_per_m2_already_total_not_multiplied_again(self):
        self.assertEqual(parse_monthly_price('Giá: 130 ngàn /\xa0m² / tháng', 26_000_000, 200),
                         (26_000_000., 'per_m2_total_verified'))

    def test_per_m2_unit_price_converted_only_once(self):
        self.assertEqual(parse_monthly_price('130 ngàn /m2/tháng', 130_000, 200),
                         (26_000_000., 'per_m2_converted'))

    def test_vietnamese_numbers_and_unknown_units(self):
        self.assertEqual(parse_monthly_price('5,48 triệu/tháng', 5_480_000, 30)[0], 5_480_000)
        self.assertEqual(parse_monthly_price('500.000 đ/tháng', 500_000, 30)[0], 500_000)
        for raw, value in [('5 triệu/tháng', 50_000_000), ('100 ngàn/ngày', 100_000), ('Thỏa thuận', 100_000), ('5 triệu/tháng', -1)]:
            self.assertTrue(np.isnan(parse_monthly_price(raw, value, 30)[0]))


class AnalysisPolicyTests(unittest.TestCase):
    def test_missing_features_do_not_remove_monthly_price(self):
        raw = pd.DataFrame([row(area_m2=np.nan, area_raw=np.nan)])
        result = prepare_analysis(raw)
        self.assertTrue(result.include_price.iloc[0])
        self.assertFalse(result.include_ppm2.iloc[0])
        self.assertEqual(result.furniture_analysis.iloc[0], 'Unknown')
        pd.testing.assert_frame_equal(raw, result[raw.columns])

    def test_house_not_in_primary_ppm2_and_room_building_count_not_feature(self):
        df = prepare_analysis(pd.DataFrame([row(category='nha_rieng'),
            row(listing_id='nt_2', url='https://example.test/2', category='nha_tro', bedrooms=18)]))
        self.assertFalse(df.include_ppm2.iloc[0])
        self.assertTrue(df.include_price.iloc[1])
        self.assertTrue(pd.isna(df.bedrooms_analysis.iloc[1]))
        self.assertEqual(df.bedrooms.iloc[1], 18)

    def test_short_term_and_shared_rooms_not_monthly_whole_unit(self):
        df = prepare_analysis(pd.DataFrame([row(title='Lưu trú 100k/ngày'),
            row(listing_id='nt_2', url='https://example.test/2', title='Tìm người ở ghép')]))
        self.assertFalse(df.include_price.any())
        self.assertTrue(pd.isna(df.price_month_vnd.iloc[0]))
        self.assertEqual(df.scope_flag.iloc[1], 'suspicious')

    def test_duplicate_candidates_independent_of_price_and_source(self):
        raw = pd.DataFrame([row(), row(source='alonhadat', listing_id='alo_2', url='https://example.test/2',
                                      price_vnd=6_000_000, price_raw='6 triệu/tháng')])
        df = prepare_analysis(raw)
        self.assertEqual(df.split_group.nunique(), 1)
        self.assertTrue(df.duplicate_group_id.ne('').all())
        scenarios = sensitivity_sets(df)[3]
        self.assertEqual(len(scenarios['base']), 2)
        self.assertEqual(len(scenarios['one_per_candidate_group']), 1)
        self.assertEqual(proposed_split_coverage(df).groups.sum(), 2)  # one group represented in two source cells
        # Both sources must stay in the same partition even though the group crosses sources.
        self.assertEqual(proposed_split_coverage(df).proposed_partition.nunique(), 1)

    def test_pending_scope_sensitivity_does_not_add_bad_prices(self):
        df = prepare_analysis(pd.DataFrame([row(), row(listing_id='nt_2', url='https://example.test/2', title='Cho thuê văn phòng'),
            row(listing_id='nt_3', url='https://example.test/3', title='Cho thuê văn phòng theo ngày')]))
        scenarios = sensitivity_sets(df)[3]
        self.assertEqual(set(scenarios['base'].listing_id), {'nt_1'})
        self.assertEqual(set(scenarios['plus_pending_scope'].listing_id), {'nt_1', 'nt_2'})

    def test_source_matching_requires_each_source_not_total_count(self):
        raw = [row(listing_id=f'nt_{i}', url=f'https://example.test/{i}') for i in range(20)]
        raw += [row(source='alonhadat', listing_id='alo_1', url='https://example.test/alo1')]
        strata, stats, matched = source_comparison(prepare_analysis(pd.DataFrame(raw)))
        self.assertTrue(strata.both_sources.all())
        self.assertFalse(strata.eligible.any())
        self.assertTrue(matched.empty)

    def test_review_persistence_and_stale_context_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root/'data/processed').mkdir(parents=True)
            original = pd.DataFrame([row(title='Căn hộ trong tòa nhà')])
            path = root/'data/processed/listings.csv'
            original.to_csv(path, index=False)
            original_bytes = path.read_bytes()
            first = run_audit(root)
            self.assertFalse(first.include_price.iloc[0])
            queue = root/'data/analysis/review_queue.csv'
            q = pd.read_csv(queue, keep_default_na=False)
            q.loc[0, ['scope_decision','review_note','reviewer']] = ['keep','Apartment within a building, not a whole-building lease','test reviewer']
            q.to_csv(queue, index=False)
            second = run_audit(root)
            self.assertTrue(second.include_price.iloc[0])
            output = (root/'data/analysis/listings_analysis.csv').read_bytes()
            run_audit(root)
            self.assertEqual(output, (root/'data/analysis/listings_analysis.csv').read_bytes())
            self.assertEqual(path.read_bytes(), original_bytes)
            changed = original.copy()
            changed.loc[0,'price_raw'] = '50 triệu/tháng'
            with self.assertRaisesRegex(ValueError, 'no longer matches'):
                read_reviews(queue, changed)


if __name__ == '__main__':
    unittest.main()
