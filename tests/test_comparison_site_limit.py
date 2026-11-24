from types import SimpleNamespace
from unittest.mock import patch
from lake_workbench.models import preparation as p


def test_only_first_ten_checked_and_all_sites_retained():
    directory = [dict(site_id=f's{i}', display_name=f'Site {i}') for i in range(25)]
    class Catalog:
        region = SimpleNamespace(key='test-limit')
        @property
        def sites(self):
            raise AssertionError('must not iterate full catalog')
        def get_site(self, key):
            return SimpleNamespace(site_id=key)
    def inspect(catalog, site, workspace):
        return dict(site_id=site.site_id, display_name=site.site_id, rank=0,
                    seed_cached=True, slope_cached=True, imagery_available=True, cache_checked=True)
    p._SITE_CACHE.clear()
    with patch.object(p, '_comparison_directory', return_value=directory), patch.object(p, '_comparison_site', side_effect=inspect) as check:
        events=list(p.iter_comparison_sites(Catalog(), 'test'))
        assert check.call_count == 10
        items={item['site_id']:item for event in events for item in event.get('items', [])}
        assert len(items)==25
        assert sum(item['cache_checked'] for item in items.values())==10
        assert events[-1]['completed']==10 and events[-1]['total']==25
        cached=list(p.iter_comparison_sites(Catalog(), 'test'))
        assert cached[-1]['completed']==10
        assert len(cached[0].get('items', []))==0
        assert check.call_count==10
    p._SITE_CACHE.clear()
