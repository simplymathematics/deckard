import deckard.plugins as plugins


def test_plugins_namespace_exports_selected_families():
    assert "anjana" in plugins.__all__
    assert "fairlearn" in plugins.__all__
    assert "lifelines" in plugins.__all__
    assert "yellowbrick" in plugins.__all__
