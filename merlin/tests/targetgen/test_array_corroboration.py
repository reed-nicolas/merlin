"""A tile grid alone is not evidence of a contraction engine."""

from merlin.targetgen.capability_derive import DerivedCapabilities, _from_rtl_facts


def test_unconfirmed_grid_does_not_license_contraction():
    derived = DerivedCapabilities()
    _from_rtl_facts({"facts": {"arrays": [{
        "name": "mesh", "rows": 16, "cols": 16, "instances": 256,
        "container": "Mesh", "element": "Tile", "corroborated": False,
    }]}}, derived)
    assert "contraction" not in derived.supported


def test_confirmed_mac_grid_licenses_contraction():
    derived = DerivedCapabilities()
    _from_rtl_facts({"facts": {"arrays": [{
        "name": "mesh", "rows": 16, "cols": 16, "instances": 256,
        "container": "Mesh", "element": "Tile", "corroborated": True,
        "mac_idiom": {"muls": 1, "adds": 1},
    }]}}, derived)
    assert derived.supported["contraction"].source == "rtl_facts"
