from pathlib import Path

from cfp_spl.feature_sources import load_feature_map, source_identity


def test_feature_source_mapping_aligns_shifted_variant_lines(tmp_path: Path):
    coverage = tmp_path / "coverage"
    coverage.mkdir()
    (coverage / "spectrum_passed_coverage.xml").write_text(
        '<coverage><project><package name="main"><file name="Account.java">'
        '<line num="28" count="1" type="stmt" featureClass="BankAccount.Account" featureLineNum="24" />'
        '</file></package></project></coverage>', encoding="utf-8"
    )
    mapping = load_feature_map(tmp_path)
    assert source_identity("main.Account:update:(I)Z#28#3", mapping) == "BankAccount.Account:24"
