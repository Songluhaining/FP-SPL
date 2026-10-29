from pathlib import Path

from cfp_spl.spectrum import load_spectrum_priors


def _write(path: Path, tests: int, failed_count: int, passed_count: int) -> None:
    count = failed_count if "failed" in path.name else passed_count
    path.write_text(
        f'''<coverage><tests count="{tests}"/><project><package name="main">
        <file name="Account"><line num="7" count="{count}" type="stmt"
        featureClass="BankAccount.Account" featureLineNum="7"/></file>
        </package></project></coverage>''', encoding="utf-8"
    )


def test_spectrum_creates_non_uniform_statement_prior(tmp_path: Path):
    coverage = tmp_path / "coverage"
    coverage.mkdir()
    _write(coverage / "spectrum_failed_coverage.xml", 4, 3, 0)
    _write(coverage / "spectrum_passed_coverage.xml", 10, 0, 1)
    values = load_spectrum_priors(tmp_path)
    evidence = values["BankAccount.Account:7"]
    assert evidence.failed_covered == 3
    assert evidence.passed_covered == 1
    assert evidence.error_prior > 0.5
    assert 0.0 <= evidence.suspiciousness <= 1.0
