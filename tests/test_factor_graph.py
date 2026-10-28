import base64

from cfp_spl.factor_graph import decode, stable_statement_key


def test_java_base64_without_padding_is_accepted():
    raw = "main.Account:update:(I)Z#25#8"
    encoded = base64.urlsafe_b64encode(raw.encode()).decode().rstrip("=")
    assert decode(encoded) == raw
    assert stable_statement_key(raw) == raw
