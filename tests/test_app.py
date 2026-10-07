"""End-to-end runs of both pages with Streamlit's AppTest."""
import logging
from pathlib import Path

from streamlit.testing.v1 import AppTest

ROOT = Path(__file__).resolve().parents[1]
ANALYSE = str(ROOT / "views" / "analyse.py")
DOCS = str(ROOT / "views" / "docs.py")
TIMEOUT = 180


def _text(at):
    return "\n".join(m.value for m in at.markdown)


def test_default_example_runs():
    at = AppTest.from_file(ANALYSE, default_timeout=TIMEOUT).run()
    assert not at.exception, at.exception
    text = _text(at)
    assert "Hyperbolic semigroup" in text and "repelling boundary fixed point" in text
    assert "Generator criterion" in text
    assert len(at.tabs) == 5


def test_library_query_param():
    at = AppTest.from_file(ANALYSE, default_timeout=TIMEOUT)
    at.query_params["lib"] = "parabolic-positive-step"
    at.run()
    assert not at.exception
    assert "Parabolic semigroup" in _text(at) and "positive hyperbolic step" in _text(at)


def test_typed_input_and_analyse_button():
    at = AppTest.from_file(ANALYSE, default_timeout=TIMEOUT).run()
    at.text_input(key="in_G").set_value("-z*(1-z)/(1+z)")
    at.button[0].click().run()  # the form's Analyse button
    assert not at.exception, at.exception
    assert "Elliptic semigroup" in _text(at)
    assert at.query_params["G"] in ("-z*(1-z)/(1+z)", ["-z*(1-z)/(1+z)"])


def test_berkson_porta_mode():
    at = AppTest.from_file(ANALYSE, default_timeout=TIMEOUT).run()
    at.radio(key="in_mode").set_value("bp").run()
    at.text_input(key="in_tau").set_value("1/2")
    at.text_input(key="in_p").set_value("(1+z)/(1-z)")
    at.button[0].click().run()
    assert not at.exception, at.exception
    text = _text(at)
    assert "Elliptic semigroup" in text and "\\frac{9}{4}" in text


def test_invalid_input_shows_error():
    at = AppTest.from_file(ANALYSE, default_timeout=TIMEOUT)
    at.query_params["mode"] = "generator"
    at.query_params["G"] = "1/(z-1/2)"
    at.run()
    assert not at.exception
    assert at.error and "pole inside the disc" in at.error[0].value


def test_point_inspector():
    at = AppTest.from_file(ANALYSE, default_timeout=TIMEOUT).run()
    at.text_input(key="insp_text").set_value("-0.3").run()
    assert not at.exception, at.exception
    assert "converges to a repelling boundary fixed point" in _text(at)


def test_docs_page():
    at = AppTest.from_file(DOCS, default_timeout=TIMEOUT).run()
    assert not at.exception, at.exception
    assert len(at.header) == 16
    assert len(at.expander) >= 30
    assert any("find_tau" in e.label for e in at.expander)


class _Collect(logging.Handler):
    def __init__(self):
        super().__init__()
        self.messages = []

    def emit(self, record):
        self.messages.append(record.getMessage())


def test_no_deprecation_warnings():
    """Streamlit logs every deprecation through this logger, whatever the version."""
    handler = _Collect()
    logger = logging.getLogger("streamlit.deprecation_util")
    logger.addHandler(handler)
    try:
        at = AppTest.from_file(ANALYSE, default_timeout=TIMEOUT).run()
        at.text_input(key="insp_text").set_value("-0.3").run()       # draws the inspector chart
        at.button[0].click().run()                                  # the form button
        AppTest.from_file(DOCS, default_timeout=TIMEOUT).run()      # docs page and its script
    finally:
        logger.removeHandler(handler)
    assert not handler.messages, handler.messages
