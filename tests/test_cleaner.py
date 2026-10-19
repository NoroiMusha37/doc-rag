import pytest

from src.doc_rag.ingestion.cleaner import clean_formulas, clean_text


def test_soft_hyphens_and_the_space_after_them_are_removed() -> None:
    assert clean_text("the ex\xad piration date and con\xadditioning") == (
        "the expiration date and conditioning"
    )


def test_a_sound_formula_is_left_alone() -> None:
    text = "See $$\\frac { 1 } { 2 } \\left( x \\right) \\rightarrow y$$ here."

    assert clean_formulas(text) == text


def test_column_marks_without_an_environment_get_one() -> None:
    assert clean_formulas("$$x & = 1 \\\\ y & = 2$$") == (
        "$$\\begin{aligned}x & = 1 \\\\ y & = 2\\end{aligned}$$"
    )


def test_an_escaped_ampersand_is_not_a_column_mark() -> None:
    assert clean_formulas("$$a \\& b$$") == "$$a \\& b$$"


@pytest.mark.parametrize(
    "latex",
    [
        pytest.param("\\frac { 1 { 2 }", id="unclosed-brace"),
        pytest.param("a } b", id="extra-closing-brace"),
        pytest.param("\\left( x", id="left-without-right"),
        pytest.param("\\begin{cases} x", id="begin-without-end"),
        pytest.param("\\begin{a} x \\end{b}", id="mismatched-environment"),
        pytest.param("x \\intertext{then} y", id="intertext"),
    ],
)
def test_a_formula_that_cannot_be_repaired_is_kept_as_code(latex: str) -> None:
    assert clean_formulas(f"Before $${latex}$$ after") == f"Before `{latex}` after"
