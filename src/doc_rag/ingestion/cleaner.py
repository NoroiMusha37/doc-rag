"""Fixes the damage that PDF text layers and the formula model leave in chunk text.

Everything stays in the text: a formula that cannot be repaired is kept as literal code
instead of being dropped, so nothing the document said is lost.
"""

import re

DISPLAY_FORMULA = re.compile(r"\$\$(.+?)\$\$", re.DOTALL)
# A soft hyphen marks where a word may break; scans leave it with a space after it.
SOFT_HYPHEN = re.compile("\xad\\s*")
# A command name ends at the first non-letter: `\left(` counts, `\leftarrow` does not.
LEFT = re.compile(r"\\left(?![a-zA-Z])")
RIGHT = re.compile(r"\\right(?![a-zA-Z])")
ALIGN_MARK = re.compile(r"(?<!\\)&")
ENVIRONMENT = re.compile(r"\\(begin|end)\{([^}]*)\}")


def clean_text(text: str) -> str:
    return clean_formulas(SOFT_HYPHEN.sub("", text))


def clean_formulas(text: str) -> str:
    """Repair each `$$...$$` formula; one that cannot be repaired becomes inline code."""

    def clean(match: re.Match[str]) -> str:
        latex = match.group(1)
        # The formula model writes column marks but leaves out the environment they need.
        if ALIGN_MARK.search(latex) and "\\begin" not in latex:
            latex = f"\\begin{{aligned}}{latex}\\end{{aligned}}"
        if _is_valid(latex):
            return f"$${latex}$$"
        return f"`{' '.join(match.group(1).split()).replace('`', "'")}`"

    return DISPLAY_FORMULA.sub(clean, text)


def _is_valid(latex: str) -> bool:
    if "\\intertext" in latex:
        return False
    depth = 0
    for character in latex.replace("\\{", "").replace("\\}", ""):
        depth += (character == "{") - (character == "}")
        if depth < 0:
            return False
    open_environments: list[str] = []
    for kind, name in ENVIRONMENT.findall(latex):
        if kind == "begin":
            open_environments.append(name)
        elif not open_environments or open_environments.pop() != name:
            return False
    return (
        depth == 0
        and not open_environments
        and len(LEFT.findall(latex)) == len(RIGHT.findall(latex))
    )
