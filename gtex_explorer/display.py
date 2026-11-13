"""Names and number formats shared by every page.

Numbers follow Indonesian convention: a dot between thousands and a decimal comma.
A missing value (None or NaN) is written as `missing`.
"""

import pandas as pd

METHODS = {
    "l1": "Regresi logistik L1",
    "anova": "ANOVA",
    "mutual_information": "Mutual Information",
    "baseline": "Baseline - tanpa seleksi fitur",
}
# What a selector's gene score is, for the score column of a gene panel.
SCORE_LABELS = {
    "anova": "Skor F (ANOVA)",
    "mutual_information": "Skor MI (nat)",
    "l1": "Koefisien absolut maksimum",
}
# Colours for series that must be told apart (methods, model letters), always taken in this order.
# Validated as a categorical palette on the white page: adjacent pairs stay apart for normal and
# colour-blind vision. The last three are under 3:1 against white, so a chart using them has the
# same numbers in a table beside it.
SERIES_COLORS = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4"]
METRICS = {
    "macro_f1": "Macro-F1",
    "macro_precision": "Macro-precision",
    "macro_recall": "Macro-recall",
    "balanced_accuracy": "Balanced accuracy",
    "accuracy": "Accuracy",
}
# A training run's status, as (symbol, name): the run picker shows both, Status proses the name.
RUN_STATUS = {
    "queued": ("○", "Menunggu worker"),
    "running": ("●", "Berjalan"),
    "paused": ("‖", "Dijeda"),
    "cancelling": ("■", "Menghentikan"),
    "cancelled": ("■", "Dihentikan"),
    "failed": ("✕", "Gagal"),
    "completed": ("✓", "Selesai"),
}


def plain(text) -> str:
    """Text typed by a user or taken from a file name, shown as itself where Streamlit reads Markdown."""
    return "".join("\\" + character if character in "\\`*_[]<>~|#:$" else character for character in str(text))


def _missing(value) -> bool:
    """Whether a value is absent: None, NaN, or pandas' NA."""
    return value is None or bool(pd.isna(value))


def count(value, missing="-") -> str:
    """A whole number with a dot between thousands: 5.000."""
    return missing if _missing(value) else f"{int(value):,}".replace(",", ".")


def decimal(value, digits=None, missing="-") -> str:
    """A number with a decimal comma: 0,05. Without `digits`, as short as possible."""
    if _missing(value):
        return missing
    text = f"{float(value):g}" if digits is None else f"{float(value):.{digits}f}"
    return text.replace(".", ",")


def percent(value, digits=2, missing="-") -> str:
    """A share as a percentage with a decimal comma: 0.9625 -> 96,25%."""
    return missing if _missing(value) else f"{float(value):.{digits}%}".replace(".", ",")


def points(value, digits=2, missing="-", sign=False) -> str:
    """A difference or SD of shares in percentage points: 0.0049 -> 0,49, or +0,49 with `sign`."""
    if _missing(value):
        return missing
    return format(float(value) * 100, ("+" if sign else "") + f".{digits}f").replace(".", ",")


def method_name(selector, *, l1_c=None, n_neighbors=None) -> str:
    """Name a feature selection method with its own setting, the same on every page."""
    text = METHODS.get(selector, str(selector))
    if selector == "l1" and not _missing(l1_c):
        text += f" - C = {decimal(l1_c)}"
    elif selector == "mutual_information" and not _missing(n_neighbors):
        text += f" - n_neighbors = {int(n_neighbors)}"
    return text
