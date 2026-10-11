"""Paper-matching display labels for the internal observable names.

``protocol.OBSERVABLE_NAMES`` (``"Yg100"``, ``"Yg210"``, ``"Ym210"``, ``"Jg10"``,
``"Jg21"``, ``"Jm"``) are the names used internally and in the MATLAB-parity
test suite. The companion manuscript instead calls these
:math:`\\mathcal{Y}_G`, :math:`\\mathcal{Y}_{G2}`, :math:`\\mathcal{Y}_M`,
:math:`\\mathcal{J}_G`, :math:`\\mathcal{J}_{G2}`, :math:`\\mathcal{J}_M`.
This module is the single place that maps one to the other, for use in
plot legends/axis labels and printed tables (demo notebook, reproduce_fig*
scripts) without duplicating the mapping at each call site.
"""

PAPER_LABELS: dict[str, str] = {
    "Yg100": r"$\mathcal{Y}_G$",
    "Yg210": r"$\mathcal{Y}_{G2}$",
    "Ym210": r"$\mathcal{Y}_M$",
    "Jg10": r"$\mathcal{J}_G$",
    "Jg21": r"$\mathcal{J}_{G2}$",
    "Jm": r"$\mathcal{J}_M$",
}

__all__ = ["PAPER_LABELS"]
