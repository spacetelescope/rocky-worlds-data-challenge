# Licensed under a 3-clause BSD style license - see LICENSE.rst

try:
    from .version import version as __version__
except ImportError:
    __version__ = ''

# Expose subpackage API at package level.
from .eclipsing_system import *  # noqa
from .results import *  # noqa


def interactive_form(form_path=None, theme='auto'):
    """Render the interactive Solara submission-form interface.

    Parameters
    ----------
    form_path : path-like, optional
        Existing form JSON file to load when the widget opens.
    theme : {'auto', 'light', 'dark'}, optional
        Visual theme for the widget. ``'auto'`` follows the notebook frontend
        where possible.

    Returns
    -------
    solara component
        Rendered interactive submission-form widget.

    Raises
    ------
    ValueError
        If ``theme`` is not one of ``'auto'``, ``'light'``, or ``'dark'``.
    """
    from .widgets import interactive_form as _interactive_form
    return _interactive_form(form_path=form_path, theme=theme)
