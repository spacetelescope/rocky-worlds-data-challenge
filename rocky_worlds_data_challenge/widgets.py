import html
import json
import os
import re
import tempfile
from copy import deepcopy
from functools import partial
from importlib import resources
from pathlib import Path

import solara

from .results import Form, empty_form_path

errors = solara.reactive('')
tab_index = solara.reactive(0)
filename = solara.reactive('form_GJ3929b.json')
loaded_source = solara.reactive(None)


def _load_form_style():
    """Load the CSS used to style the interactive submission form.

    Returns
    -------
    str
        Contents of the packaged interactive-form stylesheet.
    """
    return resources.files(
        'rocky_worlds_data_challenge.styles'
    ).joinpath('interactive_form.css').read_text()


FORM_STYLE = _load_form_style()


def _load_blank_form():
    """Load the packaged blank form definition.

    Returns
    -------
    dict
        Blank form dictionary loaded from the package data directory.
    """
    with open(empty_form_path, 'r') as form_file:
        return json.load(form_file)


BLANK_FORM = _load_blank_form()
form_dict = solara.reactive(deepcopy(BLANK_FORM))

PRIOR_PARAMETER_EXAMPLES = {
    'fixed': 'value=0.03',
    'uniform': 'lower=-500 ppm, upper=500 ppm',
    'normal': 'mean=0.3, sigma=0.05',
    'log-normal': (
        'mean=1.0, sigma=0.2 in natural-log space; use sigma, not variance'
    ),
    'truncated normal': 'mean=0.3, sigma=0.05, lower=0, upper=1',
    'log-uniform': 'lower=1e-3, upper=1',
    'custom / other': 'Describe the prior and all parameter values',
    'not applicable': 'not applicable',
}

PRIOR_PARAMETER_PLACEHOLDERS = {
    '19': 'e.g., depth_ecl (ppm)',
    '21': 'e.g., b_tra, inc (degrees), or b_ecl',
    '22': 'e.g., rprs',
    '23': 'e.g., t_tra (BMJD_TDB), or t_ecl (BMJD_TDB)',
    '24': 'e.g., ecc, ecosw, or secosw',
    '25': 'e.g., omega (degrees), esinw, or sesinw',
    '26': 'e.g., per (days), or a (a/Rs)',
}

SIMPLE_TEXT_QUESTION_NUMBERS = {
    '02',  # Group name
    '03',  # Submitter name
    '04',  # Submitter email
    '05',  # Secondary contact
    '06',  # Secondary contact email
    '07',  # Collaboration members
    '08',  # JWST pipeline version
    '13',  # Photometric extraction method
    '17',  # Uncertainty-scaling parameter names
    '18',  # Eclipse model software
    '30',  # Detrending vector summary
    '34',  # Posterior sampler
    '35',  # Posterior sampling reference
}


def _format_inline_code(text):
    """Escape text and convert small Markdown snippets to HTML.

    Parameters
    ----------
    text : object
        Text-like value to format.

    Returns
    -------
    str
        HTML-safe string with backtick-delimited snippets rendered as
        ``<code>`` elements and Markdown links rendered as anchors.
    """
    escaped = html.escape(str(text))
    escaped = re.sub(
        r'\[([^\]]+)\]\((https?://[^)]+)\)',
        r'<a href="\2" target="_blank" rel="noopener noreferrer">\1</a>',
        escaped,
    )
    return re.sub(r'`([^`]+)`', r'<code>\1</code>', escaped)


def _format_example(example):
    """Format a form example for display in the Solara widget.

    Parameters
    ----------
    example : object
        Example value from the form schema.

    Returns
    -------
    str
        HTML-safe example string.
    """
    if isinstance(example, dict):
        parts = [
            f"{key}={value}"
            for key, value in example.items()
        ]
        return _format_inline_code('; '.join(parts))

    return _format_inline_code(example)


def _placeholder_from_example(fields, default=''):
    """Return plain placeholder text from a question example.

    Parameters
    ----------
    fields : dict
        Form question metadata.
    default : str
        Fallback placeholder if the example is empty.

    Returns
    -------
    str
        Plain-text placeholder suitable for input widgets.
    """
    example = fields.get('example', '')
    if isinstance(example, dict):
        parts = [
            f"{key}={value}"
            for key, value in example.items()
        ]
        example = '; '.join(parts)

    example = str(example).strip()
    if not example:
        return default

    return example


def _prior_response(question_number):
    """Return a normalized prior-specification response dictionary.

    Parameters
    ----------
    question_number : str
        Question key in the current form dictionary.

    Returns
    -------
    dict
        Dictionary with ``parameter``, ``prior_distribution_type``, and
        ``prior_parameters`` keys.
    """
    response = form_dict.value[question_number].get('response', {})

    if not isinstance(response, dict):
        response = {}

    return {
        'parameter': response.get('parameter', ''),
        'prior_distribution_type': response.get(
            'prior_distribution_type',
            '',
        ),
        'prior_parameters': response.get('prior_parameters', ''),
    }


def _set_prior_response(question_number, response_key, value):
    """Update one field of a prior-specification response.

    Parameters
    ----------
    question_number : str
        Question key in the current form dictionary.
    response_key : str
        Prior-response field to update.
    value : object
        New field value.
    """
    updated_form = deepcopy(form_dict.value)
    response = updated_form[question_number].get('response', {})

    if not isinstance(response, dict):
        response = {}

    response.setdefault('parameter', '')
    response.setdefault('prior_distribution_type', '')
    response.setdefault('prior_parameters', '')
    response[response_key] = value

    updated_form[question_number]['response'] = response
    form_dict.set(updated_form)
    errors.set('')


@solara.component
def question_widgets(selected):
    """Render response controls for every question in the loaded form.

    Parameters
    ----------
    selected : solara.Reactive
        Reactive selected-path state used to display the loaded source.
    """

    solara.HTML(
        unsafe_innerHTML=(
            "<div class='rwddt-loaded-path'>"
            f"Responses loaded from: {_format_inline_code(selected.value)}"
            "</div>"
        )
    )

    if not form_dict.value:
        solara.Text("No form loaded yet.")

    for question_number, fields in form_dict.value.items():
        required = (
            '<span class="rwddt-required">(required)</span>'
            if fields['required'] else ''
        )
        solara.HTML(
            unsafe_innerHTML=(
                "<div class='rwddt-question-prompt'>"
                f"{int(question_number)}. {fields['prompt']} {required}"
                "</div>"
            )
        )
        description = (
            "<div class='rwddt-helper-line'>"
            "<span class='rwddt-helper-label'>Description:</span> "
            f"{_format_inline_code(fields['description'])}"
            "</div>"
            if len(fields['description']) else ''
        )
        solara.HTML(
            unsafe_innerHTML=(
                "<div class='rwddt-question-details'>"
                f"{description}"
                "<div class='rwddt-helper-line'>"
                "<span class='rwddt-helper-label'>"
                f"Example ({html.escape(fields['format'])}):"
                "</span> "
                f"{_format_example(fields['example'])}"
                "</div>"
                "</div>"
            )
        )

        def on_value(x, question_number):
            """Store one form response and clear any validation message.

            Parameters
            ----------
            x : object
                New response value.
            question_number : str
                Question key to update.
            """
            form_dict.value[question_number]['response'] = x
            errors.set('')

        if fields['format'] == 'select':
            solara.Select(
                label='Select an option',
                value=form_dict.value[question_number]['response'],
                values=fields['options'],
                on_value=partial(on_value, question_number=question_number),
                classes=['rwddt-select'],
            )
        elif fields['format'] == 'prior_spec':
            response = _prior_response(question_number)
            prior_type = response['prior_distribution_type']
            parameter_placeholder = PRIOR_PARAMETER_PLACEHOLDERS.get(
                question_number,
                'e.g., model_parameter_name',
            )
            prior_parameters_placeholder = PRIOR_PARAMETER_EXAMPLES.get(
                prior_type,
                (
                    'Use named values: value=..., lower=..., upper=..., '
                    'mean=..., sigma=...'
                ),
            )

            solara.InputText(
                label='Parameter',
                placeholder=parameter_placeholder,
                value=response['parameter'],
                on_value=partial(
                    _set_prior_response,
                    question_number,
                    'parameter',
                ),
                continuous_update=True,
            )
            solara.Select(
                label='Prior distribution type',
                value=prior_type,
                values=fields['options'],
                on_value=partial(
                    _set_prior_response,
                    question_number,
                    'prior_distribution_type',
                ),
                classes=['rwddt-select'],
            )
            solara.InputText(
                label='Prior parameters',
                placeholder=prior_parameters_placeholder,
                value=response['prior_parameters'],
                on_value=partial(
                    _set_prior_response,
                    question_number,
                    'prior_parameters',
                ),
                continuous_update=True,
            )
            solara.HTML(
                unsafe_innerHTML=(
                    "<div class='rwddt-prior-hint'>"
                    "For Normal and log-normal priors, report "
                    "<code>sigma</code> / standard deviation, not variance. "
                    "Use named values where possible."
                    "</div>"
                )
            )
        elif (fields['format'] == 'str' and
              question_number in SIMPLE_TEXT_QUESTION_NUMBERS):
            solara.InputText(
                label='',
                placeholder=_placeholder_from_example(fields),
                value=form_dict.value[question_number]['response'],
                on_value=partial(on_value, question_number=question_number),
                continuous_update=True,
            )
        elif fields['format'] == 'str':
            solara.InputTextArea(
                label='',
                placeholder=_placeholder_from_example(fields),
                value=form_dict.value[question_number]['response'],
                on_value=partial(on_value, question_number=question_number),
                continuous_update=True,
                rows=3,
            )
        elif fields['format'] == 'y/n':
            solara.ToggleButtonsSingle(
                value=form_dict.value[question_number]['response'],
                values='y n'.split(),
                on_value=partial(on_value, question_number=question_number)
            )
        elif fields['format'] == 'float/int':
            solara.InputFloat(
                label='',
                placeholder=_placeholder_from_example(fields),
                value=form_dict.value[question_number]['response'],
                optional=not fields['required'],
                on_value=partial(on_value, question_number=question_number),
                continuous_update=True
            )
        elif fields['format'] == 'list[float/int]':
            solara.InputText(
                label='',
                placeholder=_placeholder_from_example(
                    fields,
                    default='comma-separated values',
                ),
                value=form_dict.value[question_number]['response'],
                on_value=partial(on_value, question_number=question_number),
                continuous_update=True,
            )
        else:
            solara.InputTextArea(
                label='',
                placeholder=_placeholder_from_example(fields),
                value=form_dict.value[question_number]['response'],
                on_value=partial(on_value, question_number=question_number),
                continuous_update=True,
                rows=3,
            )

    solara.HTML(
        unsafe_innerHTML=(
            "<div class='rwddt-form-footer-note'>"
            "When you have finished filling out the form, scroll back to the "
            "top and use the <strong>Validate</strong> tab to check your "
            "responses, then use the <strong>Export</strong> tab to save the "
            "completed JSON form."
            "</div>"
        )
    )


@solara.component
def loading_form():
    """Render the loading indicator shown before the tabs are ready."""
    with solara.Column(
        align='center',
        classes=['rwddt-loading-panel'],
    ):
        solara.SpinnerSolara()
        solara.Text("Loading submission form...")


@solara.component
def interactive_form(form_path=None, theme='auto'):
    """Render the interactive Solara submission-form interface.

    Parameters
    ----------
    form_path : path-like, optional
        Existing form JSON file to load when the widget opens.
    theme : {'auto', 'light', 'dark'}, optional
        Visual theme for the widget.

    Raises
    ------
    ValueError
        If ``theme`` is not one of ``'auto'``, ``'light'``, or ``'dark'``.
    """
    solara.Style(FORM_STYLE)
    render_ready, set_render_ready = solara.use_state(False)

    if theme not in ['auto', 'light', 'dark']:
        raise ValueError("theme must be one of: 'auto', 'light', 'dark'.")

    def mark_ready():
        """Switch from the spinner to the fully rendered tab view."""
        set_render_ready(True)

    solara.use_effect(mark_ready, [])

    if form_path is None:
        directory = solara.reactive(Path(os.getcwd()))
        selected = solara.reactive(None)
    else:
        directory = solara.reactive(Path(os.path.dirname(form_path)))
        selected = solara.reactive(Path(form_path))

    def file_browser_filter(path):
        """Allow directories and JSON files in the form file browser.

        Parameters
        ----------
        path : path-like
            Candidate path from the Solara file browser.

        Returns
        -------
        bool
            `True` when ``path`` is a directory or JSON file.
        """
        file_extension = str(path).split('.')[-1]
        if os.path.isdir(path):
            return True

        if (os.path.exists(path) and os.path.isfile(path) and
                file_extension.lower() == 'json'):
            return True
        return False

    def on_path_select(path):
        """Update the selected JSON file path and containing directory.

        Parameters
        ----------
        path : path-like
            Selected JSON path from the file browser.
        """
        directory.set(Path(os.path.dirname(path)))
        selected.set(Path(path))

    def on_load_json():
        """Load the selected JSON file into the form state."""
        with open(selected.value, 'r') as form_file:
            form_dict.set(json.load(form_file))
        loaded_source.set(Path(selected.value))
        tab_index.set(1)

    with solara.Column(
        classes=['rwddt-form', f'rwddt-form--{theme}'],
        gap='0px',
        style=(
            'width: 100%; '
            'background: var(--rwddt-bg); '
            'color: var(--rwddt-text);'
        )
    ):
        with solara.Column(
            align='center',
            gap='0px',
            style=(
                'background: var(--rwddt-panel-bg); '
                'color: var(--rwddt-text);'
            )
        ):
            solara.HTML(
                unsafe_innerHTML=(
                    "<div class='rwddt-header'>"
                    "<img "
                    "src='https://rockyworlds.stsci.edu/images/"
                    "rp_logo_03.png' "
                    "width='50px' "
                    "alt='Rocky Worlds logo'"
                    ">"
                    "<h1>Rocky Worlds DDT Data Challenge Submission Form</h1>"
                    "</div>"
                )
            )

        if not render_ready:
            loading_form()
        else:
            with solara.lab.Tabs(
                value=tab_index,
                dark=False,
                background_color='#ADD8E6',
                color='black',
                lazy=False,
            ):
                with solara.lab.Tab('Load'):
                    with solara.Column():
                        with solara.Row():
                            def on_load_blank():
                                """Reset the widget to a blank form."""
                                if loaded_source.value is not None:
                                    form_dict.set(deepcopy(BLANK_FORM))
                                    loaded_source.set(None)
                                tab_index.set(1)

                            solara.Button(
                                "Load blank form",
                                on_click=on_load_blank,
                                style=("background-color: #D16200; "
                                       "color: white; margin-top: 10px;")
                            )
                        with solara.Column():
                            solara.Markdown('or select a JSON file to load:')
                            with solara.Card(
                                subtitle=None,
                            ):
                                solara.FileBrowser(
                                    directory=directory,
                                    selected=selected,
                                    filter=file_browser_filter,
                                    on_file_name=on_path_select,
                                    on_path_select=on_path_select,
                                )
                                solara.Button(
                                    icon_name='mdi-file-download',
                                    label='Load Selected JSON',
                                    on_click=on_load_json,
                                    style=("background-color: #D16200; "
                                           "color: white; margin-top: 10px;")
                                )

                with solara.lab.Tab('Form'):
                    question_widgets(selected)

                with solara.lab.Tab('Validate'):
                    validate_form()

                with solara.lab.Tab('Export'):
                    def get_data():
                        """Return the current form responses as JSON.

                        Returns
                        -------
                        str
                            Serialized form responses.
                        """
                        return Form(
                            dictionary=form_dict.value
                        ).to_json(validate=True)

                    with solara.Column(style=''):
                        with solara.Row():
                            def on_target_set(target):
                                """Update the export filename for a target.

                                Parameters
                                ----------
                                target : str
                                    Selected challenge target name.
                                """
                                filenames = {
                                    'GJ 3929 b': 'form_GJ3929b.json',
                                    'LHS 1140 b': 'form_LHS1140b.json',
                                }
                                filename.set(filenames[target])

                            solara.Select(
                                label='Target:',
                                value='GJ 3929 b',
                                values=['GJ 3929 b', 'LHS 1140 b'],
                                on_value=on_target_set
                            )
                            solara.InputText(
                                label='Export to filename',
                                value=filename.value,
                                on_value=filename.set
                            )
                            solara.FileDownload(
                                get_data,
                                filename=filename.value,
                                label='Export form'
                            )


def validate():
    """Validate the current interactive form and store a status message."""
    if not form_dict.value:
        errors.set("No form was loaded.")
        return

    with tempfile.NamedTemporaryFile() as tmp_file:
        with open(tmp_file.name, 'w') as fs:
            json.dump(form_dict.value, fs)
        try:
            Form(path=tmp_file.name)
            errors.set("Form responses are valid.")
        except Exception as e:
            errors.set(f'Invalid form response:\n\t{e}')


@solara.component
def validate_form():
    """Render the validation tab for the interactive form."""
    with solara.Card("Validation"):
        with solara.Column():
            solara.Button(
                label='click to validate responses',
                on_click=validate,
                style='max-width: 30vh;'
            )
            solara.Text(
                errors.value,
                classes=['rwddt-validation-message'],
            )
