import json
import os
import tempfile
import warnings
import zipfile
from dataclasses import dataclass

import h5py
import numpy as np
import pandas as pd
from numpy.typing import ArrayLike

__all__ = [
    'Photometry', 'Posterior', 'Results', 'Form'
]


json_formatting = dict(
    sort_keys=True,
    indent=2,
)

empty_form_path = os.path.join(
    os.path.dirname(os.path.abspath(__file__)),
    'data',
    'form.json'
)


POSTERIOR_LHS_1140_B_FILENAME = 'posterior_LHS1140b.txt'
POSTERIOR_GJ_3929_B_FILENAME = 'posterior_GJ3929b.txt'

PHOTOMETRY_LHS_1140_B_FILENAME = 'lc_LHS1140b.h5'
PHOTOMETRY_GJ_3929_B_FILENAME = 'lc_GJ3929b.h5'

FORM_LHS_1140_B_FILENAME = 'form_LHS1140b.json'
FORM_GJ_3929_B_FILENAME = 'form_GJ3929b.json'

N_SAMPLES_FAIL = 1_000
N_SAMPLES_WARN = 10_000
SELECT_PLACEHOLDER = 'Select an option'


class Photometry:
    """Reduced light-curve arrays for one target.

    All required and optional time-series arrays must share the same shape.
    Extra keyword arguments are preserved as additional HDF5 datasets when a
    submission archive is written.

    Attributes
    ----------
    time : array-like
        Time values for the reduced light curve.
    raw_flux : array-like
        Reduced flux values before applying the submitted model correction.
    raw_flux_err : array-like
        Uncertainties on ``raw_flux``.
    astro_model : array-like
        Astrophysical model evaluated at each time.
    noise_model : array-like
        Noise or systematics model evaluated at each time.
    full_model : array-like
        Combined astrophysical and noise model evaluated at each time.
    extra_time_series : array-like
        Optional additional arrays supplied as keyword arguments. These are
        attached as attributes and written to the photometry HDF5 group.
    """

    #: Time values for the reduced light curve.
    time: ArrayLike = None
    #: Reduced flux values before applying the submitted model correction.
    raw_flux: ArrayLike = None
    #: Uncertainties on ``raw_flux``.
    raw_flux_err: ArrayLike = None
    #: Astrophysical model evaluated at each time.
    astro_model: ArrayLike = None
    #: Noise or systematics model evaluated at each time.
    noise_model: ArrayLike = None
    #: Combined astrophysical and noise model evaluated at each time.
    full_model: ArrayLike = None

    required_fields = [
        'time',
        'raw_flux',
        'raw_flux_err',
        'astro_model',
        'noise_model',
        'full_model',
    ]

    def __init__(self, time: ArrayLike, raw_flux: ArrayLike,
                 raw_flux_err: ArrayLike, astro_model: ArrayLike,
                 noise_model: ArrayLike, full_model: ArrayLike,
                 **extra_time_series: ArrayLike):
        """Store required and optional light-curve time series.

        Parameters
        ----------
        time : array-like
            Time values for the reduced light curve.
        raw_flux : array-like
            Reduced flux values before applying the submitted model
            correction.
        raw_flux_err : array-like
            Uncertainties on ``raw_flux``.
        astro_model : array-like
            Astrophysical model evaluated at each time.
        noise_model : array-like
            Noise or systematics model evaluated at each time.
        full_model : array-like
            Combined astrophysical and noise model evaluated at each time.
        **extra_time_series : array-like
            Optional additional arrays to preserve in the photometry HDF5
            file. Each array must have the same shape as the required arrays.
        """
        self.time = time
        self.raw_flux = raw_flux
        self.raw_flux_err = raw_flux_err
        self.astro_model = astro_model
        self.noise_model = noise_model
        self.full_model = full_model

        for key, value in extra_time_series.items():
            setattr(self, key, value)

    def validate(self):
        """Require every stored time-series array to have the same shape.

        Raises
        ------
        ValueError
            If any stored light-curve array has a different shape.
        """
        shapes = {
            np.asarray(value).shape
            for key, value in self.__dict__.items()
        }

        if len(shapes) > 1:
            raise ValueError(
                "All light curve arrays must have the same shape, "
                f"got {shapes}."
            )

    @classmethod
    def load(cls, filepath='submission.zip',
             h5_filepath=PHOTOMETRY_LHS_1140_B_FILENAME):
        """Load one photometry dataset from a submission ZIP archive.

        Parameters
        ----------
        filepath : path-like, optional
            Path to the submission ZIP archive.
        h5_filepath : str, optional
            Name of the photometry HDF5 file inside ``filepath``.

        Returns
        -------
        Photometry
            Photometry object populated from the HDF5 file.
        """
        with zipfile.ZipFile(filepath, 'r') as zip_file:
            with zip_file.open(h5_filepath) as h5_stream:
                with h5py.File(h5_stream, 'r') as results:
                    return cls(**{
                        key: array[:]
                        for key, array in results['photometry'].items()
                    })


@dataclass(init=False)
class Posterior:
    """Posterior samples and parameter names for one planet.

    Attributes
    ----------
    samples : array-like
        Posterior samples with shape ``(n_parameters, n_samples)``.
    parameter_keys : array-like
        Names corresponding to each row of ``samples``. Keys are normalized to
        a flat list of Python strings.
    """

    #: Posterior samples with shape ``(n_parameters, n_samples)``.
    samples: ArrayLike = None
    #: Names corresponding to each row of ``samples``.
    parameter_keys: ArrayLike = None

    def __init__(self, samples: ArrayLike, parameter_keys: ArrayLike):
        """Normalize inputs and validate the posterior sample table.

        Parameters
        ----------
        samples : array-like
            Posterior samples with shape ``(n_parameters, n_samples)``.
        parameter_keys : array-like
            Names corresponding to each row of ``samples``.

        Raises
        ------
        ValueError
            If the posterior samples or parameter keys fail validation.
        """
        self.samples = np.asarray(samples)
        self.parameter_keys = self._normalize_parameter_keys(
            parameter_keys
        )
        self.validate()

    @staticmethod
    def _normalize_parameter_keys(parameter_keys):
        """Return parameter names as a flat list of Python strings.

        Parameters
        ----------
        parameter_keys : array-like
            Iterable of parameter names.

        Returns
        -------
        list of str
            Flattened parameter names converted to Python strings.

        Raises
        ------
        ValueError
            If ``parameter_keys`` is a single string rather than an iterable
            of strings.
        """
        if isinstance(parameter_keys, (str, bytes)):
            raise ValueError(
                "parameter_keys should be an iterable of parameter names, "
                "not a single string."
            )

        try:
            parameter_keys = np.asarray(parameter_keys).ravel().tolist()
        except TypeError:
            parameter_keys = list(parameter_keys)

        return [
            key.decode() if isinstance(key, bytes) else str(key)
            for key in parameter_keys
        ]

    def validate(self):
        """Check posterior shape, sample count, and eclipse-depth values.

        Raises
        ------
        ValueError
            If the sample array orientation, parameter-key count, sample
            count, required ``depth_ecl`` key, or ``depth_ecl`` values are
            invalid.
        UserWarning
            If the posterior has fewer than the recommended number of samples
            but still has enough samples to construct a submission.
        """
        n_params, n_samples = self.samples.shape
        if n_params > n_samples:
            raise ValueError(
                "Posterior samples should have shape (N, M) where N is "
                "the number of parameters and M is the number of samples, "
                "but samples.shape == "
                f"{self.samples.shape}."
            )

        if n_params != len(self.parameter_keys):
            raise ValueError(
                f"Posterior samples were given for {n_params} parameters, but "
                f"parameter_keys only given for {len(self.parameter_keys)} "
                "parameters. These should match."
            )

        if n_samples < N_SAMPLES_WARN:
            with warnings.catch_warnings():
                warnings.simplefilter("always", UserWarning)
                warnings.warn(
                    f"Kaggle grading of a Posterior object requires at least "
                    f"{N_SAMPLES_WARN} posterior samples, got {n_samples}.\n",
                    UserWarning,
                    stacklevel=2,
                )

        if n_samples < N_SAMPLES_FAIL:
            raise ValueError(
                f"Preparation of a Posterior object requires at least "
                f"{N_SAMPLES_FAIL} posterior samples, got {n_samples}."
            )

        if 'depth_ecl' not in self.parameter_keys:
            raise ValueError(
                "Posterior samples for the eclipse depth are required, "
                "but 'depth_ecl' is not in "
                f"parameter_keys={self.parameter_keys}."
            )

        depth_index = list(self.parameter_keys).index('depth_ecl')
        depth_ecl = np.asarray(self.samples[depth_index], dtype=float)

        if not np.isfinite(depth_ecl).all():
            raise ValueError(
                "Posterior samples for 'depth_ecl' must all be finite values. "
                "Found NaN, inf, or -inf values."
            )

        typical_abs_depth = np.nanmedian(np.abs(depth_ecl))

        if typical_abs_depth < 1:
            raise ValueError(
                "Posterior samples for 'depth_ecl' appear to be unitless "
                "eclipse depths rather than ppm. Please provide 'depth_ecl' "
                "in parts per million (ppm). For example, use 10 ppm instead "
                "of 1e-5."
            )

    @classmethod
    def load(cls, filepath='submission.zip',
             posterior_filepath=POSTERIOR_LHS_1140_B_FILENAME):
        """Load one posterior table from a submission ZIP archive.

        Parameters
        ----------
        filepath : path-like, optional
            Path to the submission ZIP archive.
        posterior_filepath : str, optional
            Name of the posterior text file inside ``filepath``.

        Returns
        -------
        Posterior
            Posterior object loaded from the CSV-formatted text file.
        """
        with zipfile.ZipFile(filepath, 'r') as zip_file:
            # The posterior file is CSV-formatted text, even though it has a
            # .txt extension to avoid Kaggle treating it as a standard
            # submission CSV.
            results = pd.read_csv(zip_file.open(posterior_filepath))

            return cls(
                samples=results.to_numpy().T,
                parameter_keys=list(results.columns),
            )


class Form:
    """Submission questionnaire loaded from or written to JSON.

    Parameters
    ----------
    path : str, optional
        Path to a JSON form file to load.
    dictionary : dict, optional
        In-memory form dictionary keyed by question number. If supplied,
        ``path`` is not read.
    validate : bool, optional
        Whether to validate responses during initialization.

    Attributes
    ----------
    path : str
        Path to a JSON form file to load.
    dictionary : dict
        In-memory form dictionary keyed by question number. Each question
        stores metadata such as prompt, description, format, example, and
        response.
    """

    #: Path to a JSON form file to load.
    path: str = None
    #: In-memory form dictionary keyed by question number.
    dictionary: dict = None

    def __init__(self, path: str = None, dictionary: dict = None,
                 validate: bool = True):
        """Load the form dictionary from disk and optionally validate it.

        Parameters
        ----------
        path : str, optional
            Path to a JSON form file to load.
        dictionary : dict, optional
            In-memory form dictionary keyed by question number. If supplied,
            ``path`` is not read.
        validate : bool, optional
            Whether to validate responses during initialization.

        Raises
        ------
        ValueError
            If validation is enabled and a form response is invalid.
        """
        self.path = path
        self.dictionary = dictionary

        if self.dictionary is None:
            with open(self.path, 'r') as json_file:
                self.dictionary = json.load(json_file)

        if validate:
            self.validate()

    @classmethod
    def blank(cls):
        """Return a blank copy of the packaged submission form.

        Returns
        -------
        Form
            Form initialized from the packaged blank form template without
            validating empty responses.
        """
        return cls(path=empty_form_path, validate=False)

    def standardized_dictionary(self):
        """Return a standardized JSON-serializable form dictionary.

        Returns
        -------
        dict
            Copy of the form dictionary with two-digit question keys and a
            ``response`` field for every question.
        """
        standardized = {}

        for question_number in sorted(self.dictionary, key=lambda k: int(k)):
            fields = dict(self.dictionary[question_number])
            fields.setdefault('response', '')
            standardized[f"{int(question_number):02d}"] = fields

        return standardized

    def validate(self):
        """Validate all form responses against their declared formats.

        Raises
        ------
        ValueError
            If a required response is missing or a response cannot be parsed
            according to its declared format.
        """
        for question_number, fields in self.dictionary.items():
            prompt = fields.get('prompt', f'Question {question_number}')
            field_format = fields.get('format', 'str')
            required = bool(fields.get('required', False))

            response = fields.get('response', '')
            if response is None:
                response = ''

            if field_format == 'prior_spec':
                valid_options = fields.get('options', [])

                if not isinstance(response, dict):
                    raise ValueError(
                        f"For prompt '{prompt}' "
                        f"(Question {question_number}), the response must be "
                        "a dictionary with keys 'parameter', "
                        "'prior_distribution_type', and 'prior_parameters'."
                    )

                parameter = str(response.get('parameter', '')).strip()
                prior_type = str(
                    response.get('prior_distribution_type', '')
                ).strip()
                prior_parameters = str(
                    response.get('prior_parameters', '')
                ).strip()

                if required and not parameter:
                    raise ValueError(
                        f"Missing a required prior parameter for prompt "
                        f"'{prompt}' (Question {question_number})."
                    )

                if required and (
                    not prior_type or prior_type == SELECT_PLACEHOLDER
                ):
                    raise ValueError(
                        f"Missing a required prior distribution type for "
                        f"prompt '{prompt}' (Question {question_number})."
                    )

                if prior_type and prior_type not in valid_options:
                    raise ValueError(
                        f"For prompt '{prompt}' "
                        f"(Question {question_number}), the prior "
                        f"distribution type '{prior_type}' "
                        f"is not one of: {valid_options}"
                    )

                if (
                    required and
                    prior_type != 'not applicable' and
                    not prior_parameters
                ):
                    raise ValueError(
                        f"Missing required prior parameters for prompt "
                        f"'{prompt}' (Question {question_number})."
                    )

                continue

            response = str(response)
            stripped_response = response.strip()

            if not len(stripped_response) and required:
                raise ValueError(
                    f"Missing a required response for prompt "
                    f"'{prompt}' (Question {question_number})."
                )

            if not len(stripped_response) and not required:
                continue

            if field_format == 'float/int':
                try:
                    float(stripped_response)
                except ValueError as e:
                    raise ValueError(
                        f"For prompt '{prompt}' "
                        f"(Question {question_number}), "
                        f"the response '{response}' "
                        f"cannot be interpreted as a Python float "
                        f"or integer.\n\nUnderlying error: {e}"
                    )

            elif field_format == 'list[float/int]':
                try:
                    list(map(float, stripped_response.split(',')))
                except ValueError as e:
                    raise ValueError(
                        f"For prompt '{prompt}' "
                        f"(Question {question_number}), "
                        f"the response '{response}' "
                        "cannot be interpreted as a comma-separated list of "
                        "floats or integers.\n\n"
                        f"Underlying error: {e}"
                    )

            elif field_format == 'y/n':
                valid_yn = ['y', 'n', 'yes', 'no']

                if stripped_response.lower() not in valid_yn:
                    raise ValueError(
                        f"For prompt '{prompt}' "
                        f"(Question {question_number}), the response "
                        f"'{response.lower()}' "
                        f"is not one of: {valid_yn}"
                    )

            elif field_format == 'select':
                valid_options = fields.get('options', [])

                if required and stripped_response == SELECT_PLACEHOLDER:
                    raise ValueError(
                        f"Missing a required selection for prompt "
                        f"'{prompt}' (Question {question_number})."
                    )

                if stripped_response not in valid_options:
                    raise ValueError(
                        f"For prompt '{prompt}' "
                        f"(Question {question_number}), the response "
                        f"'{response}' "
                        f"is not one of: {valid_options}"
                    )

    def to_json(self, validate=True):
        """Return this form as a formatted JSON string.

        Parameters
        ----------
        validate : bool, optional
            Validate the form before serializing it.

        Returns
        -------
        str
            Form dictionary serialized as formatted JSON.
        """
        if validate:
            self.validate()

        return json.dumps(
            self.standardized_dictionary(),
            **json_formatting
        )

    def save(self, filepath, overwrite=False, validate=True):
        """Save this form as a plain JSON file.

        Parameters
        ----------
        filepath : path-like
            Destination JSON filepath.
        overwrite : bool, optional
            Allow an existing file at ``filepath`` to be replaced.
        validate : bool, optional
            Validate the form before writing it.

        Returns
        -------
        str
            Absolute path to the written file.

        Raises
        ------
        FileExistsError
            If ``filepath`` already exists and ``overwrite`` is `False`.
        """
        filepath = os.fspath(filepath)

        if os.path.exists(filepath) and not overwrite:
            raise FileExistsError(
                f"File already exists at {filepath}. To overwrite it, "
                "set overwrite=True."
            )

        with open(filepath, 'w') as json_file:
            json_file.write(self.to_json(validate=validate))
            json_file.write('\n')

        abspath = os.path.abspath(filepath)
        filesize_kb = os.stat(filepath).st_size / 1e3
        print(f"Form written to {abspath}: ({filesize_kb:.3f} kB)")

        return abspath

    @classmethod
    def load(cls, json_filepath, zip_filepath='submission.zip'):
        """Load one form JSON file from a submission ZIP archive.

        Parameters
        ----------
        json_filepath : str
            Name of the JSON form file inside ``zip_filepath``.
        zip_filepath : path-like, optional
            Path to the submission ZIP archive.

        Returns
        -------
        Form
            Form loaded from the JSON file in the archive.
        """
        with zipfile.ZipFile(zip_filepath, 'r') as zip_file:
            with zip_file.open(json_filepath, mode='r') as questions:
                return cls(dictionary=json.load(questions))

    def __repr__(self):
        """Return a readable representation of the standardized form.

        Returns
        -------
        str
            Representation containing the standardized form dictionary.
        """
        formatted_dictionary = json.dumps(
            self.standardized_dictionary(),
            **json_formatting
        )
        return (
            f"{self.__class__.__name__}(\n"
            f"  dictionary={formatted_dictionary})"
        )


@dataclass(init=False)
class Results:
    """Complete challenge submission for both target planets.

    Attributes
    ----------
    photometry_GJ_3929_b : Photometry
        Reduced photometry products for GJ 3929 b.
    posterior_GJ_3929_b : Posterior
        Posterior samples for GJ 3929 b.
    form_GJ_3929_b : Form
        Completed questionnaire responses for GJ 3929 b.
    photometry_LHS_1140_b : Photometry
        Reduced photometry products for LHS 1140 b.
    posterior_LHS_1140_b : Posterior
        Posterior samples for LHS 1140 b.
    form_LHS_1140_b : Form
        Completed questionnaire responses for LHS 1140 b.
    """

    #: Reduced photometry products for GJ 3929 b.
    photometry_GJ_3929_b: Photometry = None
    #: Posterior samples for GJ 3929 b.
    posterior_GJ_3929_b: Posterior = None
    #: Completed questionnaire responses for GJ 3929 b.
    form_GJ_3929_b: Form = None
    #: Reduced photometry products for LHS 1140 b.
    photometry_LHS_1140_b: Photometry = None
    #: Posterior samples for LHS 1140 b.
    posterior_LHS_1140_b: Posterior = None
    #: Completed questionnaire responses for LHS 1140 b.
    form_LHS_1140_b: Form = None

    def __init__(self, photometry_GJ_3929_b: Photometry,
                 posterior_GJ_3929_b: Posterior,
                 form_GJ_3929_b: Form,
                 photometry_LHS_1140_b: Photometry,
                 posterior_LHS_1140_b: Posterior,
                 form_LHS_1140_b: Form):
        """Store all components needed for one complete submission.

        Parameters
        ----------
        photometry_GJ_3929_b : Photometry
            Reduced photometry products for GJ 3929 b.
        posterior_GJ_3929_b : Posterior
            Posterior samples for GJ 3929 b.
        form_GJ_3929_b : Form
            Completed questionnaire responses for GJ 3929 b.
        photometry_LHS_1140_b : Photometry
            Reduced photometry products for LHS 1140 b.
        posterior_LHS_1140_b : Posterior
            Posterior samples for LHS 1140 b.
        form_LHS_1140_b : Form
            Completed questionnaire responses for LHS 1140 b.
        """
        self.photometry_GJ_3929_b = photometry_GJ_3929_b
        self.posterior_GJ_3929_b = posterior_GJ_3929_b
        self.form_GJ_3929_b = form_GJ_3929_b
        self.photometry_LHS_1140_b = photometry_LHS_1140_b
        self.posterior_LHS_1140_b = posterior_LHS_1140_b
        self.form_LHS_1140_b = form_LHS_1140_b

    def _write_posterior_txt(self, posterior, tmp_dir, zip_file, filename):
        """Write posterior samples as CSV-formatted text.

        The contents are comma-separated and readable with
        pandas.read_csv(...), but the .txt extension avoids Kaggle treating
        these files as standard submission CSV files.

        Parameters
        ----------
        posterior : Posterior
            Posterior samples to write.
        tmp_dir : path-like
            Temporary directory used to stage the text file.
        zip_file : zipfile.ZipFile
            Open ZIP archive to receive the text file.
        filename : str
            Archive member name for the posterior text file.
        """
        tmp_txt_path = os.path.join(tmp_dir, f'tmp_{filename}')

        pd.DataFrame(
            data=posterior.samples.T,
            columns=posterior.parameter_keys
        ).to_csv(
            tmp_txt_path,
            index=False,
        )

        zip_file.write(tmp_txt_path, filename)

    def _write_photometry_h5(self, photometry, posterior, tmp_dir, zip_file,
                             filename):
        """Write photometry and posterior samples to one HDF5 file.

        Parameters
        ----------
        photometry : Photometry
            Reduced photometry products to write.
        posterior : Posterior
            Posterior samples to include alongside the photometry.
        tmp_dir : path-like
            Temporary directory used to stage the HDF5 file.
        zip_file : zipfile.ZipFile
            Open ZIP archive to receive the HDF5 file.
        filename : str
            Archive member name for the HDF5 file.
        """
        tmp_h5_path = os.path.join(tmp_dir, f'tmp_{filename}')

        with h5py.File(tmp_h5_path, 'w') as h5_file:
            samples_group = h5_file.create_dataset(
                'posterior_samples',
                data=posterior.samples
            )
            samples_group.attrs['parameter_keys'] = posterior.parameter_keys

            photometry_group = h5_file.create_group('photometry')

            for field, array in vars(photometry).items():
                photometry_group.create_dataset(field, data=array)

        zip_file.write(tmp_h5_path, filename)

    def _write_form_json(self, form, tmp_dir, zip_file, filename):
        """Write one completed form JSON file into the ZIP archive.

        Parameters
        ----------
        form : Form
            Completed form to write.
        tmp_dir : path-like
            Temporary directory used to stage the JSON file.
        zip_file : zipfile.ZipFile
            Open ZIP archive to receive the JSON file.
        filename : str
            Archive member name for the form JSON file.
        """
        tmp_json_path = os.path.join(tmp_dir, f'tmp_{filename}')

        with open(tmp_json_path, 'w') as tmp_json_file:
            tmp_json_file.write(form.to_json(validate=False))
            tmp_json_file.write('\n')

        zip_file.write(tmp_json_path, filename)

    def to_submission(self, filepath='submission.zip', overwrite=False):
        """Write a Kaggle submission ZIP archive.

        The archive contains exactly:

            posterior_LHS1140b.txt
            posterior_GJ3929b.txt
            lc_GJ3929b.h5
            lc_LHS1140b.h5
            form_GJ3929b.json
            form_LHS1140b.json

        The posterior .txt files contain CSV-formatted text.

        Parameters
        ----------
        filepath : path-like, optional
            Destination path for the submission ZIP archive.
        overwrite : bool, optional
            Allow an existing file at ``filepath`` to be replaced.

        Returns
        -------
        str
            Absolute path to the written ZIP archive.

        Raises
        ------
        FileExistsError
            If ``filepath`` already exists and ``overwrite`` is `False`.
        ValueError
            If any contained photometry, posterior, or form object fails
            validation.
        """

        self.validate()

        if os.path.exists(filepath) and not overwrite:
            raise FileExistsError(
                f"File already exists at {filepath}. To overwrite it, "
                "set overwrite=True."
            )

        with tempfile.TemporaryDirectory() as tmp_dir:
            with zipfile.ZipFile(
                filepath,
                mode='w',
                compression=zipfile.ZIP_DEFLATED
            ) as zip_file:

                self._write_posterior_txt(
                    posterior=self.posterior_LHS_1140_b,
                    tmp_dir=tmp_dir,
                    zip_file=zip_file,
                    filename=POSTERIOR_LHS_1140_B_FILENAME,
                )

                self._write_posterior_txt(
                    posterior=self.posterior_GJ_3929_b,
                    tmp_dir=tmp_dir,
                    zip_file=zip_file,
                    filename=POSTERIOR_GJ_3929_B_FILENAME,
                )

                self._write_photometry_h5(
                    photometry=self.photometry_LHS_1140_b,
                    posterior=self.posterior_LHS_1140_b,
                    tmp_dir=tmp_dir,
                    zip_file=zip_file,
                    filename=PHOTOMETRY_LHS_1140_B_FILENAME,
                )

                self._write_photometry_h5(
                    photometry=self.photometry_GJ_3929_b,
                    posterior=self.posterior_GJ_3929_b,
                    tmp_dir=tmp_dir,
                    zip_file=zip_file,
                    filename=PHOTOMETRY_GJ_3929_B_FILENAME,
                )

                self._write_form_json(
                    form=self.form_LHS_1140_b,
                    tmp_dir=tmp_dir,
                    zip_file=zip_file,
                    filename=FORM_LHS_1140_B_FILENAME,
                )

                self._write_form_json(
                    form=self.form_GJ_3929_b,
                    tmp_dir=tmp_dir,
                    zip_file=zip_file,
                    filename=FORM_GJ_3929_B_FILENAME,
                )

        abspath = os.path.abspath(filepath)
        filesize_mb = os.stat(filepath).st_size / 1e6
        print(f"Results written to {abspath}: ({filesize_mb:.3f} MB)")

        return abspath

    @classmethod
    def load(cls, filepath='submission.zip', validate=True):
        """Load a complete submission ZIP archive.

        Parameters
        ----------
        filepath : path-like, optional
            Path to the submission ZIP archive.
        validate : bool, optional
            Validate the loaded submission before returning it.

        Returns
        -------
        Results
            Complete submission reconstructed from the archive.

        Raises
        ------
        ValueError
            If ``validate`` is `True` and any loaded component is invalid.
        """
        with zipfile.ZipFile(filepath, 'r') as zip_file:
            with zip_file.open(POSTERIOR_LHS_1140_B_FILENAME) as f:
                posterior_LHS_1140_b_df = pd.read_csv(f)

            with zip_file.open(POSTERIOR_GJ_3929_B_FILENAME) as f:
                posterior_GJ_3929_b_df = pd.read_csv(f)

            with zip_file.open(PHOTOMETRY_LHS_1140_B_FILENAME) as f:
                with h5py.File(f, 'r') as h5_file:
                    photometry_LHS_1140_b = Photometry(**{
                        key: array[:]
                        for key, array in h5_file['photometry'].items()
                    })

            with zip_file.open(PHOTOMETRY_GJ_3929_B_FILENAME) as f:
                with h5py.File(f, 'r') as h5_file:
                    photometry_GJ_3929_b = Photometry(**{
                        key: array[:]
                        for key, array in h5_file['photometry'].items()
                    })

            with zip_file.open(FORM_LHS_1140_B_FILENAME, mode='r') as f:
                form_LHS_1140_b = Form(dictionary=json.load(f))

            with zip_file.open(FORM_GJ_3929_B_FILENAME, mode='r') as f:
                form_GJ_3929_b = Form(dictionary=json.load(f))

        instance = cls(
            photometry_GJ_3929_b=photometry_GJ_3929_b,
            posterior_GJ_3929_b=Posterior(
                samples=posterior_GJ_3929_b_df.to_numpy().T,
                parameter_keys=list(posterior_GJ_3929_b_df.columns),
            ),
            form_GJ_3929_b=form_GJ_3929_b,
            photometry_LHS_1140_b=photometry_LHS_1140_b,
            posterior_LHS_1140_b=Posterior(
                samples=posterior_LHS_1140_b_df.to_numpy().T,
                parameter_keys=list(posterior_LHS_1140_b_df.columns),
            ),
            form_LHS_1140_b=form_LHS_1140_b,
        )

        if validate:
            instance.validate()

        return instance

    def validate(self, verbose=True):
        """Validate all photometry, posterior, and form objects.

        Parameters
        ----------
        verbose : bool, optional
            Print a confirmation message after each component validates.

        Raises
        ------
        ValueError
            If any contained photometry, posterior, or form object is invalid.
        """
        self.photometry_GJ_3929_b.validate()
        if verbose:
            print("photometry (GJ 3929 b): valid")

        self.posterior_GJ_3929_b.validate()
        if verbose:
            print("posterior (GJ 3929 b): valid")

        self.form_GJ_3929_b.validate()
        if verbose:
            print("form (GJ 3929 b): valid")

        self.photometry_LHS_1140_b.validate()
        if verbose:
            print("photometry (LHS 1140 b): valid")

        self.posterior_LHS_1140_b.validate()
        if verbose:
            print("posterior (LHS 1140 b): valid")

        self.form_LHS_1140_b.validate()
        if verbose:
            print("form (LHS 1140 b): valid")

    def __repr__(self):
        """Return a compact summary of the submission contents.

        Returns
        -------
        str
            Summary of posterior sample shapes, photometry arrays, and form
            question counts.
        """
        return (
            f"<Results: \n"
            f"  posterior - GJ 3929 b "
            f"(samples: {self.posterior_GJ_3929_b.samples.shape}),\n"
            f"  photometry - GJ 3929 b "
            f"(arrays: {list(self.photometry_GJ_3929_b.__dict__.keys())}),\n"
            f"  form - GJ 3929 b "
            f"(questions: {len(self.form_GJ_3929_b.dictionary)}),\n"
            f"  posterior - LHS 1140 b "
            f"(samples: {self.posterior_LHS_1140_b.samples.shape}),\n"
            f"  photometry - LHS 1140 b "
            f"(arrays: {list(self.photometry_LHS_1140_b.__dict__.keys())}),\n"
            f"  form - LHS 1140 b "
            f"(questions: {len(self.form_LHS_1140_b.dictionary)})>"
        )
