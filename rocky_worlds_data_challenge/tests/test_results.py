import json
import os
import tempfile
import zipfile
from importlib import resources

import h5py
import numpy as np
import pytest

from rocky_worlds_data_challenge import (  # noqa: E501, E402
    Form,
    Photometry,
    Posterior,
    Results,
)


def _completed_template_form():
    """Return a template form with example responses filled in."""
    template_form = (
        resources.files("rocky_worlds_data_challenge") /
        "data" /
        "form.json"
    )

    with template_form.open('r') as json_file:
        completed_form = json.load(json_file)

    for fields in completed_form.values():
        fields['response'] = fields['example']

    return completed_form


def test_result_validation():
    """Validate writing and reading a complete submission archive."""

    posterior = Posterior(
        samples=np.ones((7, 10_000)),
        parameter_keys=np.array([
            'depth_ecl', 't_ecl', 'b_ecl', 'per',
            'ecosw', 'esinw', 'non-standard key'
        ])
    )

    # Number of exposures:
    phot_shape = (1500, )

    phot = Photometry(
        time=np.linspace(0, 1, phot_shape[0]),
        raw_flux=np.ones(phot_shape),
        raw_flux_err=np.ones(phot_shape),
        astro_model=np.ones(phot_shape),
        noise_model=np.ones(phot_shape),
        full_model=np.ones(phot_shape),
        centroid_x=np.ones(phot_shape),
        centroid_y=np.ones(phot_shape),
        centroid_sx=np.ones(phot_shape),
        centroid_sy=np.ones(phot_shape),
        background=np.zeros(phot_shape),
    )

    completed_form_GJ3929b = _completed_template_form()
    completed_form_LHS1140b = _completed_template_form()

    with tempfile.TemporaryDirectory() as temp_dir:
        zip_path = os.path.join(temp_dir, 'test-submission.zip')

        # contents are validated on __init__:
        results = Results(
            photometry_GJ_3929_b=phot,
            posterior_GJ_3929_b=posterior,
            form_GJ_3929_b=Form(dictionary=completed_form_GJ3929b),
            photometry_LHS_1140_b=phot,
            posterior_LHS_1140_b=posterior,
            form_LHS_1140_b=Form(dictionary=completed_form_LHS1140b),
        )

        results.to_submission(zip_path)

        with zipfile.ZipFile(zip_path, 'r') as zip_file:
            assert set(zip_file.namelist()) == {
                'posterior_LHS1140b.txt',
                'posterior_GJ3929b.txt',
                'lc_GJ3929b.h5',
                'lc_LHS1140b.h5',
                'form_GJ3929b.json',
                'form_LHS1140b.json',
            }

            with zip_file.open('lc_GJ3929b.h5') as h5_stream:
                with h5py.File(h5_stream, 'r') as h5_file:
                    assert 'background' in h5_file['photometry']

        reloaded_results = Results.load(zip_path)
        reloaded_results.validate(verbose=False)
        assert hasattr(reloaded_results.photometry_GJ_3929_b, 'background')
        np.testing.assert_equal(
            reloaded_results.photometry_GJ_3929_b.background,
            np.zeros(phot_shape),
        )


def test_form_prior_spec_validation():
    """Reject unsupported prior-distribution labels."""

    completed_form = _completed_template_form()

    # The template examples should be valid complete responses.
    Form(dictionary=completed_form)

    completed_form['21']['response']['prior_distribution_type'] = 'Gaussian'

    with pytest.raises(ValueError, match='prior distribution type'):
        Form(dictionary=completed_form)


def test_form_select_placeholder_validation():
    """Reject select placeholder values for required responses."""

    completed_form = _completed_template_form()
    completed_form['01']['response'] = 'Select an option'

    with pytest.raises(ValueError, match='required selection'):
        Form(dictionary=completed_form)

    completed_form = _completed_template_form()
    completed_form['21']['response'][
        'prior_distribution_type'
    ] = 'Select an option'

    with pytest.raises(ValueError, match='prior distribution type'):
        Form(dictionary=completed_form)


def test_posterior_normalizes_array_like_parameter_keys():
    """Convert array-like parameter keys to plain Python strings."""

    posterior = Posterior(
        samples=np.ones((2, 10_000)),
        parameter_keys=np.array(['depth_ecl', 't_ecl']),
    )

    assert posterior.parameter_keys == ['depth_ecl', 't_ecl']


def test_posterior_normalizes_unicode_parameter_keys_for_hdf5_attribute():
    """Convert fixed-width NumPy unicode parameter keys before HDF5 writing."""

    posterior = Posterior(
        samples=np.ones((2, 10_000)),
        parameter_keys=np.array(['depth_ecl', 't_ecl'], dtype='<U12'),
    )

    assert posterior.parameter_keys == ['depth_ecl', 't_ecl']

    with tempfile.TemporaryDirectory() as temp_dir:
        h5_path = os.path.join(temp_dir, 'posterior.h5')

        with h5py.File(h5_path, 'w') as h5_file:
            posterior_dataset = h5_file.create_dataset(
                'posterior_samples',
                data=posterior.samples,
            )
            posterior_dataset.attrs['parameter_keys'] = (
                posterior.parameter_keys
            )

        with h5py.File(h5_path, 'r') as h5_file:
            np.testing.assert_equal(
                h5_file['posterior_samples'].attrs['parameter_keys'],
                np.array(['depth_ecl', 't_ecl'], dtype=object),
            )


def test_posterior_rejects_single_string_parameter_keys():
    """Reject a single string supplied as posterior parameter keys."""

    with pytest.raises(ValueError, match='not a single string'):
        Posterior(
            samples=np.ones((2, 10_000)),
            parameter_keys='depth_ecl',
        )
