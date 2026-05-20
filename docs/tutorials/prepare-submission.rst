Prepare Kaggle submissions
==========================

For the Rocky Worlds Director's Discretionary Time Data Challenge.

Outline
-------

1. Format *posterior samples* for submission
2. Format *photometry* for submission
3. Format the *forms* for submission
4. Combine the above into a single ZIP

In this example, we'll draw random numbers to submit as our "results" for many
fields. You should replace these with the real thing.

.. code-block:: python

    import numpy as np

    rng = np.random.default_rng(0)

Components of a valid submission
--------------------------------

The package defines Python objects for each file that makes up a submission.
The objects are:

1. ``Posterior``: contains posterior samples.
2. ``Photometry``: contains the reduced time series flux, astrophysical model,
   noise model, and "full model" (astrophysical + noise), with optional
   additional time-series products such as target centroids or background.
3. ``Form``: lists a series of questions about your reduction and analysis
   choices that must be answered for each target, in each submission.
4. ``Results``: combines the above components and writes a ZIP archive that's
   ready to submit to Kaggle.

.. code-block:: python

    import rocky_worlds_data_challenge as rw

1. Posterior samples
--------------------

In the cell below we create some artificial samples.

.. code-block:: python

    # generate fake posterior samples:
    n_parameters = 7
    n_posterior_samples = 10_000

    samples_shape = (n_parameters, n_posterior_samples)
    samples_GJ_3929_b = rng.normal(1000, 10, size=samples_shape)
    samples_LHS_1140_b = rng.normal(1000, 10, size=samples_shape)
    samples_GJ_3929_b

If your sampler produces weighted samples, ``Posterior`` should contain
equal-weight samples.

Now we write out ``parameter_keys``, defining names for each of the sampling
parameters in the posterior samples array. The length of ``parameter_keys``
must match the first dimension of the ``samples`` array above. A plain Python
list of strings is shown here, but NumPy arrays, pandas Index objects, tuples,
and other array-like containers of parameter names are also accepted; the
``Posterior`` object will convert them to strings internally before writing the
submission files.

The expected sample shape is ``(n_parameters, n_posterior_samples)``. If your
sampler returns samples in the opposite orientation,
``(n_posterior_samples, n_parameters)``, transpose the array before creating
the ``Posterior`` object. For example, use ``samples = samples.T``.

.. code-block:: python

    # Use one parameter name for each row of the samples array.
    # Lists, tuples, NumPy arrays, and pandas Index objects are all accepted.
    parameter_keys = [
        'depth_ecl',
        't_ecl',
        'b_ecl',
        'per',
        'ecosw',
        'esinw',
        'non-standard key',
    ]

    posterior_GJ_3929_b = rw.Posterior(
        samples=samples_GJ_3929_b,
        parameter_keys=parameter_keys,
    )
    posterior_LHS_1140_b = rw.Posterior(
        samples=samples_LHS_1140_b,
        parameter_keys=parameter_keys,
    )

The ``Posterior`` object will validate your inputs to make sure the dimensions
match expectations. If the inputs fail validation, ``Posterior`` will raise an
error.

What Parameters And Standard ``parameter_keys`` Are Supported?
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The required posterior parameter is ``depth_ecl``, the eclipse depth in parts
per million (ppm). The submission tools also understand the standard parameter
names defined by ``rw.EclipsingSystem``, listed below.

If your analysis allowed eclipse depths to vary between eclipses, or if you
fit each eclipse observation separately, please combine those results into one
representative posterior for the average eclipse depth before submission. The
exact averaging procedure is up to your team, but the submitted
``depth_ecl`` samples should describe the single average eclipse depth that
you want evaluated.

If your sampler uses different names, simply rename your posterior columns or
the entries in ``parameter_keys`` before creating the ``Posterior`` object. For
example, if your samples call the eclipse depth ``fpfs_ppm``, use
``depth_ecl`` in ``parameter_keys`` for that row.

Commonly useful standard keys include:

======================= =====================================================
Key                     Meaning
======================= =====================================================
``depth_ecl``           Eclipse depth [ppm]; required for grading.
``t_ecl``               Mid-eclipse time [BMJD_TDB].
``t_tra``               Mid-transit time [BMJD_TDB].
``per``                 Orbital period [days].
``rprs``                Planet-to-star radius ratio.
``depth_tra``           Fractional transit depth; converted internally to
                        ``rprs = sqrt(depth_tra)``.
``a``                   Semimajor axis in units of stellar radii (``a/Rs``).
``inc``                 Orbital inclination [deg].
``b_tra``               Transit impact parameter in units of stellar radii.
``b_ecl``               Eclipse impact parameter in units of stellar radii.
``ecc``, ``omega``      Orbital eccentricity and longitude of periastron
                        [deg].
``ecosw``, ``esinw``    Eccentricity times the cosine/sine of longitude of
                        periastron.
``secosw``, ``sesinw``  Square root of eccentricity times the cosine/sine of
                        longitude of periastron.
``rho_star``            Stellar density in units of solar density.
``dur_tra``             Transit duration [days].
``dur_ecl``             Eclipse duration [days].
======================= =====================================================

You may include additional non-standard keys in ``parameter_keys``; they will
be preserved in the posterior files.

The ``EclipsingSystem`` helper can also convert between several equivalent
parameterizations when enough information is provided. For example, it can
derive inclination (``inc``) from an impact parameter such as ``b_tra`` when it
also has the scaled semimajor axis ``a`` and the eccentricity/orientation
information, derive ``ecc`` and ``omega`` from ``ecosw``/``esinw`` or
``secosw``/``sesinw``, compute transit and eclipse impact parameters from
``a`` and ``inc``, and use duration plus impact parameter to recover ``a`` and
``inc``. These conversions are intended to make common posterior
parameterizations easier to compare, but your ``parameter_keys`` should still
use the standard names above so the submission tools know what each row
represents.

.. code-block:: python

    # Programmatic list of standard EclipsingSystem parameter keys:
    list(rw.EclipsingSystem.__dataclass_fields__)

2. Photometry
-------------

Each submission should include a ``Photometry`` object for each target. This
is where you provide the light curve products from your own reduction and
modeling: the time array, reduced flux and flux uncertainty, astrophysical
model, noise/systematics model, and the combined full model.

The arrays in a ``Photometry`` object should all describe the same time series
and therefore must have the same length. The example below uses placeholder
arrays so the notebook can run end-to-end; for a real submission, replace
these arrays with the values produced by your analysis. This object is the
container that writes those photometric products into the submission ZIP.

You may also include additional, non-mandatory time-series products as extra
keyword arguments. For example, if your analysis tracks the subtracted
sky/background level per integration, you can pass ``background=...``. If you
measured target centroids or PSF widths, you can include
``centroid_x=...``, ``centroid_y=...``, ``centroid_sx=...``, and
``centroid_sy=...``. These optional arrays will be written as additional
datasets in the photometry HDF5 file. You are encouraged to add as many
additional time-series products as you feel are necessary to reproduce your
results.

.. code-block:: python

    # number of samples in the photometric time series:
    n_time_series = 1500
    phot_shape = (n_time_series, )

    fake_times = np.linspace(0, 1, n_time_series)
    fake_time_series = np.ones(phot_shape)

    photometry_GJ_3929_b = rw.Photometry(
        # required
        time=fake_times,
        raw_flux=fake_time_series,
        raw_flux_err=fake_time_series,
        astro_model=fake_time_series,
        noise_model=fake_time_series,
        full_model=fake_time_series,

        # optional, additional time-series products
        # examples: target centroids, PSF widths, and background level
        centroid_x=fake_time_series,
        centroid_y=fake_time_series,
        centroid_sx=fake_time_series,
        centroid_sy=fake_time_series,
        background=fake_time_series,
    )

    photometry_LHS_1140_b = rw.Photometry(
        # required
        time=fake_times,
        raw_flux=fake_time_series,
        raw_flux_err=fake_time_series,
        astro_model=fake_time_series,
        noise_model=fake_time_series,
        full_model=fake_time_series,

        # optional, additional time-series products
        # examples: target centroids, PSF widths, and background level
        centroid_x=fake_time_series,
        centroid_y=fake_time_series,
        centroid_sx=fake_time_series,
        centroid_sy=fake_time_series,
        background=fake_time_series,
    )

3. Forms
--------

Each Kaggle submission must include two completed forms, one for the analysis
of GJ 3929 b, and another for LHS 1140 b. The forms ask questions about your
team, data reduction process, assumptions, and analysis.

The blank form template is packaged with ``rocky_worlds_data_challenge``. The
recommended way to create a local editable copy is to load that packaged
template with ``Form.blank()`` and save it:

.. code-block:: python

    import rocky_worlds_data_challenge as rw

    form = rw.Form.blank()
    form.save("form_GJ3929b.json", overwrite=True, validate=False)

There are a few ways to fill in a form:

* Load and save a blank form in Python with ``Form.blank()``.
* Load an existing form in Python with ``Form(path='/path/to/form.json')``.
* Interactively create a new form or edit an existing one with
  ``interactive_form()``.

Load a blank form in Python with ``Form.blank()``
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

``form = Form.blank()`` creates a blank form to fill in. Each question is an
entry in ``form.dictionary``.

You can fill in the form programmatically in Python.

.. code-block:: python

    form = rw.Form.blank()

    # Each entry has a number as its key in form.dictionary,
    # where the question number is formatted as a zero-padded string.

    # Each entry is also a dictionary, containing several components:
    # prompt, description, required, format, example, response.

    form.dictionary['01']['response'] = "GJ 3929 b"

    prompt = form.dictionary['01']['prompt']
    response = form.dictionary['01']['response']

    print(f"{prompt}: {response}")

Load an existing form in Python
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

If you'd like to start by modifying an existing form on your machine, use:

.. code-block:: python

    import rocky_worlds_data_challenge as rw

    form = rw.Form(path='/path/to/form.json')

Interactively create and edit forms in the Jupyter notebook
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Run the following to open an interactive widget in this Jupyter notebook for
creating, editing, validating, and saving these JSON forms. The widget will
appear below the executed cell.

The form will try to follow your notebook theme automatically. If your editor
renders the form with poor contrast, you can force a theme with
``rw.interactive_form(theme='dark')`` or
``rw.interactive_form(theme='light')``.

.. code-block:: python

    rw.interactive_form()

4. Results
----------

All results are combined into a single object which can write out your
submission into a ZIP archive.

.. code-block:: python

    results = rw.Results(
        posterior_GJ_3929_b=posterior_GJ_3929_b,
        photometry_GJ_3929_b=photometry_GJ_3929_b,
        form_GJ_3929_b=rw.Form(path='/path/to/form_GJ3929b.json'),
        posterior_LHS_1140_b=posterior_LHS_1140_b,
        photometry_LHS_1140_b=photometry_LHS_1140_b,
        form_LHS_1140_b=rw.Form(path='/path/to/form_LHS1140b.json'),
    )

    # write `submission.zip` to the same directory as this notebook:
    results.to_submission('submission.zip', overwrite=True)
