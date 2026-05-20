import numpy as np
import pytest

from rocky_worlds_data_challenge import EclipsingSystem


def test_simple_circular():
    """Verify derived values for a circular edge-on system."""
    simple_circular = dict(
        rprs=0.01,
        per=2,
        depth_ecl=100,
        t_tra=0,
        a=10,
        inc=90,
        rs=0.5,
        omega=90,
        ecc=0,
    )

    system = EclipsingSystem(**simple_circular)

    np.testing.assert_allclose(
        [system.b_tra, system.b_ecl], 0, atol=1e-10
    )
    np.testing.assert_allclose(
        [system.ecosw, system.esinw], 0, atol=1e-10
    )
    np.testing.assert_allclose(
        system.rho_star, 3.354491, rtol=1e-5
    )


def test_simple_circular_no_depth():
    """Require eclipse depth when constructing an eclipsing system."""
    no_depth = dict(
        rprs=0.01,
        per=2,
        depth_ecl=None,
        t_tra=0,
        a=10,
        inc=90,
        rs=0.5,
        omega=90,
        ecc=0,
    )
    with pytest.raises(ValueError, match='Missing required parameter'):
        EclipsingSystem(**no_depth)


def test_eccentricity_from_esinw_at_quadrature():
    """Recover eccentricity and omega from ecosw/esinw inputs."""
    p = dict(
        rprs=0.01,
        per=2,
        depth_ecl=100,
        t_tra=0,
        a=10,
        inc=90,
        ecosw=0,
        esinw=0.1,
    )

    system = EclipsingSystem(**p)

    np.testing.assert_allclose(system.ecc, 0.1)
    np.testing.assert_allclose(system.omega, 90)


def test_eccentricity_from_sesinw_at_quadrature():
    """Recover eccentricity and omega from secosw/sesinw inputs."""
    p = dict(
        rprs=0.01,
        per=2,
        depth_ecl=100,
        t_tra=0,
        a=10,
        inc=90,
        secosw=0,
        sesinw=np.sqrt(0.1),
    )

    system = EclipsingSystem(**p)

    np.testing.assert_allclose(system.ecc, 0.1)
    np.testing.assert_allclose(system.omega, 90)
    np.testing.assert_allclose(system.ecosw, 0, atol=1e-15)
    np.testing.assert_allclose(system.esinw, 0.1)


def test_rprs_from_transit_depth():
    """Recover radius ratio from fractional transit depth."""
    system = EclipsingSystem(
        depth_tra=0.01,
        per=2,
        depth_ecl=100,
        t_tra=0,
        a=10,
        inc=90,
        omega=90,
        ecc=0,
    )

    np.testing.assert_allclose(system.rprs, 0.1)


def test_rprs_from_transit_depth_in_posteriors():
    """Recover radius ratio from a transit-depth posterior key."""
    samples = np.array([0.01, 2, 100, 0, 10, 90, 90, 0])
    parameter_keys = [
        'depth_tra',
        'per',
        'depth_ecl',
        't_tra',
        'a',
        'inc',
        'omega',
        'ecc',
    ]

    system = EclipsingSystem.from_posteriors(samples, parameter_keys)

    np.testing.assert_allclose(system.rprs, 0.1)


def test_transit_depth_must_be_fractional():
    """Reject transit depth inputs that look like percent or ppm values."""
    with pytest.raises(ValueError, match='unitless fractional transit depth'):
        EclipsingSystem(
            depth_tra=100,
            per=2,
            depth_ecl=100,
            t_tra=0,
            a=10,
            inc=90,
            omega=90,
            ecc=0,
        )


def test_a_inc_from_transit_duration_and_impact_parameter():
    """Invert transit duration and impact parameter into a/Rs and inc."""
    p = dict(
        rprs=0.05,
        per=4.2,
        depth_ecl=100,
        t_tra=0,
        a=18,
        inc=88.7,
        ecc=0.2,
        omega=np.degrees(np.arcsin(0.08 / 0.2)),
    )
    truth = EclipsingSystem(**p)

    recovered = EclipsingSystem(
        rprs=p['rprs'],
        per=p['per'],
        depth_ecl=p['depth_ecl'],
        t_tra=p['t_tra'],
        b_tra=truth.b_tra,
        dur_tra=truth.dur_tra,
        ecc=p['ecc'],
        omega=p['omega'],
    )

    np.testing.assert_allclose(recovered.a, truth.a)
    np.testing.assert_allclose(recovered.inc, truth.inc)


def test_a_inc_from_eclipse_duration_and_impact_parameter():
    """Invert eclipse duration and impact parameter into a/Rs and inc."""
    p = dict(
        rprs=0.05,
        per=4.2,
        depth_ecl=100,
        t_tra=0,
        a=18,
        inc=88.7,
        ecc=0.2,
        omega=np.degrees(np.arcsin(0.08 / 0.2)),
    )
    truth = EclipsingSystem(**p)

    recovered = EclipsingSystem(
        rprs=p['rprs'],
        per=p['per'],
        depth_ecl=p['depth_ecl'],
        t_tra=p['t_tra'],
        b_ecl=truth.b_ecl,
        dur_ecl=truth.dur_ecl,
        ecc=p['ecc'],
        omega=p['omega'],
    )

    np.testing.assert_allclose(recovered.a, truth.a)
    np.testing.assert_allclose(recovered.inc, truth.inc)


def test_trappist_1b_agol_2021():
    """Compare TRAPPIST-1 b derived values with Agol et al. 2021."""
    p = dict(
        rprs=0.08590,
        per=1.510826,
        depth_ecl=0,
        t_tra=7257.55044,
        a=None,
        inc=89.728,
        rs=0.5,
        ecosw=-0.00215,  # agol 2021
        esinw=0.00217,  # agol 2021
        rho_star=53.17,  # agol 2021
    )

    system = EclipsingSystem(**p)

    omega = np.degrees(np.arctan2(p['esinw'], p['ecosw']))
    ecc = p['ecosw'] / np.cos(np.radians(omega))
    np.testing.assert_allclose(
        [system.ecc, system.omega], [ecc, omega]
    )

    np.testing.assert_allclose(
        system.a, 20.843, rtol=1e-3  # agol 2021
    )

    np.testing.assert_allclose(
        system.dur_tra, 36.06 / 60 / 24, rtol=1e-2  # agol 2021
    )

    np.testing.assert_allclose(
        system.b_tra, 0.095, atol=4e-3  # agol 2021
    )


def test_toi771_lacedelli_2025():
    """Compare TOI-771 b derived values with Lacedelli et al. 2025."""
    p = dict(
        rprs=0.05369,
        per=2.3260155,
        depth_ecl=0,
        t_tra=1572.4191,
        inc=89.3,
        rs=0.5,
        ecc=0.005,
        omega=-130,
        rho_star=16.8,
    )

    system = EclipsingSystem(**p)
    system.validate()

    np.testing.assert_allclose(
        [system.ecosw, system.esinw],
        [
            p['ecc'] * np.cos(np.radians(p['omega'])),
            p['ecc'] * np.sin(np.radians(p['omega']))
        ],
        rtol=1e-2
    )

    np.testing.assert_allclose(
        system.a, 18.9, rtol=1e-2
    )

    np.testing.assert_allclose(
        system.dur_tra, 0.96 / 24, rtol=1e-2
    )

    np.testing.assert_allclose(
        system.b_tra, 0.22, atol=6e-2
    )


def test_trappist_1b_agol_2021_durations():
    """Recover TRAPPIST-1 b timing values from an eclipse duration."""

    p = dict(
        rprs=0.08590,
        per=1.510826,
        depth_ecl=0,
        inc=89.728,
        b_tra=0.095,
        rs=0.5,
        ecosw=-0.00215,
        esinw=0.00217,
        dur_ecl=36.06 / 60 / 24
    )

    t_tra = 7257.55044
    p['t_ecl'] = t_tra + (p['per'] / 2 + 4 / np.pi * p['ecosw'])

    system = EclipsingSystem(**p)

    omega = np.degrees(np.arctan2(p['esinw'], p['ecosw']))
    ecc = p['ecosw'] / np.cos(np.radians(omega))
    np.testing.assert_allclose(
        [system.ecc, system.omega], [ecc, omega]
    )

    np.testing.assert_allclose(
        system.a, 20.843, rtol=4e-2  # agol 2021
    )

    np.testing.assert_allclose(
        system.t_tra, t_tra, rtol=1e-5  # agol 2021
    )
