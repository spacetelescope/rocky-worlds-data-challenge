from dataclasses import dataclass, fields

import astropy.units as u
import numpy as np
from astropy import constants
from numpy.typing import ArrayLike

__all__ = ['EclipsingSystem']


_RPRS_ALIASES = {
    'rp_rs': 'rprs',
    'rp/rs': 'rprs',
    'rp/rstar': 'rprs',
    'k': 'rprs',
    'radius_ratio': 'rprs',
}
_TRANSIT_DEPTH_ALIASES = {
    'depth_tra',
    'transit_depth',
}


@dataclass(init=False)
class EclipsingSystem:
    """Infer and validate eclipse-system parameters from posterior samples.

    The class accepts several equivalent orbital parameterizations and fills
    in derived values such as eccentricity, impact parameters, conjunction
    times, durations, and stellar density where the supplied inputs make those
    conversions possible.

    The canonical planet-to-star radius ratio parameter is ``rprs``. The
    aliases ``rp_rs``, ``rp/rs``, ``rp/rstar``, ``k``, and
    ``radius_ratio`` are normalized to ``rprs``. A fractional transit depth
    supplied as ``depth_tra`` or ``transit_depth`` is converted to
    ``rprs = sqrt(depth_tra)``.

    Attributes
    ----------
    rprs : array-like or float
        Planet-to-star radius ratio.
    per : array-like or float
        Orbital period in days.
    depth_ecl : array-like or float
        Eclipse depth in parts per million. This value is required.
    t_ecl : array-like or float
        Mid-eclipse time in BMJD_TDB.
    t_tra : array-like or float
        Mid-transit time in BMJD_TDB.
    a : array-like or float
        Semimajor axis in units of stellar radii.
    inc : array-like or float
        Planetary orbital inclination in degrees.
    b_tra : array-like or float
        Impact parameter at transit in units of stellar radii.
    b_ecl : array-like or float
        Impact parameter at eclipse in units of stellar radii.
    rs : array-like or float
        Stellar radius in units of solar radii.
    omega : array-like or float
        Longitude of periastron in degrees.
    ecc : array-like or float
        Orbital eccentricity.
    ecosw : array-like or float
        Eccentricity times the cosine of longitude of periastron.
    esinw : array-like or float
        Eccentricity times the sine of longitude of periastron.
    secosw : array-like or float
        Square root of eccentricity times the cosine of longitude of
        periastron.
    sesinw : array-like or float
        Square root of eccentricity times the sine of longitude of
        periastron.
    rho_star : array-like or float
        Stellar density in units of the solar density.
    dur_tra : array-like or float
        Transit duration from first through fourth contact, in days.
    dur_ecl : array-like or float
        Eclipse duration from first through fourth contact, in days.
    """

    #: Planet-to-star radius ratio.
    rprs: ArrayLike | float = None

    #: Orbital period in days.
    per: ArrayLike | float = None

    #: Eclipse depth in parts per million.
    depth_ecl: ArrayLike | float = None

    #: Mid-eclipse time in BMJD_TDB.
    t_ecl: ArrayLike | float = None

    #: Mid-transit time in BMJD_TDB.
    t_tra: ArrayLike | float = None

    #: Semimajor axis in units of stellar radii.
    a: ArrayLike | float = None

    #: Planetary orbital inclination in degrees.
    inc: ArrayLike | float = None

    #: Impact parameter at transit in units of stellar radii.
    b_tra: ArrayLike | float = None

    #: Impact parameter at eclipse in units of stellar radii.
    b_ecl: ArrayLike | float = None

    #: Stellar radius in units of solar radii.
    rs: ArrayLike | float = None

    #: Longitude of periastron in degrees.
    omega: ArrayLike | float = None

    #: Orbital eccentricity.
    ecc: ArrayLike | float = None

    #: Eccentricity times the cosine of longitude of periastron.
    ecosw: ArrayLike | float = None

    #: Eccentricity times the sine of longitude of periastron.
    esinw: ArrayLike | float = None

    #: Square root of eccentricity times cosine of longitude of periastron.
    secosw: ArrayLike | float = None

    #: Square root of eccentricity times sine of longitude of periastron.
    sesinw: ArrayLike | float = None

    #: Stellar density in units of the solar density.
    rho_star: ArrayLike | float = None

    #: Transit duration from first through fourth contact, in days.
    dur_tra: ArrayLike | float = None

    #: Eclipse duration from first through fourth contact, in days.
    dur_ecl: ArrayLike | float = None

    def __init__(self, rprs: ArrayLike | float = None,
                 per: ArrayLike | float = None,
                 depth_ecl: ArrayLike | float = None,
                 t_ecl: ArrayLike | float = None,
                 t_tra: ArrayLike | float = None,
                 a: ArrayLike | float = None,
                 inc: ArrayLike | float = None,
                 b_tra: ArrayLike | float = None,
                 b_ecl: ArrayLike | float = None,
                 rs: ArrayLike | float = None,
                 omega: ArrayLike | float = None,
                 ecc: ArrayLike | float = None,
                 ecosw: ArrayLike | float = None,
                 esinw: ArrayLike | float = None,
                 secosw: ArrayLike | float = None,
                 sesinw: ArrayLike | float = None,
                 rho_star: ArrayLike | float = None,
                 dur_tra: ArrayLike | float = None,
                 dur_ecl: ArrayLike | float = None,
                 **parameter_aliases):
        """Initialize an eclipsing-system parameter set.

        Parameters are the canonical dataclass fields documented above.
        Additional keyword arguments may include supported aliases, such as
        ``rp_rs``, ``k``, or a fractional ``transit_depth``.
        """
        user_input = {
            'rprs': rprs,
            'per': per,
            'depth_ecl': depth_ecl,
            't_ecl': t_ecl,
            't_tra': t_tra,
            'a': a,
            'inc': inc,
            'b_tra': b_tra,
            'b_ecl': b_ecl,
            'rs': rs,
            'omega': omega,
            'ecc': ecc,
            'ecosw': ecosw,
            'esinw': esinw,
            'secosw': secosw,
            'sesinw': sesinw,
            'rho_star': rho_star,
            'dur_tra': dur_tra,
            'dur_ecl': dur_ecl,
            **parameter_aliases,
        }
        user_input = self._normalize_parameter_aliases(user_input)

        valid_fields = {field.name for field in fields(self)}
        unexpected = set(user_input) - valid_fields
        if unexpected:
            raise TypeError(
                "Unexpected EclipsingSystem parameter(s): "
                f"{sorted(unexpected)}."
            )

        for field in fields(self):
            setattr(self, field.name, user_input.get(field.name))

        self.__post_init__()

    def __post_init__(self):
        """Validate and complete the parameter set after initialization.

        Raises
        ------
        ValueError
            If the supplied parameters are insufficient or inconsistent.
        """
        self.validate()

    @classmethod
    def _normalize_parameter_aliases(cls, user_input):
        """Return parameters with supported aliases mapped to canonical names.

        Parameters
        ----------
        user_input : dict
            Parameter-value mapping supplied by a user or posterior table.

        Returns
        -------
        dict
            Parameter-value mapping using canonical ``EclipsingSystem`` names.

        Raises
        ------
        ValueError
            If a supplied transit-depth alias is invalid or if an alias
            conflicts with an explicitly supplied canonical value.
        """
        normalized = dict(user_input)

        def set_canonical(canonical_key, value, alias_key):
            existing = normalized.get(canonical_key)
            if existing is None:
                normalized[canonical_key] = value
                return

            try:
                matching = np.allclose(existing, value, equal_nan=True)
            except (TypeError, ValueError):
                matching = existing == value

            if not matching:
                raise ValueError(
                    f"Parameter alias '{alias_key}' conflicts with "
                    f"canonical parameter '{canonical_key}'."
                )

        for alias_key, canonical_key in _RPRS_ALIASES.items():
            if alias_key in normalized:
                set_canonical(canonical_key, normalized.pop(alias_key),
                              alias_key)

        for alias_key in _TRANSIT_DEPTH_ALIASES:
            if alias_key not in normalized:
                continue

            depth = np.asarray(normalized.pop(alias_key), dtype=float)
            if np.any(~np.isfinite(depth)) or np.any(depth < 0):
                raise ValueError(
                    f"Transit-depth alias '{alias_key}' must contain finite, "
                    "non-negative fractional transit depths."
                )
            if np.any(depth > 1):
                raise ValueError(
                    f"Transit-depth alias '{alias_key}' is interpreted as a "
                    "unitless fractional transit depth. Please use values "
                    "between 0 and 1, not percent or ppm."
                )

            rprs = np.sqrt(depth)
            if rprs.ndim == 0:
                rprs = rprs.item()
            set_canonical('rprs', rprs, alias_key)

        return normalized

    def get_ecc_omega(self):
        """Return eccentricity and longitude of periastron.

        The method accepts any one of three equivalent parameterizations:
        ``ecc``/``omega``, ``ecosw``/``esinw``, or ``secosw``/``sesinw``.
        Missing companion eccentricity parameters are filled in place when
        possible.

        Returns
        -------
        ecc : array-like or float
            Orbital eccentricity.
        omega : array-like or float
            Longitude of periastron in degrees.

        Raises
        ------
        ValueError
            If no supported eccentricity/orientation parameterization has
            been provided.
        """

        if self.ecc is not None and self.omega is not None:
            self.ecosw = self.ecc * np.cos(np.radians(self.omega))
            self.esinw = self.ecc * np.sin(np.radians(self.omega))
            return self.ecc, self.omega

        elif self.ecosw is not None and self.esinw is not None:
            omega_rad = np.arctan2(self.esinw, self.ecosw)
            self.omega = np.degrees(omega_rad)
            self.ecc = np.hypot(self.ecosw, self.esinw)
            return self.ecc, self.omega

        elif self.secosw is not None and self.sesinw is not None:
            omega_rad = np.arctan2(self.sesinw, self.secosw)
            self.omega = np.degrees(omega_rad)
            self.ecc = self.secosw**2 + self.sesinw**2

            # do a second pass to calculate ecosw, esinw:
            self.get_ecc_omega()

            return self.ecc, self.omega

        else:
            raise ValueError(
                "Must define one of: {ecc, omega}, "
                "{ecosw, esinw}, or {secosw, sesinw} (read as: 'sqrt(e) "
                "times the co/sine of omega')."
            )

    def get_b(self):
        """Return the transit and eclipse impact parameters.

        Returns
        -------
        b_tra : array-like or float
            Transit impact parameter in units of stellar radii.
        b_ecl : array-like or float
            Eclipse impact parameter in units of stellar radii.
        """

        if self.a is None or self.inc is None:
            self.get_a_inc()

        if self.b_tra is None:
            self.b_tra = (
                self.a * np.cos(np.radians(self.inc)) *
                (1 - self.ecc ** 2) / (1 + self.esinw)
            )
        if self.b_ecl is None:
            self.b_ecl = (
                self.a * np.cos(np.radians(self.inc)) *
                (1 - self.ecc ** 2) / (1 - self.esinw)
            )

        return self.b_tra, self.b_ecl

    def get_a_inc(self):
        """Return semimajor axis in stellar radii and inclination.

        The inclination is returned in degrees.

        Returns
        -------
        a : array-like or float
            Semimajor axis in units of stellar radii.
        inc : array-like or float
            Orbital inclination in degrees.
        """

        if self.ecc is None or self.esinw is None:
            self.get_ecc_omega()

        def solve_a_inc_from_duration(b, duration, sign):
            """Infer ``a/Rs`` and optionally inclination from a duration."""
            impact_factor = (1 - self.ecc**2) / (1 + sign * self.esinw)
            duration_factor = np.sqrt(1 - self.ecc**2) / (
                1 + sign * self.esinw
            )
            chord = np.sqrt((1 + self.rprs)**2 - b**2)
            sin_arg = duration * np.pi / self.per / duration_factor

            if self.inc is None:
                a_sin_i = chord / np.sin(sin_arg)
                a_cos_i = b / impact_factor
                self.a = np.hypot(a_sin_i, a_cos_i)
                self.inc = np.degrees(np.arctan2(a_sin_i, a_cos_i))
            else:
                self.a = (
                    chord /
                    (
                        np.sin(np.radians(self.inc)) *
                        np.sin(sin_arg)
                    )
                )

        if (self.a is None and self.dur_tra is not None and
                self.b_tra is not None):
            # Invert Winn (2011) eqns. 14 and 16. If inc is unknown, combine
            # the duration and impact-parameter equations to solve a and inc.
            solve_a_inc_from_duration(self.b_tra, self.dur_tra, sign=1)

        if (self.a is None and self.dur_ecl is not None and
                self.b_ecl is not None):
            # Eclipse uses the opposite sign for the eccentricity correction.
            solve_a_inc_from_duration(self.b_ecl, self.dur_ecl, sign=-1)

        if (self.a is None and self.b_tra is not None and
                self.inc is not None):
            self.a = self.b_tra / np.cos(np.radians(self.inc)) / (
                (1 - self.ecc ** 2) / (1 + self.esinw)
            )

        if (self.a is None and self.b_ecl is not None and
                self.inc is not None):
            self.a = self.b_ecl / np.cos(np.radians(self.inc)) / (
                (1 - self.ecc ** 2) / (1 - self.esinw)
            )

        if self.a is None and self.rho_star is not None:
            rho_sun = (1 * u.M_sun) / (4/3 * np.pi * u.R_sun ** 3)

            self.a = (
                self.rho_star * rho_sun *
                (constants.G * (self.per * u.d) ** 2) / (3 * np.pi)
            ).to_value(u.dimensionless_unscaled) ** (1 / 3)

        if self.inc is None and self.b_tra is not None:
            self.inc = np.degrees(np.arccos(
                self.b_tra / self.a / (
                    (1 - self.ecc ** 2) / (1 + self.esinw)
                )
            ))

        if self.inc is None and self.b_ecl is not None:
            self.inc = np.degrees(np.arccos(
                self.b_ecl / self.a / (
                    (1 - self.ecc ** 2) / (1 - self.esinw)
                )
            ))

        if self.a is not None and self.rho_star is None:
            rho_sun = (1 * u.M_sun) / (4/3 * np.pi * u.R_sun ** 3)

            self.rho_star = (
                (3 * np.pi) / (constants.G * (self.per * u.d) ** 2) *
                self.a ** 3 / rho_sun
            ).to_value(u.dimensionless_unscaled)
        return self.a, self.inc

    def get_conjunction_times(self):
        """Return the mid-transit and mid-eclipse times.

        Returns
        -------
        t_tra : array-like or float
            Mid-transit time in BMJD_TDB.
        t_ecl : array-like or float
            Mid-eclipse time in BMJD_TDB.
        """

        if self.ecosw is None:
            self.get_ecc_omega()

        # winn 2011 eqn 33
        dt_conj = self.per / 2 * (1 + 4 / np.pi * self.ecosw)

        if self.t_ecl is None:
            self.t_ecl = self.t_tra + dt_conj

        if self.t_tra is None:
            self.t_tra = self.t_ecl - dt_conj

        return self.t_tra, self.t_ecl

    def get_durations(self):
        """Return first-to-fourth-contact transit and eclipse durations.

        Returns
        -------
        dur_tra : array-like or float
            Transit duration in days.
        dur_ecl : array-like or float
            Eclipse duration in days.
        """

        if self.b_tra is None:
            self.get_b()

        if self.t_tra is None:
            self.get_conjunction_times()

        if self.ecc is None or self.esinw is None:
            self.get_ecc_omega()

        if self.inc is None:
            self.get_a_inc()

        def circular_duration(b):
            """Return the circular-orbit duration for one impact parameter."""
            return (
                self.per / np.pi *
                np.arcsin(
                    np.sqrt((1 + self.rprs)**2 - b**2) /
                    (self.a * np.sin(np.radians(self.inc)))
                )
            )

        # winn 2011 eqn 14
        transit_dur_14_circular = circular_duration(self.b_tra)
        eclipse_dur_14_circular = circular_duration(self.b_ecl)

        # winn 2011 eqn 16
        if self.dur_tra is None:
            ecc_term_transit = np.sqrt(1 - self.ecc**2) / (1 + self.esinw)
            self.dur_tra = (
                transit_dur_14_circular * ecc_term_transit
            )

        if self.dur_ecl is None:
            ecc_term_eclipse = np.sqrt(1 - self.ecc**2) / (1 - self.esinw)
            self.dur_ecl = (
                eclipse_dur_14_circular * ecc_term_eclipse
            )

        return self.dur_tra, self.dur_ecl

    def validate(self):
        """Check required inputs and derive all supported dependent values.

        Raises
        ------
        ValueError
            If ``depth_ecl`` is missing or if required orbital quantities
            cannot be derived from the supplied parameterization.
        """

        if self.depth_ecl is None:
            raise ValueError(
                "Missing required parameter: "
                "eclipse depth in ppm (`depth_ecl`)."
            )
        self.get_a_inc()
        self.get_ecc_omega()
        self.get_b()
        self.get_conjunction_times()
        self.get_durations()

    @classmethod
    def from_posteriors(cls, samples, parameter_keys):
        """Build an eclipsing system from one posterior draw and its keys.

        Parameters
        ----------
        samples : array-like
            One posterior draw containing one value for each parameter.
        parameter_keys : array-like
            Parameter names corresponding to ``samples``.

        Returns
        -------
        EclipsingSystem
            System initialized with the supplied posterior values.
        """
        user_input = {
            str(k): v
            for k, v in zip(parameter_keys, samples)
        }
        user_input = cls._normalize_parameter_aliases(user_input)
        return cls(**user_input)
