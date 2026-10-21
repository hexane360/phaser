"""Physical constants & utilities."""

from dataclasses import dataclass
import math


class _Constants():
    e_rest_energy: float = 5.1099906e5
    """Electron rest energy [eV] or mass [eV/c^2]."""
    e_rest_mass: float = 9.1093837e-31
    """Electron rest mass [kg]."""
    e_spin: float = 3.2910598e-16
    """Electron spin [eV-s]."""

    h: float = 4.135667696e-18
    """Planck's constant [keV-s]."""
    c = 299792458
    """Speed of light [m/s]."""
    hc: float = 1.23984244e4
    """Planck's constant times the speed of light [eV-angstrom]."""

    e: float = 1.60217662e-19
    """Elementary charge [C]."""


C: _Constants = _Constants()
"""Physical constants."""


@dataclass(frozen=True)
class Electron:
    """Free electron with a given kinetic energy."""

    energy: float
    """Electron kinetic energy [eV]."""

    @staticmethod
    def from_wavelength(wavelength: float) -> 'Electron':
        """Return the electron with the given `wavelength` [angstrom]."""
        total_energy = math.sqrt((C.hc / wavelength)**2 + C.e_rest_energy**2)
        return Electron(total_energy - C.e_rest_energy)

    @property
    def rest_energy(self) -> float:
        r"""Electron rest energy $m_0 c^2$ [eV]."""
        return C.e_rest_energy

    @property
    def total_energy(self) -> float:
        """Total electron energy $m c^2$ [eV]."""
        return self.energy + C.e_rest_energy

    @property
    def mass(self) -> float:
        """Electron mass [kg]."""
        return self.gamma * C.e_rest_mass

    @property
    def rest_mass(self) -> float:
        """Electron rest mass [kg]."""
        return C.e_rest_mass

    @property
    def momentum(self) -> float:
        """Electron momentum times $c$ [eV]."""
        return math.sqrt(self.energy * (2*C.e_rest_energy + self.energy))

    @property
    def wavelength(self) -> float:
        """Electron wavelength [angstrom]."""
        return C.hc / self.momentum

    @property
    def gamma(self) -> float:
        r"""Electron Lorentz factor $\gamma$ [unitless]."""
        return self.energy / C.e_rest_energy + 1.

    @property
    def beta(self) -> float:
        """Electron beta factor $v/c$ [unitless]."""
        return math.sqrt(1 - self.gamma**-2)

    @property
    def velocity(self) -> float:
        """Electron velocity [m/s]."""
        return self.beta * C.c

    @property
    def interaction_param(self) -> float:
        r"""Electron interaction parameter $\sigma$ [radians/V-angstrom]."""
        m0_h2 = (C.e_rest_energy / C.hc**2)  # RM/h^2 = RE/(hc)^2 [1/(eV angstrom^2)]
        return 2*math.pi * self.wavelength * (self.gamma * m0_h2)


__all__ = [
    'C', 'Electron'
]