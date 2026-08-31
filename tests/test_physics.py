
import pytest

from phaser.utils.physics import Electron


def test_electron_200keV():
    e = Electron(200e3)

    d = {
        'energy': 200000.,             # given
        'rest_energy': 510999.06,      # from constant
        'total_energy': 710999.06,     # TE = KE + RE
        'mass': 1.267471e-30,          # TE/c^2
        'rest_mass': 9.1093837e-31,    # from constant
        'momentum': 494367.9034888,    # sqrt(KE^2 + 2*KE*RE)
        'wavelength': 0.02507934741,   # h*c/momentum
        'gamma': 1.391390152459,       # long calculation
        'beta': 0.69531442627,         # sqrt(1 - gamma^-2)
        'velocity': 208450020.93,      # beta * c
        'interaction_param': 7.2883988338e-4, # 2 pi m e lambda / h^2
    }

    for (k, v) in d.items():
        assert pytest.approx(v, rel=1e-10) == getattr(e, k)


def test_electron_0keV():
    e = Electron(0.)

    d = {
        'energy': 0.,                  # given
        'rest_energy': 510999.06,      # from constant
        'total_energy': 510999.06,     # TE = KE + RE
        'mass': 9.1093837e-31,         # TE/c^2
        'rest_mass': 9.1093837e-31,    # from constant
        'momentum': 0.,                # KE = 0
        'gamma': 1.,
        'beta': 0.,                    # v = 0
        'velocity': 0.,                # v = 0
    }

    for (k, v) in d.items():
        assert pytest.approx(v, rel=1e-10) == getattr(e, k)

    with pytest.raises(ZeroDivisionError):
        e.wavelength

    with pytest.raises(ZeroDivisionError):
        e.interaction_param


def test_electron_roundtrip():
    e1 = Electron(200e3)
    e2 = Electron.from_wavelength(e1.wavelength)

    assert pytest.approx(e1.energy) == e2.energy