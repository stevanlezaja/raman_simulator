import custom_types as ct


class Signal:
    def __init__(self):
        self._power = ct.Power(0.1, 'mW')
        self._wavelength = ct.Length(1470, 'nm')

    @property
    def power(self):
        return self._power

    @power.setter
    def power(self, new: ct.Power | tuple[float, str]):
        if isinstance(new, ct.Power):
            self._power = new
            return
        self._power = new

    @property
    def wavelength(self):
        return self._wavelength

    @wavelength.setter
    def wavelength(self, new: ct.Length | tuple[float, str]):
        if isinstance(new, ct.Length):
            self._wavelength = new
            return
        self._wavelength = new
