import sys
import termios
import select

import custom_types as ct
import raman_amplifier as ra

from ..controller_base import Controller


class ManualController(Controller):
    def __init__(
        self,
        n_pumps: int,
        power_step: ct.Power | None = None,
        wavelength_step: ct.Length | None = None,
    ):
        super().__init__()
        self.n_pumps = n_pumps
        self.active_pump_idx = 0

        self.power_step = power_step or ct.Power(0.1, "W")
        self.wavelength_step = wavelength_step or ct.Length(5, "nm")

        self.power_change: list[ct.Power] = [
            ct.Power(0.0, "W") for _ in range(n_pumps)
        ]
        self.wavelength_change: list[ct.Length] = [
            ct.Length(0.0, "m") for _ in range(n_pumps)
        ]

    def _read_key(self, timeout: float | None = 0.2) -> str:
        return self._read_key_unix(timeout)

    def _enable_raw_keep_isig(self, fd):
        old = termios.tcgetattr(fd)
        new = termios.tcgetattr(fd)

        new_lflag = new[3]
        new_lflag &= ~(termios.ICANON | termios.ECHO)
        new_lflag |= termios.ISIG
        new[3] = new_lflag

        new[6][termios.VMIN] = 1
        new[6][termios.VTIME] = 0

        termios.tcsetattr(fd, termios.TCSANOW, new)
        return old

    def _read_key_unix(self, timeout: float | None) -> str:
        fd = sys.stdin.fileno()
        old_attrs = None
        try:
            old_attrs = self._enable_raw_keep_isig(fd)

            rlist, _, _ = select.select([sys.stdin], [], [], timeout)
            if not rlist:
                return ""

            ch1 = sys.stdin.read(1)
            if ch1 == "\x1b":
                ch2 = sys.stdin.read(1)
                if ch2 == "[":
                    ch3 = sys.stdin.read(1)
                    return {
                        "A": "UP",
                        "B": "DOWN",
                        "C": "RIGHT",
                        "D": "LEFT",
                    }.get(ch3, "")
                return ""

            if ch1 in ("q", "Q"):
                return "QUIT"

            return ch1

        finally:
            if old_attrs is not None:
                termios.tcsetattr(fd, termios.TCSANOW, old_attrs)

    def _reset_deltas(self):
        for i in range(self.n_pumps):
            self.power_change[i] = ct.Power(0.0, "W")
            self.wavelength_change[i] = ct.Length(0.0, "m")

    def get_control(
        self,
        curr_input: ra.RamanInputs,
        curr_output: ra.Spectrum[ct.Power],
        target_output: ra.Spectrum[ct.Power],
    ) -> ra.RamanInputs:

        try:
            while True:
                key = self._read_key(timeout=0.25)
                if key == "":
                    continue

                # Quit
                if key == "QUIT":
                    return ra.RamanInputs(
                        powers=[ct.Power(0.0, "W")] * self.n_pumps,
                        wavelengths=[ct.Length(0.0, "m")] * self.n_pumps,
                    )

                # Pump selection: '1' -> pump 0, '2' -> pump 1, ...
                if key.isdigit():
                    idx = int(key) - 1
                    if 0 <= idx < self.n_pumps:
                        self.active_pump_idx = idx
                        print(f"Active pump set to {idx + 1}")
                    continue

                self._reset_deltas()
                i = self.active_pump_idx

                if key == "UP":
                    self.power_change[i] = self.power_step
                    break

                if key == "DOWN":
                    self.power_change[i] = -self.power_step
                    break

                if key == "RIGHT":
                    self.wavelength_change[i] = self.wavelength_step
                    break

                if key == "LEFT":
                    self.wavelength_change[i] = -self.wavelength_step
                    break

                if key in ("s", "S"):
                    break

            control = ra.RamanInputs(
                powers=self.power_change.copy(),
                wavelengths=self.wavelength_change.copy(),
            )

            print(
                f"ManualController | pump={self.active_pump_idx + 1} | Δ={control}"
            )
            return control

        except KeyboardInterrupt as e:
            print("\nManualController: interrupted (Ctrl+C).")
            raise KeyboardInterrupt() from e

    def update_controller(
        self,
        error: ra.Spectrum[ct.Power],
        control_delta: ra.RamanInputs,
    ) -> None:
        pass
