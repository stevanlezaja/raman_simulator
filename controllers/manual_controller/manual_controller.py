import sys
import os
import time
import select

# Unix-only
import termios

import custom_types as ct
import raman_amplifier as ra

from ..controller_base import Controller


class ManualController(Controller):
    def __init__(self):
        self.power_change: list[ct.Power] = [ct.Power(0, "W")]
        self.wavelength_change: list[ct.Length] = [ct.Length(0, "m")]
        self.power_step: ct.Power = ct.Power(0.1, "W")
        self.wavelength_step: ct.Length = ct.Length(50, "nm")

    # ----------------------------
    # Cross-platform single-key reader
    # ----------------------------
    def _read_key(self, timeout: float | None = 0.2) -> str:
        """
        Return one of: 'UP','DOWN','LEFT','RIGHT','QUIT','', or a single character.
        '' means timeout/no-key.
        """
        return self._read_key_unix(timeout)

    def _enable_raw_keep_isig(self, fd):
        """
        Put terminal in a per-character mode but keep ISIG (so Ctrl+C still works).
        Returns the previous termios attrs for restoration.
        """
        old = termios.tcgetattr(fd)
        new = termios.tcgetattr(fd)

        # local flags: disable canonical mode and echo, but ensure ISIG bit is set
        new_lflag = new[3]
        new_lflag &= ~(termios.ICANON | termios.ECHO)  # turn off canonical and echo
        new_lflag |= termios.ISIG  # ensure signals (Ctrl+C, Ctrl+Z) still work
        new[3] = new_lflag

        # control chars: read returns after 1 char (VMIN=1, VTIME=0)
        new[6][termios.VMIN] = 1
        new[6][termios.VTIME] = 0

        termios.tcsetattr(fd, termios.TCSANOW, new)
        return old

    def _read_key_unix(self, timeout: float | None) -> str:
        fd = sys.stdin.fileno()
        old_attrs = None
        try:
            old_attrs = self._enable_raw_keep_isig(fd)

            # Wait with select so we can timeout and allow signals (SIGINT) to be handled
            rlist, _, _ = select.select([sys.stdin], [], [], timeout)
            if not rlist:
                return ""

            ch1 = sys.stdin.read(1)
            if ch1 == "\x1b":  # escape sequence (likely arrow)
                # try to read the rest (non-blocking because VMIN=1)
                ch2 = sys.stdin.read(1) if sys.stdin in rlist else sys.stdin.read(1)
                if ch2 == "[":
                    ch3 = sys.stdin.read(1)
                    if ch3 == "A":
                        return "UP"
                    if ch3 == "B":
                        return "DOWN"
                    if ch3 == "C":
                        return "RIGHT"
                    if ch3 == "D":
                        return "LEFT"
                return ""
            if ch1 in ("q", "Q"):
                return "QUIT"
            return ch1

        finally:
            if old_attrs is not None:
                # ALWAYS restore terminal settings
                termios.tcsetattr(fd, termios.TCSANOW, old_attrs)

    # ----------------------------
    # The controller method
    # ----------------------------
    def get_control(
        self,
        curr_input: ra.RamanInputs,
        curr_output: ra.Spectrum[ct.Power],
        target_output: ra.Spectrum[ct.Power],
    ) -> ra.RamanInputs:
        """
        Wait for one arrow key or 'q'. Ctrl+C raises KeyboardInterrupt.
        Returns a RamanInputs with step-sized changes.
        """
        try:
            while True:

                key = self._read_key(timeout=0.25)  # small timeout to remain responsive
                if key == "":
                    continue  # no key pressed yet, keep waiting
                if key == "QUIT":
                    # user chose to quit — return zero-change (or you can choose other behavior)
                    return ra.RamanInputs(
                        powers=[ct.Power(0, "W")], wavelengths=[ct.Length(0, "m")]
                    )
                if key == "UP":
                    self.power_change[0] = self.power_step
                    self.wavelength_change[0] = ct.Length(0.0, 'm')
                    break
                if key == "DOWN":
                    self.power_change[0] = -self.power_step
                    self.wavelength_change[0] = ct.Length(0.0, 'm')
                    break
                if key == "RIGHT":
                    self.power_change[0] = ct.Power(0.0, 'W')
                    self.wavelength_change[0] = self.wavelength_step
                    break
                if key == "LEFT":
                    self.power_change[0] = ct.Power(0.0, 'W')
                    self.wavelength_change[0] = -self.wavelength_step
                    break
                if key in ("s", "S"):
                    self.power_change[0] = ct.Power(0.0, 'W')
                    self.wavelength_change[0] = ct.Length(0.0, 'm')
                    break


            control = ra.RamanInputs(
                powers=self.power_change, wavelengths=self.wavelength_change
            )

            print(f"Manual Controller output: {control}")

            return control

        except KeyboardInterrupt as e:
            # Clean exit on Ctrl+C — terminal state was restored in _read_key_unix finally block,
            # but we still handle KeyboardInterrupt here and return a safe "no-change".
            print("\nManualController: interrupted (Ctrl+C). Exiting control step.")
            raise KeyboardInterrupt() from e

    def update_controller(
        self, error: ra.Spectrum[ct.Power], control_delta: ra.RamanInputs
    ) -> None:
        pass
