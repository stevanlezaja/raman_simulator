"""
Class representing a control loop for a Raman amplification system.

The ControlLoop coordinates a RamanSystem and a controller to iteratively
adjust the Raman amplifier inputs in order to reach a desired target output
spectrum. It maintains the current control inputs and the latest output
spectrum at each step.
"""

import numpy as np
import matplotlib.axes
from typing import Optional, Any
import logging

import raman_amplifier as ra
import raman_system as rs
import controllers as ctrl
import custom_types as ct


log = logging.Logger("[Control Loop]", level=logging.INFO)


class ControlLoop:
    """
    Represents a feedback control loop for a Raman amplification system.

    The ControlLoop manages the interaction between a RamanSystem and a controller,
    iteratively adjusting the amplifier inputs to achieve a desired target output
    spectrum. It tracks the current control inputs and the most recent output spectrum.

    Attributes
    ----------
    raman_system : rs.RamanSystem
        The RamanSystem instance being controlled.
    controller : ctrl.Controller
        The controller responsible for computing the control inputs.
    target : Optional[ra.Spectrum[ct.Power]]
        Desired output spectrum for the control loop.
    curr_control : ra.RamanInputs
        Current control inputs applied to the Raman amplifier.
    curr_output : Optional[ra.Spectrum[ct.Power]]
        Most recent output spectrum observed from the Raman system.
    """

    manual_controller = ctrl.ManualController

    def __init__(self, raman_system: rs.RamanSystem, controller: ctrl.Controller):
        """
        Initialize a ControlLoop instance.

        Parameters
        ----------
        raman_system : rs.RamanSystem
            The RamanSystem instance to be controlled.
        controller : ctrl.Controller
            The controller object responsible for generating control inputs.

        Attributes
        ----------
        target : Optional[ra.Spectrum[ct.Power]]
            Desired output spectrum for the control loop.
        curr_control : ra.RamanInputs
            Current control inputs applied to the Raman amplifier.
        curr_output : Optional[ra.Spectrum[ct.Power]]
            Most recent output spectrum observed from the Raman system.
        """

        self.raman_system = raman_system
        self.controller = controller
        self.target: Optional[ra.Spectrum[ct.Power]] = None
        self.curr_control: ra.RamanInputs = ra.RamanInputs(n_pumps=1)
        self.curr_output: Optional[ra.Spectrum[ct.Power]] = None
        self.history: dict[str, list[Any]] = {'RamanInputs': [], 'powers': [], 'wavelengths': []}

    def set_target(self, target: ra.Spectrum[ct.Power]):
        """
        Set the desired target output spectrum for the control loop.

        Parameters
        ----------
        target : ra.Spectrum[ct.Power]
            The desired output spectrum that the controller will try to achieve.
        """

        self.target = target

    def get_raman_output(self) -> ra.Spectrum[ct.Power]:
        """
        Run the Raman system simulation and return the current output spectrum.

        This method updates the RamanSystem and retrieves the resulting output
        spectrum based on the current amplifier inputs.

        Returns
        -------
        ra.Spectrum[ct.Power]
            The current output spectrum of the Raman system.
        """

        log.debug("Current inputs %s", self.curr_control)
        self.raman_system.update()
        return self.raman_system.output_spectrum

    def get_control(self) -> ra.RamanInputs:
        """
        Compute the control inputs based on the current output and target spectrum.

        Returns
        -------
        ra.RamanInputs
            The control inputs to apply to the Raman amplifier.

        Notes
        -----
        - If `target` is not set, returns default (zero) RamanInputs.
        - Requires `curr_output` to be set before calling.
        """

        assert self.curr_output is not None
        if self.target is None:
            return ra.RamanInputs()
        control = self.controller.get_control(curr_input=self.curr_control,
                                              curr_output=self.curr_output,
                                              target_output=self.target)
        self.history['RamanInputs'].append(control)
        self.history['powers'].append([p.W for p in self.curr_control.powers])
        self.history['wavelengths'].append([w.nm for w in self.curr_control.wavelengths])
        return control

    def apply_control(self):
        """
        Apply the current control inputs to the Raman amplifier.

        Updates the amplifier's pump power and wavelength based on `curr_control`.
        Logs the old and new pump power values.
        """
        self.curr_control.clamp_values()

        log.debug("Old pump power: %s", self.raman_system.raman_amplifier.pump_power)
        self.raman_system.raman_amplifier.pump_power = self.curr_control.powers[0]
        log.debug("New pump power: %s", self.raman_system.raman_amplifier.pump_power)

        log.debug("Old pump wavelength: %s", self.raman_system.raman_amplifier.pump_wavelength)
        self.raman_system.raman_amplifier.pump_wavelength = self.curr_control.wavelengths[0]
        log.debug("New pump wavelength: %s", self.raman_system.raman_amplifier.pump_wavelength)

    def step(self):
        """
        Perform a single control loop iteration.

        Steps:
            1. Retrieve the current Raman output spectrum.
            2. Compute the control inputs needed to reach the target spectrum.
            3. Apply the control inputs to the Raman amplifier.
        
        Updates `curr_output` and `curr_control` attributes accordingly.
        """

        self.curr_output = self.get_raman_output()
        self.curr_control += self.get_control()
        if isinstance(self.controller, ctrl.BernoulliController):
            assert self.target is not None
            self.controller.update_controller(self.curr_output - self.target, self.curr_control)
        self.apply_control()

    @property
    def is_valid(self) -> bool:
        """
        Returns true of control loop is valid (correctly initialized)
        """
        valid = self.controller.is_valid ^ self.raman_system.is_valid
        return valid

    def plot_loss(self, ax: matplotlib.axes.Axes) -> None:
        if hasattr(self.controller, 'plot_loss') and callable(self.controller.plot_loss):  # type: ignore
            self.controller.plot_loss(ax)  # type: ignore
            return
        ax.plot(ra.mse(self.curr_output, self.target))  # type: ignore
        ax.set_xlabel("Iteration")  # type: ignore
        ax.set_ylabel("MSE")  # type: ignore
        ax.set_title("MSE over time")  # type: ignore
        ax.grid()  # type: ignore
        ax.legend()  # type: ignore

    def plot_spectrums(self, ax: matplotlib.axes.Axes):
        assert self.target is not None
        assert self.curr_output is not None
        ax.plot( # type: ignore
            [f.Hz for f in self.target.frequencies],
            [val.value for val in self.target.values],
            label="Target",
        )
        ax.plot( # type: ignore
            [f.Hz for f in self.curr_output.frequencies],
            [val.value for val in self.curr_output.values],
            label="Current Output",
        )
        ax.set_xlabel("Frequency (Hz)")  # type: ignore
        ax.set_ylabel("Power (mW)")  # type: ignore
        ax.set_title("Target vs Current Output Spectrum")  # type: ignore
        ax.grid()  # type: ignore
        ax.legend()  # type: ignore

    def plot_parameter_2d(self, ax: matplotlib.axes.Axes):
        power_arr = np.array(self.history['powers'])
        wl_arr = np.array(self.history['wavelengths'])

        ax.plot(power_arr, wl_arr)  # type: ignore
        ax.scatter(power_arr[-1], wl_arr[-1], label="Current")  # type: ignore
        ax.scatter(power_arr[0], wl_arr[0], label="Initial")  # type: ignore
        ax.set_xlabel("Power [W]")  # type: ignore
        ax.set_ylabel("Wavelength [nm]")  # type: ignore
        ax.set_ylim(bottom=1420, top=1490)
        ax.set_xlim(left=0.0, right=1.0)
        ax.set_title("Wavelength step probability evolution")  # type: ignore
        ax.grid()  # type: ignore
        ax.legend()  # type: ignore

    def plot_power_evolution(self, ax: matplotlib.axes.Axes):
        power_arr = np.array(self.history['powers'])
        for i in range(power_arr.shape[1]):
            ax.plot(power_arr[::-1, i], range(len(power_arr[:, i])), label=f"Power {i}")  # type: ignore
        ax.set_xlabel("Iteration")  # type: ignore
        ax.set_ylabel("Power (W)")  # type: ignore
        ax.set_xlim(left=0.0, right=1.0)
        ax.set_title("Power evolution")  # type: ignore
        ax.grid()  # type: ignore
        ax.legend()  # type: ignore

    def plot_wavelength_evolution(self, ax: matplotlib.axes.Axes):
        wl_arr = np.array(self.history['wavelengths'])
        for i in range(wl_arr.shape[1]):
            ax.plot(wl_arr[:, i], label=f"Wavelength {i}")  # type: ignore
        ax.set_xlabel("Iteration")  # type: ignore
        ax.set_ylabel("Wavelength (nm)")  # type: ignore
        ax.set_ylim(bottom=1420, top=1490)
        ax.set_title("Wavelength evolution")  # type: ignore
        ax.grid()  # type: ignore
        ax.legend()  # type: ignore
