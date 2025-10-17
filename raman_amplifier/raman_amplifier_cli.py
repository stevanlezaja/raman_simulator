from .raman_amplifier import RamanAmplifier


class RamanAmplifierCli():
    PUMP_POWER_MW = 500
    PUMP_WAVELENGTH_NM = 1450
    PUMPING_RATIO = 0.5

    def make(self) -> RamanAmplifier:
        raman_amplifier = RamanAmplifier()

        wavelength_input = input(f"Insert pump wavelength [nm]: \n (Default value: {RamanAmplifierCli.PUMP_WAVELENGTH_NM} nm)")
        try:
            raman_amplifier.pump_wavelength.nm = float(wavelength_input)
        except Exception as e:
            print(f"ERROR SETTING PUMP WAVELENGTH {e}")
            raman_amplifier.pump_wavelength.nm = RamanAmplifierCli.PUMP_WAVELENGTH_NM

        power_input = input(f"Insert pump power [mW]: \n (Default value: {RamanAmplifierCli.PUMP_POWER_MW} mW)")
        try:
            raman_amplifier.pump_power.mW = float(power_input)
        except Exception as e:
            print(f"ERROR SETTING PUMP POWER {e}")
            raman_amplifier.pump_power.mW = RamanAmplifierCli.PUMP_POWER_MW

        pumping_ratio_input = input(f"Insert pumping ratio [0 - 1]: \n (Default value: {RamanAmplifierCli.PUMPING_RATIO})")
        try:
            raman_amplifier.pumping_ratio = float(pumping_ratio_input)
        except Exception as e:
            print(f"ERROR SETTING PUMP POWER {e}")
            raman_amplifier.pumping_ratio = RamanAmplifierCli.PUMPING_RATIO

        return raman_amplifier