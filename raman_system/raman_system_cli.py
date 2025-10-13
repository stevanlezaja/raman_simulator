import fibers as fib
import raman_amplifier as ra

from .raman_system import RamanSystem


class RamanSystemCli():
    raman_amplifier_cli = ra.RamanAmplifierCli()

    def make(self) -> RamanSystem:
        raman_system = RamanSystem()
        print("=== Making the Raman System ===")
        raman_system.fiber = fib.make()
        raman_system.raman_amplifier = self.raman_amplifier_cli.make()
        return raman_system