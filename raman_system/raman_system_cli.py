import fibers as fib
import raman_amplifier as ra

from .raman_system import RamanSystem


class RamanSystemCli():
    raman_amplifier_cli = ra.RamanAmplifierCli()

    def make(self) -> RamanSystem:
        raman_system = RamanSystem()
        print("=== Making the Raman System ===")
        fiber_cli = fib.FiberCli()
        raman_system.fiber = fiber_cli.make()
        raman_system.raman_amplifier = self.raman_amplifier_cli.make()
        return raman_system