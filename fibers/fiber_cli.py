import fibers
from .fiber import Fiber


class FiberCli():
    LENGTH_KM = 10

    def make(self) -> Fiber:
        def get_normalized_input():
            fiber_type_input = input(f"Insert fiber type: \nAvailable options: [{fiber_options}]\n")

            # Normalize input: remove spaces and make it lowercase
            normalized_input = ''.join(fiber_type_input.split()).lower()


            return normalized_input

        available_fibers = [
            name for name in dir(fibers)
            if isinstance(getattr(fibers, name), type)
            and issubclass(getattr(fibers, name), Fiber)
            and getattr(fibers, name) is not Fiber
        ]

        fiber_options = ', '.join(available_fibers)
        
        # Create a dictionary that maps the normalized version of fiber types
        normalized_fiber_classes = {cls.lower().replace(' ', ''): cls for cls in available_fibers}

        normalized_input = get_normalized_input()

        # Check if the normalized input matches any of the available fiber types
        while normalized_input not in normalized_fiber_classes:
            print(f"Invalid fiber type: {normalized_input}. Please select from the available options.")
            normalized_input = get_normalized_input()

        # Create an instance of the selected fiber class
        fiber_class_name = normalized_fiber_classes[normalized_input]
        fiber_class = getattr(fibers, fiber_class_name)
        fiber = fiber_class()
        
        length_input = input(f"Insert pump wavelength [km]: \n (Default value: {FiberCli.LENGTH_KM} km)")
        try:
            fiber.length.km = float(length_input)
        except Exception as e:
            print(f"ERROR SETTING PUMP WAVELENGTH {e}")
            fiber.length.km = FiberCli.LENGTH_KM

        return fiber