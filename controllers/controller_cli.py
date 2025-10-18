from utils.string import get_normalized_input

import controllers
from .controller_base import Controller


class ControllerCli():

    def make(self) -> Controller:

        available_controllers = [
            name for name in dir(controllers)
            if isinstance(getattr(controllers, name), type)
            and issubclass(getattr(controllers, name), Controller)
            and getattr(controllers, name) is not Controller
        ]

        controller_options = ', '.join(available_controllers)
        
        # Create a dictionary that maps the normalized version of fiber types
        normalized_fiber_classes = {cls.lower().replace(' ', ''): cls for cls in available_controllers}

        normalized_input = get_normalized_input(controller_options)

        # Check if the normalized input matches any of the available fiber types
        while normalized_input not in normalized_fiber_classes:
            print(f"Invalid fiber type: {normalized_input}. Please select from the available options.")
            normalized_input = get_normalized_input(controller_options)

        # Create an instance of the selected fiber class
        class_name = normalized_fiber_classes[normalized_input]
        controller_class = getattr(controllers, class_name)
        controller = controller_class()

        return controller
