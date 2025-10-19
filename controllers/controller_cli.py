"""
controller_cli.py
=================

Provides an interactive command-line interface (CLI) for constructing and
configuring controller instances dynamically at runtime.

This module inspects the available controller classes defined in the
`controllers` package, prompts the user to select one interactively,
and initializes it. After creation, it invokes the controller's
`populate_parameters()` method to allow parameter configuration.

Example:
    >>> cli = ControllerCli()
    >>> controller = cli.make()
    >>> print(type(controller))
    <class 'controllers.bernoulli_controller.BernoulliController'>
"""

from utils.string import get_normalized_input

import controllers
from .controller_base import Controller


class ControllerCli():
    """
    Command-line interface for interactive controller selection and setup.

    The `ControllerCli` class scans all subclasses of `Controller` defined
    in the `controllers` package and allows the user to instantiate one
    interactively via terminal input.

    Attributes:
        None explicitly stored; controller creation is handled via `make()`.
    """

    def make(self) -> Controller:
        """
        Interactively create and configure a controller instance.

        This method:
            1. Discovers all available subclasses of `Controller` inside
               the `controllers` package.
            2. Prompts the user to select a controller type by name.
            3. Instantiates the selected controller.
            4. Calls its `populate_parameters()` method for user-defined
               parameter setup.
            5. Returns the fully initialized controller instance.

        Returns:
            Controller: The user-selected and configured controller instance.

        Raises:
            ValueError: If the user repeatedly enters invalid controller names.
            AttributeError: If a matching controller class cannot be found.
        """

        available = [
            name for name in dir(controllers)
            if isinstance(getattr(controllers, name), type)
            and issubclass(getattr(controllers, name), Controller)
            and getattr(controllers, name) is not Controller
        ]

        controller_options = ', '.join(available)

        normalized_classes = {cls.lower().replace(' ', ''): cls for cls in available}

        normalized_input = get_normalized_input(controller_options)

        while normalized_input not in normalized_classes:
            print(f"Invalid fiber type: {normalized_input}. "
                  "Please select from the available options.")
            normalized_input = get_normalized_input(controller_options)

        class_name = normalized_classes[normalized_input]
        controller_class = getattr(controllers, class_name)
        controller = controller_class()

        controller.populate_parameters()

        return controller
