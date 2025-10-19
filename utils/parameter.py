import custom_types as ct


def get_numeric_input(prompt: str, default: float) -> float:
    """Prompt for a numeric input, returning a float (or int) safely."""
    input_string = input(f"{prompt} [default: {default}]: ").strip()
    if not input_string:
        return default
    try:
        value = float(input_string)
        # Return int if the input looks integer-like (optional)
        return int(value) if value.is_integer() else value
    except ValueError:
        print(f"Invalid input '{input_string}'. Using default value.")
        return default


def get_unit_input(parameter: ct.units.Unit, default_parameter: ct.units.Unit, param_name: str) -> ct.units.Unit:
    """Prompt for a unit-valued parameter, using a safe numeric input."""
    value = get_numeric_input(
        f"Insert {param_name} [{parameter.default_unit}]",
        default_parameter.value
    )
    return type(parameter)(value, parameter.default_unit)
