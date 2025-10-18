import custom_types as ct


def get_unit_input(parameter: ct.units.Unit, default_parameter: ct.units.Unit, param_name: str) -> ct.units.Unit:
    """Safely prompt for a unit-valued parameter without mutating the input object."""
    prompt = (
        f"Insert {param_name} [{parameter.default_unit}] "
        f"(Default: {default_parameter.value} {default_parameter.default_unit}): "
    )
    input_string = input(prompt).strip()

    if not input_string:
        # Empty input -> use default
        return type(parameter)(default_parameter.value, default_parameter.default_unit)

    try:
        value = float(input_string)
        return type(parameter)(value, parameter.default_unit)
    except ValueError:
        print(f"Invalid input '{input_string}'. Using default value.")
        return type(parameter)(default_parameter.value, default_parameter.default_unit)
