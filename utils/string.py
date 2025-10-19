def get_normalized_input(options: str):
    type_input = input(f"Insert fiber type: \nAvailable options: [{options}]\n")

    # Normalize input: remove spaces and make it lowercase
    normalized_input = ''.join(type_input.split()).lower()

    return normalized_input