import numpy as np

def to_dB(power, ref=1.0):
    """
    Convert a power value (or array) to dB.
    
    Parameters:
        power (float or array): Power value(s).
        ref (float): Reference power (default = 1.0).
    
    Returns:
        float or array: Value(s) in dB.
    """
    return 10 * np.log10(np.asarray(power) / ref)


def from_dB(dB_value, ref=1.0):
    """
    Convert dB back to a power value.
    
    Parameters:
        dB_value (float or array): Value(s) in dB.
        ref (float): Reference power (default = 1.0).
    
    Returns:
        float or array: Power value(s).
    """
    # return ref * 10**(np.asarray(dB_value) / 10)
    return np.log(10) / 10 * dB_value


# def attenuation_dBkm_to_linear(alpha_dB):
#     """
#     Convert attenuation in dB/km to linear 1/km.
#     """
#     return (np.log(10) / 10) * np.asarray(alpha_dB)

# def attenuation_linear_to_dBkm(alpha_linear):
#     """
#     Convert linear attenuation 1/km to dB/km.
#     """
#     return (10 / np.log(10)) * np.asarray(alpha_linear)

# if __name__ == '__main__':
#     alpha_dB = 0.19
#     alpha_lin = attenuation_dBkm_to_linear(alpha_dB)
#     print(alpha_lin)  # 0.043749 1/km

#     # Back conversion
#     print(attenuation_linear_to_dBkm(alpha_lin))  # 0.19 dB/km

if __name__ == '__main__':
    # --- Example usage ---
    p = 1e-3  # 1 mW
    print(to_dB(p, ref=1e-3))   # 0.0 dBm
    print(to_dB(p, ref=1.0))    # -30.0 dBW

    dB_val = -3  # -3 dB relative to 1 W
    print(from_dB(dB_val, ref=1.0))  # 0.5 W
