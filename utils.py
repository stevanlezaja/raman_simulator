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
    return ref * 10**(np.asarray(dB_value) / 10)


if __name__ == '__main__':
    # --- Example usage ---
    p = 1e-3  # 1 mW
    print(to_dB(p, ref=1e-3))   # 0.0 dBm
    print(to_dB(p, ref=1.0))    # -30.0 dBW

    dB_val = -3  # -3 dB relative to 1 W
    print(from_dB(dB_val, ref=1.0))  # 0.5 W
