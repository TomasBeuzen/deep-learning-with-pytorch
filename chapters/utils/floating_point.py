import math
import struct
import numpy as np


def binary(number: float):
    """Print the mantissa and exponent of a float.

    Parameters
    ----------
    number : float
        A floating point number

    Returns
    -------
    None : NoneType
        Returns nothing, prints output.

    Examples
    --------
    >>> binary(1.25)
     Decimal: 1.25 x 2^0
      Binary: 1.01 x 2^0

        Sign: 0 (+)
    Mantissa: 01 (0.25)
    Exponent: 0 (0)
    """
    packed = struct.pack("!d", float(number))
    integers = [c for c in packed]
    binaries = [bin(i) for i in integers]
    stripped_binaries = [s.replace("0b", "") for s in binaries]
    padded = [s.rjust(8, "0") for s in stripped_binaries]
    final = "".join(padded)
    assert len(final) == 64, "something went wrong..."
    sign, exponent_plus_1023, mantissa = final[0], final[1:12], final[12:]
    sign_str = "" if int(sign) == 0 else "-"
    mantissa_base10 = (
        int(mantissa, 2) / 2 ** 52
    )  # shift decimal point from end of binary string to beginning of it
    mantissa_base10_str = str(mantissa_base10)[2:]  # throw away the leading "0."
    mantissa_index = mantissa.rfind("1")
    mantissa_index = 0 if mantissa_index == -1 else mantissa_index
    exponent_base10 = int(exponent_plus_1023, 2) - 1023
    print(f" Decimal: {sign_str}1.{mantissa_base10_str} x 2^{exponent_base10}")
    print(
        f"  Binary: {sign_str}1.{mantissa[:mantissa_index + 1]} x 2^{exponent_base10:b}"
    )
    print()
    print(f"    Sign: {sign} ({'+' if sign == '0' else '-'})")
    print(f"Mantissa: {mantissa[:mantissa_index + 1]} ({mantissa_base10})")
    print(f"Exponent: {exponent_base10:b} ({exponent_base10})")


def calc_spacing(number: float):
    """Calculate spacing at the given float."""
    return np.nextafter(number, 2 * number) - number


def float_rep(num):
    if not isinstance(num, float):
        raise ValueError("Please enter a floating point number.")
    sig_digits = 52 + abs(math.frexp(num)[1])
    num_str = f"{num}"
    num_str_float = f"{num:.{sig_digits}f}"
    print(f"You entered: {num}")
    if sum(int(_) for _ in num_str_float[len(num_str) :]) > 0:
        print(f"Which is inexactly stored as: {num_str_float}")
    else:
        print(f"Which is exactly stored as: {num}")