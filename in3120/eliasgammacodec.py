# pylint: disable=missing-module-docstring
# pylint: disable=line-too-long

class EliasGammaCodec:
    """
    A simple encoder/decoder for Elias γ codes. See Section 5.3.2 in
    https://nlp.stanford.edu/IR-book/pdf/05comp.pdf for details.

    Note that the textbook uses 1s followed by a 0 as the unary code,
    whereas https://en.wikipedia.org/wiki/Elias_gamma_coding uses 0s
    followed by a 1. For example, the value 13 would be encoded as '1110101'
    according to the textbook, but as '0001101' according to Wikipedia.
    This implementation follows the textbook.
    """

    @staticmethod
    def encode(number: int) -> str:
        """
        Given a positive integer, returns a string of bits representing the
        integer's γ code.

        The current implementation is extremely inefficient, and is only meant for
        educational purposes. A real-world implementation would use bitwise operations
        and not string manipulations.
        """
        assert number > 0
        binary = bin(number)                # The number in binary as a string of bits, with a '0b' prefix.
        offset = binary[3:]                 # The number in binary, with the leading 1 removed.
        length = ('1' * len(offset)) + '0'  # The length of the offset, in unary code.
        return length + offset

    @staticmethod
    def decode(bits: str) -> int:
        """
        Given a string of bits representing a γ-encoded integer,
        transforms this string of bits back into the original integer.

        The current implementation is extremely inefficient, and is only meant for
        educational purposes. A real-world implementation would use bitwise operations
        and not string manipulations.
        """
        assert bits
        length = bits.index('0') + 1        # Read the unary code up to the 0 that terminates it.
        offset = bits[length:]              # The remainder, if any, is the offset.
        binary = '1' + offset               # The 1 that was chopped off in encoding is prepended.
        return int(binary, 2)
