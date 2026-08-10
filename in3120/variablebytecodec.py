# pylint: disable=missing-module-docstring
# pylint: disable=consider-using-f-string

import mmap
from struct import pack
from typing import Tuple


# Poor-man's solution since typing.Buffer doesn't currently exist.
Buffer = bytearray | mmap.mmap | memoryview


class VariableByteCodec:
    """
    A simple encoder/decoder for variable-byte codes. See Figure 5.8 in
    https://nlp.stanford.edu/IR-book/pdf/05comp.pdf for details.
    """

    @staticmethod
    def encode(number: int, destination: Buffer) -> int:
        """
        Encodes the given number, and appends the resulting bytes to the given
        destination buffer.

        Returns the number of bytes that were appended.
        """
        assert destination is not None
        assert number >= 0
        values = []
        while True:
            values.append(number % 128)
            if number < 128:
                break
            number = number // 128
        values.reverse()
        values[-1] += 128
        destination.extend(pack('%dB' % len(values), *values))
        return len(values)

    @staticmethod
    def decode(source: Buffer, start: int) -> Tuple[int, int]:
        """
        Starting at the given position in the source buffer, decodes the next number.

        Returns a pair comprised of the decoded number, and the number of bytes
        read from the source buffer.
        """
        assert source is not None
        assert start >= 0
        number = 0
        where = start
        while True:
            byte = source[where]
            where += 1
            if byte < 128:
                number = 128 * number + byte
            else:
                number = 128 * number + (byte - 128)
                return number, where - start
