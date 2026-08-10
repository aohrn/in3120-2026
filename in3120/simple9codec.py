# pylint: disable=missing-module-docstring
# pylint: disable=line-too-long

from typing import List, Iterator


class Simple9Codec:
    """
    A simple encoder/decoder for Simple-9 codes. The Simple-9 compression algorithm is
    described in the paper "Inverted Index Compression Using Word-Aligned Binary Codes"
    by V. N. Anh and A. Moffat.

    The Simple-9 algorithm packs data into 32-bit words. There exists an almost identical
    algorithm called Simple-8b that packs data into 64-bit words.
    """

    # Define the bit-widths and the number of values that can fit in a 32-bit word.
    _SELECTORS = [(1, 28), (2, 14), (3, 9), (4, 7), (5, 5), (6, 4), (7, 3), (10, 2), (28, 1)]

    @staticmethod
    def encode(numbers: List[int]) -> Iterator[int]:
        """
        Encodes a list of integers into a sequence of 32-bit words using Simple-9
        compression. Each integer must lie in the range [0, 2²⁸ - 1].

        Each 32-bit word consists of 4 control bits (indicating which selector
        that was used) and 28 data bits (holding the values packed according to the
        selector.)

        The current implementation is only meant for educational purposes.
        """
        where, length = 0, len(numbers)
        while where < length:
            for i, (width, count) in enumerate(Simple9Codec._SELECTORS):
                if where + count <= length:
                    limit = (1 << width)
                    if all(0 <= numbers[j] < limit for j in range(where, where + count)):
                        yield (i << 28) | sum(numbers[where + j] << (width * j) for j in range(count))
                        where += count
                        break
            else:
                raise ValueError("Number out of range or invalid input sequence.")

    @staticmethod
    def decode(words: Iterator[int]) -> Iterator[int]:
        """
        Decodes a sequence of 32-bit words into the original list of integers.

        The current implementation is only meant for educational purposes.
        """
        for word in words:
            width, count = Simple9Codec._SELECTORS[word >> 28]
            mask = (1 << width) - 1
            yield from ((word >> (width * i)) & mask for i in range(count))
