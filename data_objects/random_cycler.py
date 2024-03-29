import random
import pdb


"""Minimally altered code from https://github.com/CorentinJ/Real-Time-Voice-Cloning/tree/master/encoder/data_objects"""


class RandomCycler:
    """
    Creates an internal copy of a sequence and allows access to its items in a constrained random
    order. For a source sequence of n items and one or several consecutive queries of a total
    of m items, the following guarantees hold (one implies the other):
        - Each item will be returned between m // n and ((m - 1) // n) + 1 times.
        - Between two appearances of the same item, there may be at most 2 * (n - 1) other items.
    """

    def __init__(self, source):
        """
        Initialize the RandomCycler object.

        Args:
            source (iterable): The source sequence to create an internal copy from.

        Raises:
            Exception: If the source collection is empty.
        """
        if len(source) == 0:
            raise Exception(
                "Can't create RandomCycler from an empty collection")

        self.all_items = list(source)
        self.next_items = []

    def sample(self, count: int):
        """
        Sample a specified number of items from the sequence.

        Args:
            count (int): The number of items to sample.

        Returns:
            list: A list of sampled items.

        Raises:
            IndexError: If the count exceeds the number of available items.
        """
        def shuffle(l): return random.sample(l, len(l))

        out = []
        while count > 0:
            if count >= len(self.all_items):
                out.extend(shuffle(list(self.all_items)))
                count -= len(self.all_items)
                continue
            n = min(count, len(self.next_items))
            out.extend(self.next_items[:n])
            count -= n
            self.next_items = self.next_items[n:]
            if len(self.next_items) == 0:
                self.next_items = shuffle(list(self.all_items))

        return out

    def __next__(self):
        """
        Return the next sampled item from the sequence.

        Returns:
            Any: The next sampled item.
        """
        return self.sample(1)[0]
