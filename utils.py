import heapq


class Doc:
    def __init__(self, id, words) -> None:
        self.id = id
        self.words = words

    def __iter__(self):
        return iter(self.words)

    def __len__(self):
        return len(self.words)


class Query:
    def __init__(self, id, words) -> None:
        self.id = id
        self.words = words

    def __iter__(self):
        return iter(self.words)


class KMaxHeap:
    """
    Heap based structure that given a collection returns
    the k max elements in O(nlog(n)) time without storing
    all the collection in memory with O(k) space. Returns
    the min of the k max elements in O(1).

    Parameters:
    k : int -- maximun number of elements
    """

    def __init__(self, k: int):
        self.q = []  # empty min heap, min = q[0]
        self.k = k

    def push(self, item):
        if len(self.q) >= self.k:  # an item must be poped
            if item > self.q[0]:
                # min of heap is increased
                heapq.heappop(self.q)
                heapq.heappush(self.q, item)
        else:
            heapq.heappush(self.q, item)

    def pop(self):
        return heapq.heappop(self.q)

    def min(self):
        return self.q[0]

    def __len__(self):
        return len(self.q)

    def to_list(self, reverse=False):
        s = [0] * len(self.q)
        i = len(self.q) - 1 if reverse else 0
        while self.q:
            s[i] = heapq.heappop(self.q)
            i += -1 if reverse else 1
        return s
