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