class data:
    def __init__(self, y=0) -> None:
        self._feature = []
        self._label = int(y)

    def __setitem__(self, index, value)->None:
        self._feature[index] = value

    def __getitem__(self, index)->None:
        return self._feature[index]

    def __iter__(self):
        for i in self._feature:
            yield i

    def __len__(self):
        return len(self._feature)

    def append(self, x):
        self._feature.append(float(x))

    def __int__(self) -> int:
        return self._label

    def set_label(self, y = 0):
        self._label = int(y)
        return self

    