import numpy as np


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

class data_class:
    def __init__(self, x:list(data), y = 0) -> None:
        self._data = x
        self._label = y
    def __len__(self):
        return len(self._data)
    def __int__(self)->int:
        return self._label
    def __iter__(self):
        for i in self._data:
            yield i
    def __getitem__(self, index)->data:
        return self._data[index]
    def __setitem__(self, index, value)->None:
        self._data[index] = value
    def append(self, x):
        self._data.append(x)
    def set_label(self, y = 0):
        self._label = int(y)
        return self
    def get_feature_in_matrix(self):
        return np.array([i._feature for i in self._data])
    
class data_space:
    def __init__(self) -> None:
        self._data_class:list(data_class) = []
        self._class_index = []
    def __getitem__(self, index)->data_class:
        return self._data_class[index]
    def __setitem__(self, index, value:data_class)->None:
        self._data_class[index] = value
    def get_data_class(self, label):
        return self._data_class[self._class_index.index(label)]
    def append_class(self, x:data_class):
        try:
            index = self._class_index.index(int(x))
            for i in x:
                self._data_class[index].append(i)
        except:
            self._data_class.append(x)
            self._class_index.append(x._label)
    def append_data(self, x:data):
        try:
            index = self._class_index.index(int(x))
            self._data_class[index].append(x)
        except:
            self._data_class.append(data_class([x], int(x)))
            self._class_index.append(int(x))