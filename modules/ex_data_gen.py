import numpy as np
import matplotlib.pyplot as plt

class ShapeSeries:
    """
    Description:
        time-series data representing shapes
    """
    def __init__(self, length=100):
        self.shapes = {'rectangle': [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0],\
                       'triangle': [0.2, 0.4, 0.6, 0.8, 0.6, 0.4, 0.2],\
                       'M': [1.0, 0.7, 0.4, 0.1, 0.4, 0.7, 1.0],\
                       'U': [1.0, 0.1, 0.1, 0.1, 0.1, 0.1, 1.0]}
        self.label_map = {'rectangle': 0, 'triangle': 1, 'M': 2, 'U': 3}
        self.inverse_label_map = {value:key for key, value in self.label_map.items()}
        self.length = length
        self.shape_length = 7
        self.padding = 5

    def gen_shape(self, shape):
        start_index = np.random.choice(self.length - len(self.shapes.keys()) - 2*self.padding, size=1, replace=True)[0] + self.padding
        series = np.zeros(self.length)
        series[start_index: start_index+self.shape_length] = self.shapes[shape]
        return series, shape 


    def generate(self, size):
        shape_indices = np.random.choice(len(self.shapes.keys()), size=size, replace=True)
        start_indices = np.random.choice(self.length - len(self.shapes.keys()) - 2*self.padding, size=size, replace=True) + self.padding
        for j, i in enumerate(start_indices):
            series = np.zeros(self.length)
            shape = list(self.shapes.keys())[shape_indices[j]]
            series[i: i+self.shape_length] = self.shapes[shape]
            yield series, shape 

    def plot_shape(self, series):
        fig = plt.figure(figsize=(6, 6))
        ax = fig.add_subplot(111)
        nonzero = list(np.where(series > 0)[0])
        a, b = nonzero[0], nonzero[-1]
        nonzero = [a-i for i in range(self.padding+1, 1, -1)] + nonzero + [b+i for i in range(1, self.padding+1)]
        ax.stem(nonzero, series[nonzero[0]:nonzero[-1]])
        plt.show()

    def create_dataset(self, size, path, train=True):
        data = np.zeros((size, self.length))
        labels = np.zeros((size, len(self.shapes.keys())))
        for i, sample in enumerate(self.generate(size)):
            data[i, :] = sample[0]
            labels[i, self.label_map[sample[1]]] = 1.0 
        if train:
            fname = 'train'
        else:
            fname = 'test'
        np.save(path + '/{}_data.npy'.format(fname), data)
        np.save(path + '/{}_labels.npy'.format(fname), labels)

    def dist_to_label(self, dist):
        index = np.argmax(dist)
        return self.inverse_label_map[index]
