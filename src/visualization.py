import numpy as np
from glob import glob
from imageio import imread
from skimage.transform import resize
from matplotlib import pyplot as plt
from tensorflow.keras.utils import Sequence
from sklearn.model_selection import train_test_split



class DataGenerator(Sequence):
    
    'Generates data for Keras'
    def __init__(self, image_files, mask_files, batch_size = 16, size = 256, shuffle = True):
        'Initialization'
        self.image_files = image_files
        self.mask_files = mask_files
        self.batch_size = batch_size
        self.size = size
        self.shuffle = shuffle
        self.on_epoch_end()

    def __len__(self):
        'Denotes the number of batches per epoch'
        return int(np.floor(len(self.image_files) / self.batch_size))

    def __getitem__(self, index):
        'Generate one batch of data'
        # Generate indexes of the batch
        indexes = self.indexes[index * self.batch_size:(index + 1) * self.batch_size]

        # Find list of IDs
        image_files_batch = [self.image_files[k] for k in indexes]
        mask_files_batch = [self.mask_files[k] for k in indexes]

        # Generate data
        x, y = self.__data_generation(image_files_batch, mask_files_batch)

        return x, y

    def on_epoch_end(self):
        'Updates indexes after each epoch'
        self.indexes = np.arange(len(self.image_files))
        if self.shuffle == True:
            np.random.shuffle(self.indexes)

    def __data_generation(self, image_files_batch, mask_files_batch):
        'Generates data containing batch_size samples'
        # Initialization
        x, y = [], []

        # Generate data
        for i in range(self.batch_size):
            image = resize(imread(image_files_batch[i]), (self.size, self.size))
            mask = resize(imread(mask_files_batch[i]), (self.size, self.size))
            x.append(image)
            y.append(mask)
        
        x = np.array(x)
        y = np.array(y)

        return x, y



def visualize(train_image_location, train_mask_location, batch_size = 8, image_size = 512):
    datagen = DataGenerator(
        sorted(glob(train_image_location + '/*')),
        sorted(glob(train_mask_location + '/*')),
        batch_size, image_size
    )
    x_batch, y_batch = datagen.__getitem__(0)
    fig, axes = plt.subplots(nrows = 4, ncols = 2, figsize = (16, 16))
    plt.setp(axes.flat, xticks = [], yticks = [])
    c = 1
    for i, ax in enumerate(axes.flat):
        if i % 2 == 0:
            ax.imshow(x_batch[c])
            ax.set_xlabel('Image_' + str(c))
        else:
            ax.imshow(y_batch[c])
            ax.set_xlabel('Mask_' + str(c))
            c += 1
    plt.show()