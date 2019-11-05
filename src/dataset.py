from config import *
import tensorflow as tf
from matplotlib import pyplot as plt


def load_data(image_path, mask_path):
    '''Read Image and Corresponding Mask
    Params:
        image_path  -> Path to image
        mask_path   -> Path to mask
    '''
    # Reading Image
    image = tf.io.read_file(image_path)
    image = tf.image.decode_png(image, channels = 3)
    image = tf.image.random_hue(image, 0.1, seed = None)
    image = tf.cast(image, dtype = tf.float32)
    image = tf.image.resize(images = image, size = [IMAGE_SIZE, IMAGE_SIZE])
    image = (image - 127.5) / 127.5
    # Reading Mask
    mask = tf.io.read_file(mask_path)
    mask = tf.image.decode_png(mask, channels = 3)
    mask = tf.cast(mask, dtype = tf.uint8)
    mask = tf.image.resize(images = mask, size = [IMAGE_SIZE, IMAGE_SIZE])
    return image, mask


def get_datasets(x_train, y_train, x_val, y_val, batch_size=8):
    '''Get Training and Validation Datasets
    Params:
        x_train     -> List of paths of training images
        y_train     -> List of paths of training masks
        x_val       -> List of paths of validation images
        y_val       -> List of paths of validation masks
        batch_size  -> Batch Size

    '''
    # Train Dataset
    train_dataset = tf.data.Dataset.from_tensor_slices((x_train, y_train))
    train_dataset = train_dataset.apply(
        tf.data.experimental.map_and_batch(
            map_func = load_data, batch_size = batch_size,
            num_parallel_calls = tf.data.experimental.AUTOTUNE,
            drop_remainder = True
        )
    )
    train_dataset = train_dataset.repeat()
    train_dataset = train_dataset.prefetch(tf.data.experimental.AUTOTUNE)
    
    # Validation Dataset
    val_dataset = tf.data.Dataset.from_tensor_slices((x_val, y_val))
    val_dataset = val_dataset.apply(
        tf.data.experimental.map_and_batch(
            map_func = load_data, batch_size = batch_size,
            num_parallel_calls = tf.data.experimental.AUTOTUNE,
            drop_remainder = True
        )
    )
    val_dataset = val_dataset.repeat()
    val_dataset = val_dataset.prefetch(tf.data.experimental.AUTOTUNE)
    
    return train_dataset, val_dataset


def visualize(dataset):
    '''Dataset Visualization Function
    Params:
        dataset    -> Dataset
    '''
    for x, y in train_dataset.take(1):
        x_batch = x.numpy()
        y_batch = y.numpy()
        fig, axes = plt.subplots(nrows = 4, ncols = 2, figsize = (16, 16))
        plt.setp(axes.flat, xticks = [], yticks = [])
        c = 1
        for i, ax in enumerate(axes.flat):
            if i % 2 == 0:
                ax.imshow(x_batch[c] * 127.5 + 127.5)
                ax.set_xlabel('Image_' + str(c))
            else:
                ax.imshow(y_batch[c])
                ax.set_xlabel('Mask_' + str(c))
                c += 1
        plt.show()