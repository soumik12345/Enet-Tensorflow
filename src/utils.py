import tensorflow as tf
from tensorflow.python.client import device_lib


def setup_environment():
    print('Tensorflow Version:', tf.__version__)

    local_device_protos = device_lib.list_local_devices()
    print([x.name for x in local_device_protos if x.device_type == 'GPU'])

    tf.compat.v1.disable_eager_execution()


def show_available_devices():
    local_device_protos = device_lib.list_local_devices()
    print(local_device_protos)