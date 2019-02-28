import numpy as np

def load_mnist(fp_train_images, fp_train_labels, fp_test_images, fp_test_labels):
    train_images = np.frombuffer(open(fp_train_images, 'rb').read(), offset=16, dtype=np.uint8).reshape(-1, 28 * 28)
    train_labels = np.frombuffer(open(fp_train_labels, 'rb').read(), offset=8, dtype=np.uint8)
    test_images = np.frombuffer(open(fp_test_images, 'rb').read(), offset=16, dtype=np.uint8).reshape(-1, 28 * 28)
    test_labels = np.frombuffer(open(fp_test_labels, 'rb').read(), offset=8, dtype=np.uint8)

    train_images = np.float32(train_images) / 255
    test_images = np.float32(test_images) / 255

    return train_images, np.int32(train_labels), test_images, np.int32(test_labels)
