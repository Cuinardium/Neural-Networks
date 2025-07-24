import cv2
import numpy as np
import os

# --------------------- Supervised ---------------------


def load_shapes_dataset(img_size=200, train_ratio=0.8, samples_per_class=100):
    folders = ["data/shapes/square", "data/shapes/triangle"]
    labels = []
    images = []

    for folder in folders:
        for path in os.listdir(os.getcwd() + "/" + folder):
            img = cv2.imread(folder + "/" + path, 0)
            images.append(cv2.resize(img, (img_size, img_size)))
            labels.append(folders.index(folder))

    # Shuffle data
    data = list(zip(images, labels))
    np.random.shuffle(data)
    images = [d[0] for d in data]
    labels = [d[1] for d in data]

    # Reduce amount of data
    squares = []
    triangles = []
    for image, label in zip(images, labels):
        if label == 0:
            squares.append(image)
        else:
            triangles.append(image)
    images = squares[:samples_per_class] + triangles[:samples_per_class]
    labels = [0] * samples_per_class + [1] * samples_per_class

    sqare_qty = labels.count(0)
    triangle_qty = labels.count(1)
    # Separate data into training sets and testing sets
    train_qty = int(len(images) * train_ratio)
    square_train_qty = int(train_qty * sqare_qty / (sqare_qty + triangle_qty))
    triangle_train_qty = train_qty - square_train_qty

    print(f"There are {sqare_qty} squares and {triangle_qty} triangles")

    train_images = (
        images[:square_train_qty]
        + images[samples_per_class : samples_per_class + triangle_train_qty]
    )
    train_labels = (
        labels[:square_train_qty]
        + labels[samples_per_class : samples_per_class + triangle_train_qty]
    )
    test_images = (
        images[square_train_qty:samples_per_class]
        + images[samples_per_class + triangle_train_qty :]
    )
    test_labels = (
        labels[square_train_qty:samples_per_class]
        + labels[samples_per_class + triangle_train_qty :]
    )

    print(
        f"There are {len(train_images)} training images and {len(test_images)} testing images"
    )

    # Training set first
    train_images = np.array(train_images)
    train_images = train_images.astype("float32")
    train_images /= 255

    # Testing set second
    test_images = np.array(test_images)
    test_images = test_images.astype("float32")
    test_images /= 255

    train_labels = np.array(train_labels)
    test_labels = np.array(test_labels)

    return train_images, train_labels, test_images, test_labels


# TODO: parametrize noise
def load_digits_dataset():
    digits = [
        [
            [0.0, 1.0, 1.0, 1.0, 0.0],
            [1.0, 0.0, 0.0, 0.0, 1.0],
            [1.0, 0.0, 0.0, 1.0, 1.0],
            [1.0, 0.0, 1.0, 0.0, 1.0],
            [1.0, 1.0, 0.0, 0.0, 1.0],
            [1.0, 0.0, 0.0, 0.0, 1.0],
            [0.0, 1.0, 1.0, 1.0, 0.0],
        ],
        [
            [0.0, 0.0, 1.0, 0.0, 0.0],
            [0.0, 1.0, 1.0, 0.0, 0.0],
            [0.0, 0.0, 1.0, 0.0, 0.0],
            [0.0, 0.0, 1.0, 0.0, 0.0],
            [0.0, 0.0, 1.0, 0.0, 0.0],
            [0.0, 0.0, 1.0, 0.0, 0.0],
            [0.0, 1.0, 1.0, 1.0, 0.0],
        ],
        [
            [0.0, 1.0, 1.0, 1.0, 0.0],
            [1.0, 0.0, 0.0, 0.0, 1.0],
            [0.0, 0.0, 0.0, 0.0, 1.0],
            [0.0, 0.0, 0.0, 1.0, 0.0],
            [0.0, 0.0, 1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0, 0.0, 0.0],
            [1.0, 1.0, 1.0, 1.0, 1.0],
        ],
        [
            [0.0, 1.0, 1.0, 1.0, 0.0],
            [1.0, 0.0, 0.0, 0.0, 1.0],
            [0.0, 0.0, 0.0, 0.0, 1.0],
            [0.0, 0.0, 1.0, 1.0, 0.0],
            [0.0, 0.0, 0.0, 0.0, 1.0],
            [1.0, 0.0, 0.0, 0.0, 1.0],
            [0.0, 1.0, 1.0, 1.0, 0.0],
        ],
        [
            [0.0, 0.0, 0.0, 1.0, 0.0],
            [0.0, 0.0, 1.0, 1.0, 0.0],
            [0.0, 1.0, 0.0, 1.0, 0.0],
            [1.0, 0.0, 0.0, 1.0, 0.0],
            [1.0, 1.0, 1.0, 1.0, 1.0],
            [0.0, 0.0, 0.0, 1.0, 0.0],
            [0.0, 0.0, 0.0, 1.0, 0.0],
        ],
        [
            [1.0, 1.0, 1.0, 1.0, 1.0],
            [1.0, 0.0, 0.0, 0.0, 0.0],
            [1.0, 1.0, 1.0, 1.0, 0.0],
            [0.0, 0.0, 0.0, 0.0, 1.0],
            [0.0, 0.0, 0.0, 0.0, 1.0],
            [1.0, 0.0, 0.0, 0.0, 1.0],
            [0.0, 1.0, 1.0, 1.0, 0.0],
        ],
        [
            [0.0, 0.0, 1.0, 1.0, 0.0],
            [0.0, 1.0, 0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0, 0.0, 0.0],
            [1.0, 1.0, 1.0, 1.0, 0.0],
            [1.0, 0.0, 0.0, 0.0, 1.0],
            [1.0, 0.0, 0.0, 0.0, 1.0],
            [0.0, 1.0, 1.0, 1.0, 0.0],
        ],
        [
            [1.0, 1.0, 1.0, 1.0, 1.0],
            [0.0, 0.0, 0.0, 0.0, 1.0],
            [0.0, 0.0, 0.0, 1.0, 0.0],
            [0.0, 0.0, 1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0, 0.0, 0.0],
            [0.0, 1.0, 0.0, 0.0, 0.0],
            [0.0, 1.0, 0.0, 0.0, 0.0],
        ],
        [
            [0.0, 1.0, 1.0, 1.0, 0.0],
            [1.0, 0.0, 0.0, 0.0, 1.0],
            [1.0, 0.0, 0.0, 0.0, 1.0],
            [0.0, 1.0, 1.0, 1.0, 0.0],
            [1.0, 0.0, 0.0, 0.0, 1.0],
            [1.0, 0.0, 0.0, 0.0, 1.0],
            [0.0, 1.0, 1.0, 1.0, 0.0],
        ],
        [
            [0.0, 1.0, 1.0, 1.0, 0.0],
            [1.0, 0.0, 0.0, 0.0, 1.0],
            [1.0, 0.0, 0.0, 0.0, 1.0],
            [0.0, 1.0, 1.0, 1.0, 1.0],
            [0.0, 0.0, 0.0, 0.0, 1.0],
            [0.0, 0.0, 0.0, 1.0, 0.0],
            [0.0, 1.0, 1.0, 0.0, 0.0],
        ],
    ]

    labels = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
    # One-hot encoding
    labels = [[1 if i == label else 0 for i in range(10)] for label in labels]

    training_set = np.array(digits)
    training_set = training_set.reshape(training_set.shape[0], -1)
    training_labels = np.array(labels)

    test_set = training_set.copy()
    test_labels = training_labels.copy()

    # Add noise to test set
    for i, sample in enumerate(test_set):
        noise = np.random.normal(0, 0.1, sample.shape)
        test_set[i] = np.clip(sample + noise, 0, 1)

    return training_set, training_labels, test_set, test_labels


# --------------------- Unsupervised ---------------------


def load_font_data():
    font = [
        [0x04, 0x04, 0x02, 0x00, 0x00, 0x00, 0x00],  # 0x60, `
        [0x00, 0x0E, 0x01, 0x0D, 0x13, 0x13, 0x0D],  # 0x61, a
        [0x10, 0x10, 0x10, 0x1C, 0x12, 0x12, 0x1C],  # 0x62, b
        [0x00, 0x00, 0x00, 0x0E, 0x10, 0x10, 0x0E],  # 0x63, c
        [0x01, 0x01, 0x01, 0x07, 0x09, 0x09, 0x07],  # 0x64, d
        [0x00, 0x00, 0x0E, 0x11, 0x1F, 0x10, 0x0F],  # 0x65, e
        [0x06, 0x09, 0x08, 0x1C, 0x08, 0x08, 0x08],  # 0x66, f
        [0x0E, 0x11, 0x13, 0x0D, 0x01, 0x01, 0x0E],  # 0x67, g
        [0x10, 0x10, 0x10, 0x16, 0x19, 0x11, 0x11],  # 0x68, h
        [0x00, 0x04, 0x00, 0x0C, 0x04, 0x04, 0x0E],  # 0x69, i
        [0x02, 0x00, 0x06, 0x02, 0x02, 0x12, 0x0C],  # 0x6a, j
        [0x10, 0x10, 0x12, 0x14, 0x18, 0x14, 0x12],  # 0x6b, k
        [0x0C, 0x04, 0x04, 0x04, 0x04, 0x04, 0x04],  # 0x6c, l
        [0x00, 0x00, 0x0A, 0x15, 0x15, 0x11, 0x11],  # 0x6d, m
        [0x00, 0x00, 0x16, 0x19, 0x11, 0x11, 0x11],  # 0x6e, n
        [0x00, 0x00, 0x0E, 0x11, 0x11, 0x11, 0x0E],  # 0x6f, o
        [0x00, 0x1C, 0x12, 0x12, 0x1C, 0x10, 0x10],  # 0x70, p
        [0x00, 0x07, 0x09, 0x09, 0x07, 0x01, 0x01],  # 0x71, q
        [0x00, 0x00, 0x16, 0x19, 0x10, 0x10, 0x10],  # 0x72, r
        [0x00, 0x00, 0x0F, 0x10, 0x0E, 0x01, 0x1E],  # 0x73, s
        [0x08, 0x08, 0x1C, 0x08, 0x08, 0x09, 0x06],  # 0x74, t
        [0x00, 0x00, 0x11, 0x11, 0x11, 0x13, 0x0D],  # 0x75, u
        [0x00, 0x00, 0x11, 0x11, 0x11, 0x0A, 0x04],  # 0x76, v
        [0x00, 0x00, 0x11, 0x11, 0x15, 0x15, 0x0A],  # 0x77, w
        [0x00, 0x00, 0x11, 0x0A, 0x04, 0x0A, 0x11],  # 0x78, x
        [0x00, 0x11, 0x11, 0x0F, 0x01, 0x11, 0x0E],  # 0x79, y
        [0x00, 0x00, 0x1F, 0x02, 0x04, 0x08, 0x1F],  # 0x7a, z
        [0x06, 0x08, 0x08, 0x10, 0x08, 0x08, 0x06],  # 0x7b, {
        [0x04, 0x04, 0x04, 0x00, 0x04, 0x04, 0x04],  # 0x7c, |
        [0x0C, 0x02, 0x02, 0x01, 0x02, 0x02, 0x0C],  # 0x7d, }
        [0x08, 0x15, 0x02, 0x00, 0x00, 0x00, 0x00],  # 0x7e, ~
        [0x1F, 0x1F, 0x1F, 0x1F, 0x1F, 0x1F, 0x1F],  # 0x7
    ]
    labels = [
        "`",
        "a",
        "b",
        "c",
        "d",
        "e",
        "f",
        "g",
        "h",
        "i",
        "j",
        "k",
        "l",
        "m",
        "n",
        "o",
        "p",
        "q",
        "r",
        "s",
        "t",
        "u",
        "v",
        "w",
        "x",
        "y",
        "z",
        "{",
        "|",
        "}",
        "~",
        "del",
    ]

    font_matrix = np.zeros((len(font), 7, 5), dtype=np.int8)
    for i, char in enumerate(font):
        for j, row in enumerate(char):
            for k in range(5):
                font_matrix[i, j, k] = (row >> (4 - k)) & 0x01

    return font_matrix, labels


def load_emoji_data():
    emojis = [
        [0x3C, 0x42, 0xA5, 0x81, 0xBD, 0x99, 0x42, 0x3C],  # :D
        [0x3C, 0x42, 0xA5, 0x81, 0x99, 0xA5, 0x42, 0x3C],  # :(
        [0x3C, 0x42, 0xA5, 0x81, 0x81, 0x81, 0x42, 0x3C],  # :x
        [0x3C, 0x42, 0xA5, 0x81, 0xA5, 0x99, 0x42, 0x3C],  # :)
        [0x3C, 0x42, 0xA5, 0xC3, 0x81, 0x99, 0x42, 0x3C],  # ;|
        [0x3C, 0x42, 0xA5, 0x81, 0x99, 0x99, 0x42, 0x3C],  # :O
        [0x3C, 0x42, 0xA5, 0x81, 0xBD, 0x89, 0x42, 0x3C],  # :P
        [0x3C, 0x42, 0xA5, 0xA5, 0x81, 0x99, 0x42, 0x3C],  # x|
        [0x3C, 0x42, 0xA5, 0x81, 0xBD, 0x81, 0x42, 0x3C],  # :|
        [0x3C, 0x42, 0x81, 0x93, 0x81, 0x8D, 0x42, 0x3C],  # ^|
        [0x3C, 0x42, 0x81, 0xC9, 0x81, 0xB1, 0x42, 0x3C],  # V|
        [0x3C, 0x42, 0xA1, 0xA5, 0x81, 0x9D, 0x42, 0x3C],  # ?|
    ]

    labels = [
        ":D",
        ":(",
        ":x",
        ":)",
        ";|",
        ":O",
        ":P",
        "X|",
        ":|",
        "^|",
        "V|",
        "?|"
    ]

    emoji_matrix = np.zeros((len(emojis), 8, 8), dtype=np.int8)
    for i, char in enumerate(emojis):
        for j, row in enumerate(char):
            for k in range(8):
                emoji_matrix[i, j, k] = (row >> (7 - k)) & 0x01

    return emoji_matrix, labels
