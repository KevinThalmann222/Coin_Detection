import cv2
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
from tensorflow.keras import layers, models
from LoadingBar import progress_bar


class TrainModel:
    def __init__(self) -> None:
        self.root_path = Path(__file__).parent
        self.train_path = self.root_path / "train"
        self.test_path = self.root_path / "test"
        self.image_size = (224, 224)
        self.label_list = ["1euro", "2euro", "5cent", "10cent", "20cent", "50cent"]
        self.model_name = "coin_model.model"

    def process_image(self, image, size, show_image=False):
        read_image = cv2.imread(str(image))
        resize_image = cv2.resize(read_image, size)
        # blur_image = cv2.GaussianBlur(resize_image, (5, 5), 1)
        # erosion = cv2.erode(blur_image, np.ones((1, 1)), iterations=1)
        # dilate = cv2.dilate(erosion, np.ones((1, 1)), iterations=8)  # erweitern

        if show_image:
            plt.subplot(1, 2, 1)
            plt.imshow(resize_image)
            plt.show()

        return resize_image / 255

    def get_train_datas(self):
        train_datas = []
        train_label = []

        for path in self.train_path.iterdir():
            for i, img in enumerate(path.iterdir()):
                progress_bar(i, len(list(path.iterdir())), length=30, text=f"importing: {path.stem} train")
                image = self.process_image(image=img, size=self.image_size, show_image=False)
                train_datas.append(image)
                train_label.append([int(self._get_label(path))])

        return np.stack(train_datas), np.array(train_label, dtype="uint8")

    def get_test_datas(self):
        test_datas = []
        test_label = []

        for path in self.test_path.iterdir():
            for i, img in enumerate(path.iterdir()):
                progress_bar(i, len(list(path.iterdir())), length=30, text=f"importing: {path.stem} test")
                pic = cv2.imread(str(img))
                pic = cv2.resize(pic, self.image_size)
                test_datas.append(pic / 255)
                test_label.append([int(self._get_label(path))])

        return np.stack(test_datas), np.array(test_label, dtype="uint8")

    def _get_label(self, path):
        if path.stem.startswith("1"):
            return 1
        elif path.stem.startswith("2"):
            return 2
        elif path.stem.startswith("5"):
            return 3
        elif path.stem.startswith("10"):
            return 4
        elif path.stem.startswith("20"):
            return 5
        elif path.stem.startswith("50"):
            return 6
        else:
            return None

    def create_model(self):
        model = models.Sequential()

        model.add(layers.Conv2D(filters=64, kernel_size=3, padding="same", activation="relu", input_shape=(self.image_size[0], self.image_size[1], 3)))
        model.add(layers.Conv2D(filters=64, kernel_size=3, padding="same", activation="relu"))
        model.add(layers.MaxPool2D(pool_size=2, strides=2, padding="same"))

        model.add(layers.Conv2D(filters=128, kernel_size=3, padding="same", activation="relu"))
        model.add(layers.Conv2D(filters=128, kernel_size=3, padding="same", activation="relu"))
        model.add(layers.MaxPool2D(pool_size=2, strides=2, padding="same"))

        model.add(layers.Conv2D(filters=256, kernel_size=3, padding="same", activation="relu"))
        model.add(layers.Conv2D(filters=256, kernel_size=3, padding="same", activation="relu"))
        model.add(layers.Conv2D(filters=256, kernel_size=3, padding="same", activation="relu"))
        model.add(layers.MaxPool2D(pool_size=2, strides=2, padding="same"))

        model.add(layers.Conv2D(filters=512, kernel_size=3, padding="same", activation="relu"))
        model.add(layers.Conv2D(filters=512, kernel_size=3, padding="same", activation="relu"))
        model.add(layers.Conv2D(filters=512, kernel_size=3, padding="same", activation="relu"))
        model.add(layers.MaxPool2D(pool_size=2, strides=2, padding="same"))

        model.add(layers.Conv2D(filters=512, kernel_size=3, padding="same", activation="relu"))
        model.add(layers.Conv2D(filters=512, kernel_size=3, padding="same", activation="relu"))
        model.add(layers.Conv2D(filters=512, kernel_size=3, padding="same", activation="relu"))
        model.add(layers.MaxPool2D(pool_size=2, strides=2, padding="same"))

        model.add(layers.Flatten())
        model.add(layers.Dense(units=4096, activation="relu"))
        # model.add(layers.Dense(units=1000, activation="relu"))
        # model.add(layers.Dropout(rate=0.2))
        model.add(layers.Dense(units=6, activation="softmax"))

        model.compile(optimizer="adam", loss="sparse_categorical_crossentropy", metrics=["accuracy"])

        train_data, train_label = self.get_train_datas()
        test_data, test_label = self.get_test_datas()

        model.fit(train_data, train_label, epochs=6, validation_data=(test_data, test_label))

        loss, accuracy = model.evaluate(test_data, test_label)
        print("Loss:", loss)
        print("Accuracy", accuracy)

        model.save(self.model_name)


if __name__ == "__main__":
    coin_dec = TrainModel()
    coin_dec.create_model()
