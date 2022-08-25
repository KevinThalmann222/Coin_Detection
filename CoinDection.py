import math
import random
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras import models
from CoinDetectionModelTrain import TrainModel


class Dection:
    def __init__(self) -> None:
        self.train_model = TrainModel()
        self.label_list = self.train_model.label_list
        self.model = models.load_model(self.train_model.model_name)
        self.images = []
        self.real_label = []
        self.predicted_label = []

    def predict_random_image(self, num_of_prediction=1):
        for _ in range(num_of_prediction):
            random_coin_folder = self.train_model.test_path / random.choice(self.label_list)
            coin_image_list = list(random_coin_folder.iterdir())
            random_image_path = str(coin_image_list[random.randint(0, len(coin_image_list) - 1)])
            image = self.train_model.process_image(random_image_path, self.train_model.image_size)
            prediction = self.model.predict(np.array([image]))
            predicted_coin = self.label_list[np.argmax(prediction)]
            print("coin image     : ", random_coin_folder.stem)
            print("predicted coin : ", predicted_coin)

            self.images.append(image)
            self.predicted_label.append(predicted_coin)
            self.real_label.append(random_coin_folder.stem)

    def show_predict_image(self):
        if not self.images:
            self.predict_random_image()
        for i in range(length := len(self.images)):
            plt.subplot(math.ceil(x := math.sqrt(length)), int(round(x, 0)), i + 1)
            plt.xticks([])
            plt.yticks([])
            plt.ylabel(f"Label: {self.real_label[i]}")
            plt.xlabel(f"Predict: {self.predicted_label[i]}")
            plt.imshow(self.images[i])
        plt.show()


if __name__ == "__main__":
    dection = Dection()
    dection.predict_random_image(num_of_prediction=11)
    dection.show_predict_image()
