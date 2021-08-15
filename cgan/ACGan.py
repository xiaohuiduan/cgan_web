from keras.layers import BatchNormalization, Activation, Embedding
from keras.layers import Concatenate
from keras.layers import Conv2D, Conv2DTranspose
from keras.layers import Input, Reshape, Flatten
from keras.layers.advanced_activations import LeakyReLU
from keras.models import Sequential, Model
import numpy as np


class ACGan:
    def __init__(self, model_file):
        self.hair = ['green hair', 'yellow hair', 'red hair', 'black hair', 'blue hair', 'purple hair']
        self.eye = ['purple eyes', 'green eyes', 'brown eyes', 'red eyes', 'blue eyes']
        self.latent_size = 100
        self.G = self.build_generator()
        self.G.load_weights(model_file)



    def build_generator(self):
        num_class_hairs = 6
        num_class_eyes = 5
        kernel_init = 'glorot_uniform'
        latent_size = 100
        model = Sequential()
        model.add(Reshape((1, 1, -1), input_shape=(latent_size + 16,)))
        model.add(
            Conv2DTranspose(filters=512, kernel_size=(4, 4), strides=(1, 1), padding="valid",
                            data_format="channels_last",
                            kernel_initializer=kernel_init, ))
        model.add(BatchNormalization(momentum=0.5))
        model.add(LeakyReLU(0.2))
        model.add(
            Conv2DTranspose(filters=256, kernel_size=(4, 4), strides=(2, 2), padding="same",
                            data_format="channels_last",
                            kernel_initializer=kernel_init))
        model.add(BatchNormalization(momentum=0.5))
        model.add(LeakyReLU(0.2))
        model.add(
            Conv2DTranspose(filters=128, kernel_size=(4, 4), strides=(2, 2), padding="same",
                            data_format="channels_last",
                            kernel_initializer=kernel_init))
        model.add(BatchNormalization(momentum=0.5))
        model.add(LeakyReLU(0.2))
        model.add(
            Conv2DTranspose(filters=64, kernel_size=(4, 4), strides=(2, 2), padding="same", data_format="channels_last",
                            kernel_initializer=kernel_init))
        model.add(BatchNormalization(momentum=0.5))
        model.add(LeakyReLU(0.2))
        model.add(Conv2D(filters=64, kernel_size=(3, 3), strides=(1, 1), padding="same", data_format="channels_last",
                         kernel_initializer=kernel_init))
        model.add(BatchNormalization(momentum=0.5))
        model.add(LeakyReLU(0.2))
        model.add(
            Conv2DTranspose(filters=3, kernel_size=(4, 4), strides=(2, 2), padding="same", data_format="channels_last",
                            kernel_initializer=kernel_init))
        model.add(Activation('tanh'))
        # 3 inputs
        latent = Input(shape=(latent_size,))
        eyes_class = Input(shape=(1,), dtype='int32')
        hairs_class = Input(shape=(1,), dtype='int32')
        # embedding
        eyes = Flatten()(Embedding(num_class_eyes, 8, init='glorot_normal')(eyes_class))
        hairs = Flatten()(Embedding(num_class_hairs, 8, init='glorot_normal')(hairs_class))
        # h = merge(, mode='concat')
        h = Concatenate()([latent, hairs, eyes])
        fake_image = model(h)
        m = Model(input=[latent, hairs_class, eyes_class], output=fake_image)
        return m

    def generate_images(self, batch_size, hair_color, eyes_color):
        """
        产生图片
        :param batch_size: 图片的数量
        :param hair_color: hair序号
        :param eyes_color: eys序号
        :return:
        """

        def gen_noise(images_num, latent_size):
            """
            产生nosise
            :param images_num: 想要图片的数量
            :param latent_size:
            :return:
            """
            return np.random.normal(0, 1, size=(images_num, latent_size))

        noise = gen_noise(batch_size, self.latent_size)

        return self.G.predict([noise, hair_color, eyes_color])

    def createRandom(self, batch_size=1):
        hair_color = np.random.randint(0, len(self.hair),batch_size)
        eye_color = np.random.randint(0, len(self.eye),batch_size)

        hair_color = np.array(hair_color).reshape(-1, 1)
        eye_color = np.array(eye_color).reshape(-1, 1)
        return self.generate_images(batch_size, hair_color, eye_color)

    def create_special(self, hair_color, eye_color, batch_size=1):
        hair_color = np.array([hair_color]*batch_size).reshape(-1, 1)
        eye_color = np.array([eye_color]*batch_size).reshape(-1, 1)
        return self.generate_images(batch_size, hair_color, eye_color)