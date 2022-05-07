'''
Group members:
n9854258 Tan Chieu Duong Nguyen
n10362380 Hsiang-Ling Fan
n10178414 Pei-Fang Shen
'''

import random
import tensorflow_datasets as tfds
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

from tensorflow import keras
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Lambda
from tensorflow.keras.optimizers import RMSprop

class Dataset:
    '''
    This class will facilitate the creation of a few-shot dataset
    from the Omniglot dataset that can be sampled from quickly while also
    allowing to create new labels at the same time.
    '''

    def __init__(self, training):
        # Download the tfrecord files containing the omniglot data and convert to a dataset
        split = "train" if training else "test"
        ds = tfds.load('omniglot', split=split)
        # Iterate over the dataset to get each individual image and its class,
        # and put that data into a dictionary.
        self.data = {}

        def transform(images):
            # This function will shrink the Omniglot images to the desired size,
            # scale pixel values and convert the RGB image to grayscale
            images = tf.image.convert_image_dtype(images, tf.float32)
            images = tf.image.rgb_to_grayscale(images)
            images = tf.image.resize(images, [28, 28])
            return images

        # set alphebet as class
        for value in ds:
            image = transform(value["image"]).numpy()
            label = str(value["alphabet"].numpy())
            if label not in self.data:
                self.data[label] = []
            self.data[label].append(image)
            self.labels = list(self.data.keys())

class BaseNetwork:
    '''
    The initial function will create train pairs and train labels, 
    test pairs and test labels, train_and_test pairs and train_test_pairs.
    Train pairs is derived from training split, test pairs is derived from test split 
    and train_and_test pairs is derived from both splits.
    '''
    def __init__(self, train, test, input_shape, batch_size, epochs):
        self.tr_x, self.tr_y = self.create_pairs(train, train.labels)
        self.te_x, self.te_y = self.create_pairs(test, test.labels)
        self.tr_te_x = np.concatenate((self.tr_x, self.te_x), axis=0)
        self.tr_te_y = np.concatenate((self.tr_y, self.te_y), axis=0)

        self.input_shape = input_shape
        self.num_classes = len(train.labels) + len(test.labels)
        self.batch_size = batch_size
        self.epochs = epochs

    def cnn_base_network(self, input_shape, num_classes):
        '''
        Base network to be shared (eq. to feature extraction).
        '''
        cnn_model = keras.models.Sequential()

        # Adds layers to the sequential model
        cnn_model.add(keras.layers.Conv2D(64, kernel_size=(3, 3),
                                        activation='relu',
                                        input_shape=input_shape))
        cnn_model.add(keras.layers.MaxPooling2D(pool_size=(2, 2)))
        cnn_model.add(keras.layers.Flatten())
        cnn_model.add(keras.layers.Dense(128, activation='relu'))
        cnn_model.add(keras.layers.Dropout(0.1))
        cnn_model.add(keras.layers.Dense(num_classes, activation='softmax'))

        return cnn_model

def euclidean_distance(vects):
    '''
    Calculate the euclidean distance.
    '''
    x, y = vects
    sum_square = tf.reduce_sum(tf.square(x - y), axis=1, keepdims=True)
    return tf.sqrt(sum_square)

def eucl_dist_output_shape(shapes):
    '''
    Configure the output of Lambda.
    '''
    shape1, shape2 = shapes
    return (shape1[0], 1)

@tf.function
def contrastive_loss(y_true, y_predict):
    '''
    Calculate contrastive loss.
    '''
    margin = 1
    square_predict = tf.square(y_predict)*0.5
    margin_square = tf.square(tf.maximum(margin - y_predict, 0))*0.5
    return tf.reduce_mean(y_true * square_predict + (1 - y_true) * margin_square)

def accuracy(y_true, y_predict):
    '''
    Compute classification accuracy with a fixed threshold on distances.
    '''
    return tf.keras.backend.mean(tf.keras.backend.equal(y_true, tf.keras.backend.cast(y_predict < 0.5, y_true.dtype)))

class SiameseNetwork(BaseNetwork):
    '''
    This class is used to create the Siamese network and train it.
    '''
    def create_pairs(self, x, num_classes):
        '''
        This function will create positive pairs and negative pairs.
        '''
        pairs = []
        labels = []
        for index, value in enumerate(num_classes):
            n = len(x.data[value])-1
            for i in range(n):
                z1, z2 = x.data[value][i], x.data[value][i+1]
                pairs += [[z1, z2]]
                inc = random.randrange(1, len(num_classes))
                index_n = (index + inc) % len(num_classes)
                index_v = random.randint(0, len(x.data[num_classes[index_n]])-1)
                z3 = x.data[num_classes[index_n]][index_v]
                pairs += [[z1, z3]]
                labels += [1., 0.]
        return np.array(pairs), np.array(labels)

    def create_model(self):
        '''
        Create Siamese Network model.
        Input shape is the size of the picture.
        Number of classes is the number of alphabet in omniglot.
        '''
        tf.keras.backend.clear_session()
        # Network definition
        base_network = self.cnn_base_network(self.input_shape, self.num_classes)
        base_network.summary()

        input_a = Input(shape=self.input_shape)
        input_b = Input(shape=self.input_shape)

        # Siamese network share equal weights,
        # and we will use same embedding CNN BaseNetwork for the two towers.
        tower_a = base_network(input_a)
        tower_b = base_network(input_b)

        distance = Lambda(euclidean_distance, output_shape=eucl_dist_output_shape)([tower_a, tower_b])

        model = Model([input_a, input_b], distance)

        rms = RMSprop()

        # Compile the model with the contrastive loss
        model.compile(loss=contrastive_loss, optimizer=rms, metrics=[accuracy])
        model.summary()
        return model

    def test_loss(self):
        '''
        This function is used to test the implement of contrastive loss with numpy and tensorflow.
        Once the results of numpy and tensorflow are different, return error.
        '''
        num_data = 10
        feat_dim = 6

        embeddings = [np.random.rand(num_data, feat_dim).astype(np.float32),
                    np.random.rand(num_data, feat_dim).astype(np.float32)]
        labels = np.random.randint(0, 1, size=(num_data)).astype(np.float32)

        # Compute loss with Numpy
        loss_np = 0.
        e1 = embeddings[0]
        e2 = embeddings[1]
        for i in range(num_data):
            dist = np.sum(np.square(e1[i] - e2[i]))
            dist2 = np.sqrt(dist)
            s_pred = np.square(dist2)*0.5
            m_square = np.square(max(1-dist2, 0))*0.5
            loss_np += (labels[i]*s_pred + (1-labels[i])*m_square)
        loss_np /= num_data
        print('Contrastive loss computed: ', loss_np)

        # Test implementation of contrastive_loss function 
        distance = euclidean_distance(embeddings)
        loss_tf_val = contrastive_loss(labels, distance)
        print('Contrastive loss computed with Tensorflow: ', loss_tf_val)

        assert np.allclose(loss_tf_val, loss_np)

    def draw_result(self, history):
        '''
        Plot the history result with accuracy and loss.
        '''
        # Plot training & validation accuracy values
        plt.plot(history.history['accuracy'])
        plt.plot(history.history['val_accuracy'])
        plt.title('Model accuracy')
        plt.ylabel('Accuracy')
        plt.xlabel('Epoch')
        plt.legend(['Train', 'Test'], loc='upper left')
        plt.show()

        # Plot training & validation loss values
        plt.plot(history.history['loss'])
        plt.plot(history.history['val_loss'])
        plt.title('Model loss')
        plt.ylabel('Loss')
        plt.xlabel('Epoch')
        plt.legend(['Train', 'Test'], loc='upper left')
        plt.show()

    def evaluation(self, model):
        '''
        Model evaluation.
        '''
        score_tr = model.evaluate([self.tr_x[:, 0], self.tr_x[:, 1]], self.tr_y, verbose=0)
        print('Train loss: %0.2f%%' % (100 * score_tr[0]))
        print('Train accuracy: %0.2f%%' % (100 * score_tr[1]))
        score_tr_te = model.evaluate([self.tr_te_x[:, 0], self.tr_te_x[:, 1]], self.tr_te_y, verbose=0)
        print('Train & Test loss: %0.2f%%' % (100 * score_tr_te[0]))
        print('Train & Test accuracy: %0.2f%%' % (100 * score_tr_te[1]))
        score_te = model.evaluate([self.te_x[:, 0], self.te_x[:, 1]], self.te_y, verbose=0)
        print('Test loss: %0.2f%%' % (100 * score_te[0]))
        print('Test accuracy: %0.2f%%' % (100 * score_te[1]))

    def start_training(self):
        '''
        Start training the Siamese network.

        It will create a new model to validate with different dataset.
        Therefore, three models are created: 
        1- Training dataset, 
        2- Test dataset,
        3- Train & Test dataset.
        '''
        data = [[[self.tr_x, self.tr_y], "training"], 
                [[self.te_x, self.te_y], "test"],
                [[self.tr_te_x, self.tr_te_y], "train_and_test"]]
        self.historys = []

        for value in data:
            print("Start training Siamese with training dataset, and validate with: " + value[1])
            model = self.create_model()
            history = model.fit([self.tr_x[:, 0], self.tr_x[:, 1]], self.tr_y,
                validation_data=([value[0][0][:, 0], value[0][0][:, 1]], value[0][1]),
                batch_size=self.batch_size,
                epochs=self.epochs)
            self.historys.append(history)
            self.draw_result(history)
            self.evaluation(model)
            print()

def triplet_loss(x, alpha=0.2):
    '''
    Triple loss function.
    '''
    # Triplet Loss function.
    anchor, positive, negative = x
    # distance between the anchor and the positive
    pos_dist = tf.reduce_sum(tf.square(anchor-positive), axis=1)
    # distance between the anchor and the negative
    neg_dist = tf.reduce_sum(tf.square(anchor-negative), axis=1)
    loss = tf.maximum(pos_dist-neg_dist+alpha, 0.)
    return loss

@tf.function
def identity_loss(y_true, y_pred):
    '''
    Triple loss function.
    '''
    return tf.reduce_mean(y_pred)

class TripletNetwork(BaseNetwork):
    '''
    This class is used to create the implement of triplet network and train it.

    The initial function will create train pairs and train labels, 
    test pairs and test labels, train_and_test pairs and train_test_pairs.
    
    Train pairs is derived from training split, test pairs is derived from test split 
    and train_and_test pairs is derived from both splits.
    '''
    def create_pairs(self, x, num_classes):
        '''
        This function will create positive pairs and negative pairs.
        '''
        pairs = []
        labels = []
        for index, value in enumerate(num_classes):
            n = len(x.data[value])-1
            for i in range(n):
                z1, z2 = x.data[value][i], x.data[value][i+1]
                inc = random.randrange(1, len(num_classes))
                index_n = (index + inc) % len(num_classes)
                index_v = random.randint(0, len(x.data[num_classes[index_n]])-1)
                z3 = x.data[num_classes[index_n]][index_v]
                pairs += [[z1, z2, z3]]
                labels += [1.]
        return np.array(pairs), np.array(labels)

    def create_model(self):
        '''
        Create Siamese network:
        Input shape is the size of the picture.
        Number of classes is the number of alphabet in omniglot.
        '''
        tf.keras.backend.clear_session()
        # Definition
        base_network = self.cnn_base_network(self.input_shape, self.num_classes)
        base_network.summary()

        in_anc = tf.keras.layers.Input(shape=self.input_shape)
        in_pos = tf.keras.layers.Input(shape=self.input_shape)
        in_neg = tf.keras.layers.Input(shape=self.input_shape)

        em_anc = base_network(in_anc)
        em_pos = base_network(in_pos)
        em_neg = base_network(in_neg)

        # Loss calculation
        loss = Lambda(triplet_loss)([em_anc, em_pos, em_neg])
        net = tf.keras.models.Model([in_anc, in_pos, in_neg], outputs=loss)

        # Training preparation
        rms = RMSprop()
        net.compile(loss=identity_loss, optimizer=rms)
        net.summary()
        return net

    def test_loss(self):
        '''
        Use to test the triplet loss with numpy and tensorflow.
        When the results of numpy and tensorflow are different, return error.
        '''
        num_data = 10
        feat_dim = 6
        margin = 0.2

        embeddings = [np.random.rand(num_data, feat_dim).astype(np.float32),
                    np.random.rand(num_data, feat_dim).astype(np.float32),
                    np.random.rand(num_data, feat_dim).astype(np.float32)]
        labels = np.random.randint(0, 1, size=(num_data)).astype(np.float32)

        # Loss calculation using Numpy
        loss_np = 0.
        anchor = embeddings[0]
        positive = embeddings[1]
        negative = embeddings[2]
        for i in range(num_data):
            pos_dist = np.sum(np.square(anchor[i] - positive[i]))
            neg_dist = np.sum(np.square(anchor[i] - negative[i]))
            loss_np += max(pos_dist-neg_dist+margin, 0.)
        loss_np /= num_data
        print('Triplet loss computed with Numpy', loss_np)

        # Test implementation for triplet loss function 
        loss_tf_pre = triplet_loss(embeddings, margin)
        loss_tf_val = identity_loss(labels, loss_tf_pre)
        print('Triplet loss computed with tensorflow', loss_tf_val)

        assert np.allclose(loss_np, loss_tf_val)

    def draw_result(self, history):
        '''
        Plot the history result with accuracy and loss.
        '''
        plt.plot(history.history['loss'])
        plt.plot(history.history['val_loss'])
        plt.title('Model loss')
        plt.ylabel('Loss')
        plt.xlabel('Epoch')
        plt.legend(['Train', 'Test'], loc='upper left')
        plt.show()

    def start_training(self):
        '''
        Start training triplet network.

        Create new model to validate with different dataset.
        Therefore, three models are created: 
        1- Training dataset, 
        2- Test dataset,
        3- Train & Test dataset.
        '''
        data = [[[self.tr_x, self.tr_y], "training"], 
                [[self.te_x, self.te_y], "test"],
                [[self.tr_te_x, self.tr_te_y], "train_and_test"]]
        self.historys = []

        for value in data:
            print("Start training Triplet with training dataset, and validate with: " + value[1])
            model = self.create_model()
            history = model.fit([self.tr_x[:, 0], self.tr_x[:, 1], self.tr_x[:, 2]], self.tr_y,
                validation_data=([value[0][0][:, 0], value[0][0][:, 1], value[0][0][:, 2]], value[0][1]),
                batch_size=self.batch_size,
                epochs=self.epochs)
            self.historys.append(history)
            self.draw_result(history)
            print()

if __name__ == "__main__":
    training = Dataset(training=True)
    testing = Dataset(training=False)

    inputShape = (28, 28, 1)
    epochs = 8
    batchSize = 256

    # # Create Siamese, then do Loss function testing, then Start training
    Siamese = SiameseNetwork(training, testing, inputShape, batchSize, epochs)
    Siamese.test_loss()
    Siamese.start_training()

    # Create Triplet, then do Loss function testing, then Start training
    Triplet = TripletNetwork(training, testing, inputShape, batchSize, epochs)
    Triplet.test_loss()
    Triplet.start_training()
