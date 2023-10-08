import tensorflow as tf
from tensorflow.keras.regularizers import l2
from tensorflow.keras.activations import sigmoid
from src.layers import MultiHeadSelfAttention

tfk = tf.keras
tfkl = tf.keras.layers
tfkc = tf.keras.callbacks

class WeakRM(tf.keras.Model):

    def __init__(self, training=True):
        super(WeakRM, self).__init__()

        self.conv1 = tfkl.Conv1D(50, 15, padding='same', activation='relu')

        self.conv3 = tfkl.Conv1D(32, 15, padding='same', activation='relu')
        self.conv2 = tfkl.Conv1D(16, 5, padding='same', activation='relu',
                                kernel_regularizer=l2(0.005))
        self.dropout1 = tfkl.Dropout(0.25)
        self.dropout2 = tfkl.Dropout(0.25)
        self.dropout3 = tfkl.Dropout(0.25)

        self.pool1 = tfkl.MaxPool1D(pool_size=2)

        self.att_v = tfkl.Dense(128, activation='tanh')
        self.att_u = tfkl.Dense(128, activation='sigmoid')

        self.attention_weights = tfkl.Dense(1)

        #self.classifier = tfkl.Dense(1, activation='sigmoid')

    def call(self, inputs, training=True, mask=None):
        input_bag = tf.squeeze(inputs, axis=0)

        inst_conv1 = self.conv1(input_bag)
        inst_pool1 = self.pool1(inst_conv1)
        inst_pool1 = self.dropout1(inst_pool1, training=False)

        inst_conv3 = self.conv3(inst_pool1)
        inst_conv3 = self.dropout3(inst_conv3, training=False)

        inst_conv2 = self.conv2(inst_conv3)
        inst_conv2 = self.dropout2(inst_conv2, training=training)

        inst_features = tfkl.Flatten()(inst_conv2)

        attention_vmatrix = self.att_v(inst_features)
        attention_umatrix = self.att_u(inst_features)

        gated_attention = self.attention_weights(attention_vmatrix * attention_umatrix)

        gated_attention = tf.transpose(gated_attention, perm=[1, 0])
        gated_attention = tfkl.Softmax()(gated_attention)

        bag_features = tf.matmul(gated_attention, inst_features)

        #bag_probability = self.classifier(bag_features)

        return bag_features, gated_attention

class WeakRMFine(tf.keras.Model):

    def __init__(self, training=True):
        super(WeakRMFine, self).__init__()
        self.conv2 = tfkl.Conv1D(16, 5, padding='same', activation='relu',
                                kernel_regularizer=l2(0.005))
        self.dropout2 = tfkl.Dropout(0.25)


        self.pool1 = tfkl.MaxPool1D(pool_size=2)

        self.att_v = tfkl.Dense(128, activation='tanh')
        self.att_u = tfkl.Dense(128, activation='sigmoid')

        self.attention_weights = tfkl.Dense(1)

        #self.classifier = tfkl.Dense(1, activation='sigmoid')

    def call(self, inputs, training=True, mask=None):
        #input_bag = tf.squeeze(inputs, axis=0)
        inst_conv2 = self.conv2(inputs)
        inst_conv2 = self.dropout2(inst_conv2, training=training)

        inst_features = tfkl.Flatten()(inst_conv2)

        attention_vmatrix = self.att_v(inst_features)
        attention_umatrix = self.att_u(inst_features)

        gated_attention = self.attention_weights(attention_vmatrix * attention_umatrix)

        gated_attention = tf.transpose(gated_attention, perm=[1, 0])
        gated_attention = tfkl.Softmax()(gated_attention)

        bag_features = tf.matmul(gated_attention, inst_features)

        #bag_probability = self.classifier(bag_features)

        return bag_features, gated_attention

class WeakMASS(tf.keras.Model):
    def __init__(self, training=True):
        super(WeakMASS, self).__init__()

        self.conv1 = tfkl.Conv1D(32, 15, dilation_rate=1, padding='same', activation='relu')
        self.pool1 = tfkl.MaxPool1D(pool_size=2)

        self.conv2 = tfkl.Conv1D(32, 15, dilation_rate=2, padding='same', activation='relu')
        self.pool2 = tfkl.MaxPool1D(pool_size=2)

        self.conv3 = tfkl.Conv1D(32, 15, dilation_rate=4, padding='same', activation='relu')
        self.pool3 = tfkl.MaxPool1D(pool_size=2)

        self.conv4 = tfkl.Conv1D(16, 5, dilation_rate=1, padding='same', activation=None)
        self.batch4 = tfkl.BatchNormalization()
        self.prelu4 = tfkl.PReLU()
        self.dropout4 = tfkl.Dropout(0.25)

        self.conv6 = tfkl.Conv1D(16, 5, dilation_rate=1, padding='same', activation=None)
        self.batch6 = tfkl.BatchNormalization()
        self.prelu6 = tfkl.PReLU()
        self.dropout6 = tfkl.Dropout(0.25)

        self.conv7 = tfkl.Conv1D(16, 5, dilation_rate=1, padding='same', activation=None)
        self.batch7 = tfkl.BatchNormalization()
        self.prelu7 = tfkl.PReLU()
        self.dropout7 = tfkl.Dropout(0.25)

        self.dropout8 = tfkl.Dropout(0.25)

        self.birnn = tfkl.Bidirectional(tf.keras.layers.RNN(tf.keras.layers.LSTMCell(16),return_sequences=True))

        self.att_v = tfkl.Dense(128, activation='tanh')
        self.att_u = tfkl.Dense(128, activation='sigmoid')

        self.attention_weights = tfkl.Dense(1)

    def call(self, inputs, training=True, mask=None):
        input_bag = tf.squeeze(inputs, axis=0)

        feature1 = self.conv1(input_bag)
        feature1 = self.pool1(feature1)

        feature2 = self.conv2(input_bag)
        feature2 = self.pool2(feature2)

        feature3 = self.conv3(input_bag)
        feature3 = self.pool3(feature3)

        features = tfkl.Concatenate()([feature1, feature2, feature3])


        features = self.conv4(features)
        con_features = self.prelu4(features)
        features = self.dropout4(features, training=training)

        features = self.conv6(features)
        features = self.prelu6(features)
        features = self.dropout6(features, training=training)

        features = tfkl.Add()([features, con_features])

        features = self.conv7(features)
        features = self.prelu7(features)
        features = self.dropout7(features, training=training)

        features = self.birnn(features)

        features = tfkl.GlobalMaxPool1D()(features)

        attention_vmatrix = self.att_v(features)
        attention_umatrix = self.att_u(features)

        gated_attention = self.attention_weights(attention_vmatrix * attention_umatrix)

        gated_attention = tf.transpose(gated_attention, perm=[1, 0])
        gated_attention = tfkl.Softmax()(gated_attention)

        bag_features = tf.matmul(gated_attention, features)


        return bag_features, gated_attention

class DenseClassifier(tf.keras.Model):

    def __init__(self, training=True):
        super(DenseClassifier, self).__init__()
        self.dropout1 = tfkl.Dropout(0.25)
        self.dropout2 = tfkl.Dropout(0.25)
        self.pool1 = tfkl.MaxPool1D(pool_size=2)

        self.dense1 = tfkl.Dense(64)
        self.dense2 = tfkl.Dense(32)
        self.classifier = tfkl.Dense(1, activation='sigmoid')

    def call(self,  bag_features, training=True):

        #bag_features = self.pool1(bag_features)

        bag_features = self.dense1(bag_features)
        bag_features = self.dropout1(bag_features, training=training)

        bag_features = self.dense2(bag_features)
        bag_features = self.dropout2(bag_features, training=training)

        bag_probability = self.classifier(bag_features)

        return bag_probability

class WeakRMOld(tf.keras.Model):

    def __init__(self, training=True):
        super(WeakRMOld, self).__init__()

        self.conv1 = tfkl.Conv1D(32, 15, padding='same', activation='relu')
        self.conv2 = tfkl.Conv1D(16, 5, padding='same', activation='relu',
                                kernel_regularizer=l2(0.005))
        self.dropout1 = tfkl.Dropout(0.25)
        self.dropout2 = tfkl.Dropout(0.25)
        self.pool1 = tfkl.MaxPool1D(pool_size=2)

        self.att_v = tfkl.Dense(128, activation='tanh')
        self.att_u = tfkl.Dense(128, activation='sigmoid')

        self.attention_weights = tfkl.Dense(1)

    def call(self, inputs, training=True, mask=None):
        input_bag = tf.squeeze(inputs, axis=0)

        inst_conv1 = self.conv1(input_bag)
        inst_pool1 = self.pool1(inst_conv1)
        inst_pool1 = self.dropout1(inst_pool1, training=training)

        inst_conv2 = self.conv2(inst_pool1)
        inst_conv2 = self.dropout2(inst_conv2, training=training)

        inst_features = tfkl.Flatten()(inst_conv2)

        attention_vmatrix = self.att_v(inst_features)
        attention_umatrix = self.att_u(inst_features)

        gated_attention = self.attention_weights(attention_vmatrix * attention_umatrix)

        gated_attention = tf.transpose(gated_attention, perm=[1, 0])
        gated_attention = tfkl.Softmax()(gated_attention)

        bag_features = tf.matmul(gated_attention, inst_features)

        return bag_features, gated_attention

class WeakRMOldClassifier(tf.keras.Model):

    def __init__(self, training=True):
        super(WeakRMOldClassifier, self).__init__()

        self.classifier = tfkl.Dense(1, activation='sigmoid')

    def call(self,  bag_features, training=True):

        #bag_features = self.pool1(bag_features)

        bag_probability = self.classifier(bag_features)

        return bag_probability

class WeakRMTestNets(tf.keras.Model):

    def __init__(self, training=True):
        super(WeakRMTestNets, self).__init__()

        self.conv1 = tfkl.Conv1D(32, 15, padding='same', activation='relu')
        self.conv2 = tfkl.Conv1D(16, 5, padding='same', activation='relu',
                                kernel_regularizer=l2(0.005))
        self.dropout1 = tfkl.Dropout(0.25)
        self.dropout2 = tfkl.Dropout(0.25)
        self.pool1 = tfkl.MaxPool1D(pool_size=2)

        self.att_v = tfkl.Dense(128, activation='tanh')
        self.att_u = tfkl.Dense(128, activation='sigmoid')

        self.attention_weights = tfkl.Dense(1)

        self.classifier = tfkl.Dense(1, activation='sigmoid')

    def call(self, inputs, training=True, mask=None):
        input_bag = tf.squeeze(inputs, axis=0)

        inst_conv1 = self.conv1(input_bag)
        inst_pool1 = self.pool1(inst_conv1)
        inst_pool1 = self.dropout1(inst_pool1, training=training)

        inst_conv2 = self.conv2(inst_pool1)
        inst_conv2 = self.dropout2(inst_conv2, training=training)

        inst_features = tfkl.Flatten()(inst_conv2)

        attention_vmatrix = self.att_v(inst_features)
        attention_umatrix = self.att_u(inst_features)

        gated_attention = self.attention_weights(attention_vmatrix * attention_umatrix)

        gated_attention = tf.transpose(gated_attention, perm=[1, 0])
        gated_attention = tfkl.Softmax()(gated_attention)

        bag_features = tf.matmul(gated_attention, inst_features)

        bag_probability = self.classifier(bag_features)

        return bag_probability, bag_features