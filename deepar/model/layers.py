import tensorflow as tf
from tensorflow.keras.layers import LSTM, Layer
from tensorflow.keras.activations import softplus

# class LSTMResetStateful(LSTM):

#     def __init__(self, *args, **kwargs):
#         super(LSTMResetStateful, self).__init__(*args, **kwargs)

#     def build(self, *args, **kwargs):
#         super(LSTMResetStateful, self).build(*args, **kwargs)

#     def call(self, inputs, mask = None, training = None):
        
#         if training:
#             self.reset_states()
        
#         return super(LSTMResetStateful, self).call(inputs, mask = mask, training = training)

class GaussianLayer(Layer):
    """
        from https://github.com/arrigonialberto86/deepar, learns mu and scale outputs in single layer
    """

    def __init__(self, output_dim = 1, count_data = False, **kwargs):
        self.output_dim = output_dim
        self.count_data = count_data
        self.kernel_1, self.kernel_2, self.bias_1, self.bias_2 = [], [], [], []
        super(GaussianLayer, self).__init__(**kwargs)

    def build(self, input_shape):
        features = input_shape[2]
        self.kernel_1 = self.add_weight(name='kernel_1',
                                        shape=(features, self.output_dim),
                                        initializer='glorot_normal',
                                        trainable=True)
        self.kernel_2 = self.add_weight(name='kernel_2',
                                        shape=(features, self.output_dim),
                                        initializer='glorot_normal',
                                        trainable=True)
        self.bias_1 = self.add_weight(name='bias_1',
                                      shape=(self.output_dim,),
                                      initializer='glorot_normal',
                                      trainable=True)
        self.bias_2 = self.add_weight(name='bias_2',
                                      shape=(self.output_dim,),
                                      initializer='glorot_normal',
                                      trainable=True)
        super(GaussianLayer, self).build(input_shape)

    def call(self, x):
        output_mu = tf.matmul(x, self.kernel_1) + self.bias_1
        output_sig = tf.matmul(x, self.kernel_2) + self.bias_2
        output_sig = softplus(output_sig)
        if self.count_data:
            output_mu = softplus(output_mu)
        return [output_mu, output_sig]

    def compute_output_shape(self, input_shape):
        return [(input_shape[0], self.output_dim), (input_shape[0], self.output_dim)]