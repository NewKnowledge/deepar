import typing
import math

import tensorflow as tf
from tensorflow.keras.layers import (
    Dense,
    LSTM,
    Embedding,
)
from tensorflow.keras.models import Model
from tensorflow.keras.activations import softplus


# TODO separate training and inference model objects (not stateful vs. stateful)


class DeepARModel(Model):
    def __init__(
        self,
        num_features: int = None,
        train_window: int = 20,
        cardinalities: typing.List[int] = None,
        emb_dim: typing.List[int] = None,
        output_dim: int = 1,
        lstm_dim: int = 40,
        num_layers: int = 2,
        batch_size: int = 16,
        dropout: float = 0.1,
        init: str = "glorot_uniform",
        count_data: bool = False,
    ):
        """ initialize DeepAR model
        
        Keyword Arguments:
            num_features {int} -- number of input features (covariates + AR terms) (default: {None})
            train_window {int} -- the length of time series sampled for training. consistent throughout (default: {20})
            cardinalities {typing.List[int]} -- cardinalities of categorical features (default: {None})
            emb_dim {typing.List[int]} -- dimension of categorical embeddings, if None, dimension of each
                    embedding will be min(50, (cardinality + 1) / 2) (default: {None})
            output_dim {int} -- output dimension (default: {1})
            lstm_dim {int} -- dimension of lstm cells  (default: {40})
            num_layers {int} -- number of lstm layers (default: {2})
            batch_size {int} -- number of time series to sample in each batch, needed in constructor of 
                stateful LSTM object (default: {16})
            dropout {float} -- dropout (default: {0.1})
            init {str} -- initializer for weights in network (default: {"glorot_uniform"})
            count_data {bool} -- whether target values are count (positive reals) or continuous (default: {True})
        """
        super(DeepARModel, self).__init__()

        self.count_data = count_data

        if emb_dim is None:
            emb_dim = [math.ceil(min(50, (card + 1) / 2)) for card in cardinalities]
        else:
            if len(emb_dim) != len(cardinalities):
                raise ValueError(
                    "Length of list of embedding dimensions must be the same as the number of categorical features in ts_obj"
                )

        self.embeddings = [
            Embedding(card, dim, embeddings_initializer=init)
            for card, dim in zip(cardinalities, emb_dim)
        ]

        # hierarchical lstm
        self.lstms = [
            LSTM(
                lstm_dim,
                return_sequences=True,
                stateful=True,
                dropout=dropout,
                recurrent_dropout=dropout,
                kernel_initializer=init,
                recurrent_initializer=init,
                unit_forget_bias=True,
            )
            for i in range(0, num_layers)
        ]
        # output parameter transforms
        self.mu = Dense(
            output_dim, kernel_initializer=init, bias_initializer=init, name="mu",
        )
        self.sigma = Dense(
            output_dim, kernel_initializer=init, bias_initializer=init, name="sigma",
        )

        # must specify lstm input shapes bc need stateful lstm for inference
        self.lstms[0].build((batch_size, train_window, num_features + sum(emb_dim)))
        [lstm.build((batch_size, train_window, lstm_dim)) for lstm in self.lstms[1:]]

    def call(
        self, inputs: typing.List[tf.Tensor], training: bool = False
    ) -> typing.Tuple[tf.Tensor]:
        """ forward pass through model
        
        Arguments:
            inputs {[typing.List[tf.Tensor]]} -- [X_continouous, X_categorical_0, X_categorical_1, ... X_categorical_n]
        
        Keyword Arguments:
            training {bool} -- whether forward pass should be conducted in training mode (dropout on) (default: {False})
        
        Returns:
            typing.Tuple[tf.Tensor] -- shape and scale parameters of distribution
        """

        cont_inputs = [inputs[0]]
        cat_inputs = inputs[1:]

        # generate paramaters
        embedded = [
            embedding(cat) for embedding, cat in zip(self.embeddings, cat_inputs)
        ]
        x = tf.concat(cont_inputs + embedded, axis=-1)
        for lstm in self.lstms:
            if training:
                lstm.reset_states()
            x = lstm(x, training=training)
        mu = self.mu(x)

        # softplus according to dtype
        sigma = softplus(self.sigma(x))
        if self.count_data:
            mu = softplus(mu)

        return mu, sigma
    
    def reset_lstm_states(
        self, 
    ):
        """ reset all lstm states to values at which they were initialized
        """
        [lstm.reset_states() for lstm in self.lstms]
