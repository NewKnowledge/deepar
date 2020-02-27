import typing
import math

import numpy as np
import pandas as pd

import tensorflow as tf
from tensorflow.keras.losses import Loss, Reduction
from tensorflow.keras.activations import softplus


def unscale(
    mu: tf.Tensor, sigma: tf.Tensor, scale_values: tf.Tensor,
) -> typing.Tuple[tf.Tensor, tf.Tensor]:
    """ unscales predictions (mu and scale) using scale_keys and values
    
    Arguments:
        mu {tf.Tensor} -- mu
        scale {tf.Tensor} -- scale
        scale_values {tf.Tensor} -- values that should unscale mu and scale
            unscaled_mu = mu * scale_values
            unscaled_sigma = sigma / sqrt(scale_values)
    
    Returns:
        typing.Tuple[tf.Tensor, tf.Tensor] -- (unscaled mu, unscaled scale)
    """

    # unscale mu and sigma
    scale_values = tf.expand_dims(tf.expand_dims(scale_values, -1), -1)
    unscaled_mu = tf.multiply(mu, scale_values)
    unscaled_sigma = tf.divide(sigma, tf.sqrt(scale_values))
    return unscaled_mu, unscaled_sigma


class GaussianLogLikelihood(Loss):
    """ Custom GaussianLogLikelihood loss function
    
    Keyword Arguments:
        mask_value {int} -- value in y_true that should be masked in loss (missing tgt in training set) (default: {0})
    """

    def __init__(
        self, mask_value: int = 0,
    ):

        super(GaussianLogLikelihood, self).__init__(
            reduction=Reduction.AUTO, name="gaussian_log_likelihood"
        )
        self._mask_value = mask_value

    def _mask_loss(self, loss_term: tf.Tensor, y_true: tf.Tensor,) -> tf.Tensor:

        """
        util function
            mask loss tensor (y_true) according to mask_value locations
        """

        # mask loss with mask_value
        mask = tf.not_equal(y_true, tf.constant(self._mask_value, dtype=tf.float32))
        mask = tf.dtypes.cast(mask, tf.float32)
        return tf.multiply(loss_term, mask)

    ## TODO check numerical stability!!
    def call(
        self, y_true: tf.Tensor, y_pred_bundle: typing.Tuple[tf.Tensor, tf.Tensor]
    ) -> tf.Tensor:

        """ calculates loss
        
        Arguments:
            y_true {tf.Tensor} -- true values
            y_pred_bundle {typing.Tuple[tf.Tensor, tf.Tensor]} -- tuple of (mu, sigma)
        
        Returns:
            tf.Tensor -- loss
        """

        mu, sigma = y_pred_bundle
        batch_size = mu.shape[0]

        # loss
        loss_term = 0.5 * tf.math.log(sigma) + 0.5 * tf.divide(
            tf.square(y_true - mu), sigma
        )

        # mask
        masked_loss_term = self._mask_loss(loss_term, y_true)

        # divide by batch size bc auto reduction will sum over batch size
        return (masked_loss_term + 1e-6 + 6) / batch_size


class NegativeBinomialLogLikelihood(Loss):
    """ Custom NegativeBinomialLogLikelihood loss function
    
    Keyword Arguments:
        mask_value {int} -- value in y_true that should be masked in loss (missing tgt in training set) (default: {0})
    """

    def __init__(
        self, mask_value: int = 0,
    ):

        super(NegativeBinomialLogLikelihood, self).__init__(
            reduction=Reduction.AUTO, name="negative_binomial_log_likelihood"
        )
        self._mask_value = mask_value

    def _mask_loss(self, loss_term: tf.Tensor, y_true: tf.Tensor,) -> tf.Tensor:

        """
        util function
            mask loss tensor (y_true) according to mask_value locations
        """

        # mask loss with mask_value
        mask = tf.not_equal(y_true, tf.constant(self._mask_value, dtype=tf.float32))
        mask = tf.dtypes.cast(mask, tf.float32)
        return tf.multiply(loss_term, mask)

    def call(
        self, y_true: tf.Tensor, y_pred_bundle: typing.Tuple[tf.Tensor, tf.Tensor]
    ) -> tf.Tensor:

        """ calculates loss
        
        Arguments:
            y_true {tf.Tensor} -- true values
            y_pred_bundle {typing.Tuple[tf.Tensor, tf.Tensor]} -- tuple of (mu, sigma)
        
        Returns:
            tf.Tensor -- loss
        """

        mu, alpha = y_pred_bundle
        batch_size = mu.shape[0]

        # loss
        alpha_y_pred = tf.multiply(alpha, mu)
        alpha_div = tf.divide(1.0, alpha)
        denom = tf.math.log(1 + alpha_y_pred)
        log_loss = (
            tf.math.lgamma(y_true + alpha_div)
            - tf.math.lgamma(y_true + 1.0)
            - tf.math.lgamma(alpha_div)
        )
        loss_term = (
            log_loss
            - tf.divide(denom, alpha)
            + tf.multiply(y_true, tf.math.log(alpha_y_pred) - denom)
        )
        loss_term = -loss_term

        # mask
        masked_loss_term = self._mask_loss(loss_term, y_true)

        # divide by batch size bc auto reduction will sum over batch size
        return (masked_loss_term + 1e-6 + 6) / batch_size
