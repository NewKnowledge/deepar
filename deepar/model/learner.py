import logging
import os
import time
import math
import typing

import numpy as np

import tensorflow as tf
from tensorflow.keras.optimizers import Adam, SGD
from tensorflow.keras.callbacks import (
    # ModelCheckpoint,
    # LearningRateScheduler,
    # ReduceLROnPlateau,
    TensorBoard,
)
from tensorflow.keras.metrics import RootMeanSquaredError, MeanAbsoluteError, Mean

from deepar.model.loss import (
    unscale,
    GaussianLogLikelihood,
    NegativeBinomialLogLikelihood,
)
from deepar.dataset.time_series import (
    TimeSeriesTrain,
    TimeSeriesTest,
    train_ts_generator,
    test_ts_generator,
)
from deepar.model.models import DeepARModel
from deepar.model.callbacks import EarlyStopping

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


class DeepARLearner:
    def __init__(
        self,
        ts_obj: TimeSeriesTrain,
        output_dim: int = 1,
        emb_dim: typing.List[int] = None,
        lstm_dim: int = 40,
        num_layers: int = 2,
        dropout: float = 0.1,
        optimizer: str = "adam",
        clip_gradient: int = 10,
        lr: float = 0.001,
        batch_size: int = 16,
        init: str = "glorot_uniform",
        # scheduler=None,
        train_window: int = 20,
        random_seed: int = None,
        hparams: typing.Dict[str, typing.Union[int, float]] = None,
    ):
        """ initialize DeepAR model
        
        Arguments:
            ts_obj {TimeSeriesTrain} -- training dataset object
        
        Keyword Arguments:
            output_dim {int} -- dimension of output variables (default: {1})
            emb_dim {int} -- dimension of categorical embeddings, if None, dimension of each
                embedding will be min(50, (cardinality + 1) / 2) (default: {None})
            lstm_dim {int} -- dimension of lstm cells (default: {40})
            num_layers {int} -- number of lstm layers (default: {2})
            dropout {float} -- dropout (default: {0.1})
            optimizer {str} -- options are "adam" and "sgd" (default: {"adam"})
            clip_gradient {int} -- absolute value at which to clip gradients (default: {10})
            lr {float} -- learning rate (default: {0.001})
            batch_size {int} -- number of time series to sample in each batch (default: {16})
            init {str} -- initializer for weights in network (default: {"glorot_uniform"})
            train_window {int} -- the length of time series sampled for training. consistent throughout (default: {20})
            random_seed {int} -- optional random seed to control randomness throughout learner (default: {None})
            hparams {typing.Dict[str, typing.Union[int, float]]} -- 
                dict of hyperparameters (keys) and hp domain (values) over which to hp search (default: {None})
        """

        if random_seed is not None:
            tf.random.set_seed(random_seed)

        if hparams is None:
            self._lr = lr
            self._train_window = train_window
            self._batch_size = batch_size
        else:
            self._lr = hparams["learning_rate"]
            self._train_window = hparams["window_size"]
            self._batch_size = hparams["batch_size"]
            emb_dim = hparams["emb_dim"]
            lstm_dim = hparams["lstm_dim"]
            dropout = hparams["lstm_dropout"]

        # Invariance - Window size must be <= max_series_length + negative observations to respect covariates
        if self._train_window > ts_obj.max_age:
            logger.info(
                f"Training set with max observations {ts_obj.max_age} does not support train window size of "
                + f"{self._train_window}, resetting train window to {ts_obj.max_age}"
            )
            self._train_window = ts_obj.max_age
        # self.scheduler = scheduler
        self._ts_obj = ts_obj

        # define model
        self._model = DeepARModel(
            num_features=self._ts_obj.features,
            train_window=self._train_window,
            cardinalities=self._ts_obj.cardinalities,
            emb_dim=emb_dim,
            output_dim=output_dim,
            lstm_dim=lstm_dim,
            num_layers=num_layers,
            batch_size=batch_size,
            dropout=dropout,
            init=init,
            count_data=self._ts_obj.count_data,
        )

        # define optimizer
        if optimizer == "adam":
            self._optimizer = Adam(learning_rate=self._lr, clipvalue=clip_gradient)
        elif optimizer == "sgd":
            self._optimizer = SGD(
                lr=self._lr, momentum=0.1, nesterov=True, clipvalue=clip_gradient
            )
        else:
            raise ValueError("Optimizer must be one of `adam` and `sgd`")

        # define loss function
        if self._ts_obj.count_data:
            self._loss_fn = NegativeBinomialLogLikelihood(
                mask_value=self._ts_obj.mask_value
            )
        else:
            self._loss_fn = GaussianLogLikelihood(mask_value=self._ts_obj.mask_value)

    def save_weights(self, filepath: str):
        """ saves model's current weights to filepath
        
        Arguments:
            filepath {str} -- filepath
        """
        self._model.save_weights(filepath)

    def load_weights(self, filepath: str):
        """ loads model's current weights from filepath
        
        Arguments:
            filepath {str} -- filepath
        """
        self._model.load_weights(filepath)

    def _training_loop(
        self,
        filepath: str,
        train_gen: train_ts_generator,  # can name of function be type?
        val_gen: train_ts_generator,
        epochs: int = 100,
        steps_per_epoch: int = 50,
        early_stopping: int = True,
        stopping_patience: int = 5,
        stopping_delta: int = 1,
    ) -> typing.Tuple[tf.Tensor, int]:
        """ 
        util function
            iterates over batches, updates gradients, records metrics, writes to tb, checkpoints, early stopping
        """

        # set up metrics to track during training
        batch_loss_avg = Mean()
        epoch_loss_avg = Mean()
        eval_loss_avg = Mean()
        eval_mae = MeanAbsoluteError()
        eval_rmse = RootMeanSquaredError()

        # set up early stopping callback
        early_stopping_cb = EarlyStopping(
            patience=stopping_patience, active=early_stopping, delta=stopping_delta
        )

        # Iterate over epochs.
        best_metric = math.inf
        for epoch in range(epochs):
            logger.info(f"Start of epoch {epoch}")
            start_time = time.time()
            import sys

            for batch, (x_batch_train, scale_values, y_batch_train) in enumerate(
                train_gen
            ):
                # compute loss
                with tf.GradientTape(persistent=True) as tape:
                    mu, scale = self._model(x_batch_train, training=True)
                    
                    mu, scale = unscale(
                        mu, scale, scale_values
                    )
                    loss_value = self._loss_fn(y_batch_train, (mu, scale))

                # sgd
                if self._tb:
                    tf.summary.scalar(
                        "train_loss", loss_value, epoch * steps_per_epoch + batch
                    )
                batch_loss_avg(loss_value)
                epoch_loss_avg(loss_value)
                grads = tape.gradient(loss_value, self._model.trainable_weights)
                self._optimizer.apply_gradients(
                    zip(grads, self._model.trainable_weights)
                )

                # Log 5x per epoch.
                if batch % (steps_per_epoch // 5) == 0 and batch != 0:
                    logger.info(
                        f"Epoch {epoch}: Avg train loss over last {(steps_per_epoch // 5)} steps: {batch_loss_avg.result()}"
                    )
                    batch_loss_avg.reset_states()

                # Run each epoch batches times
                epoch_loss_avg_result = epoch_loss_avg.result()
                if batch == steps_per_epoch:
                    logger.info(
                        f"Epoch {epoch} took {round(time.time() - start_time, 0)}s : Avg train loss: {epoch_loss_avg_result}"
                    )
                    break

            # validation
            if val_gen is not None:
                logger.info(f"End of epoch {epoch}, validating...")
                start_time = time.time()
                for batch, (x_batch_val, scale_values, y_batch_val) in enumerate(val_gen):

                    # compute loss, doesn't need to be persistent bc not updating weights
                    with tf.GradientTape() as tape:

                        # treat as training -> reset lstm states inbetween each batch
                        mu, scale = self._model(x_batch_val, training=True)

                        # unscale parameters
                        mu, scale = unscale(
                            mu, scale, scale_values
                        )

                        # calculate loss
                        loss_value = self._loss_fn(y_batch_val, (mu, scale))

                    # log validation metrics (avg loss, avg MAE, avg RMSE)
                    eval_mae(y_batch_val, mu)
                    eval_rmse(y_batch_val, mu)
                    eval_loss_avg(loss_value)
                    if batch == steps_per_epoch:
                        break

                # logging
                eval_mae_result = eval_mae.result()
                logger.info(f"Validation took {round(time.time() - start_time, 0)}s")
                logger.info(
                    f"Epoch {epoch}: Val loss on {steps_per_epoch} steps: {eval_loss_avg.result()}"
                )
                logger.info(
                    f"Epoch {epoch}: Val MAE: {eval_mae_result}, RMSE: {eval_rmse.result()}"
                )
                if self._tb:
                    tf.summary.scalar("val_loss", eval_loss_avg.result(), epoch)
                    tf.summary.scalar("val_mae", eval_mae_result, epoch)
                    tf.summary.scalar("val_rmse", eval_rmse.result(), epoch)
                new_metric = eval_mae_result

                # early stopping
                if early_stopping_cb(eval_mae_result):
                    break

                # reset metric states
                eval_loss_avg.reset_states()
                eval_mae.reset_states()
                eval_rmse.reset_states()
            else:
                if early_stopping_cb(epoch_loss_avg_result):
                    break
                new_metric = epoch_loss_avg_result

            # update best_metric and save new checkpoint if improvement
            if new_metric < best_metric:
                best_metric = new_metric
                if filepath is not None:
                    self._checkpointer.save(file_prefix=filepath)
                else:
                    self.save_weights("model_best_weights.h5")

            # reset epoch loss metric
            epoch_loss_avg.reset_states()

        # load in best weights before returning if not using checkpointer
        if filepath is None:
            self.load_weights("model_best_weights.h5")
            os.remove("model_best_weights.h5")
        return best_metric, epoch + 1

    def fit(
        self,
        checkpoint_dir: str = None,
        validation: bool = True,
        steps_per_epoch: int = 50,
        epochs: int = 100,
        early_stopping: bool = True,
        stopping_patience: int = 5,
        stopping_delta: int = 1,
        tensorboard: bool = True,
    ) -> typing.Tuple[tf.Tensor, int]:

        """ fits DeepAR model for steps_per_epoch * epochs iterations
        
        Keyword Arguments:
            checkpoint_dir {str} -- directory to save checkpoint and tensorboard files (default: {None})
            validation {bool} -- whether to perform validation. If True will automatically try
                to use validation data sequestered in construction of time series object(default: {True})
            steps_per_epoch {int} -- number of steps to process each epoch (default: {50})
            epochs {int} -- number of epochs (default: {100})
            early_stopping {bool} -- whether to include early stopping callback (default: {True})
            stopping_patience {int} -- early stopping callback patience, measured in epochs (default: {5})
            stopping_delta {int} -- early stopping delta, the comparison to determine stopping will be
                previous metric vs. new metric +- stopping_delta (default: {1})
            tensorboard {bool} -- whether to write output to tensorboard logs (default: {True})
        
        Returns: 
            Tuple(float, int) --    1) final_metric best (train loss or eval MAE) after fitting
                                    2) number of iterations completed (might have been impacted by stopping criterion) 
        """

        self._epochs = epochs

        # try to load previously saved checkpoint from filepath
        self._checkpointer = tf.train.Checkpoint(
            optimizer=self._optimizer, model=self._model
        )
        if checkpoint_dir is not None:
            if not os.path.exists(checkpoint_dir):
                os.makedirs(checkpoint_dir)
            filepath = os.path.join(checkpoint_dir, "{epoch:04d}.ckpt")
            latest_ckpt = tf.train.latest_checkpoint(checkpoint_dir)
            if latest_ckpt:
                self._checkpointer.restore(latest_ckpt)
        elif tensorboard:
            # create tensorboard log files in default location
            checkpoint_dir = "./tb/"
            tb_writer = tf.summary.create_file_writer(checkpoint_dir)
            tb_writer.set_as_default()
        else:
            filepath = None
        self._tb = tensorboard

        # train generator
        train_gen = train_ts_generator(
            self._model,
            self._ts_obj,
            self._batch_size,
            self._train_window,
        )

        # validation generator
        if validation:
            val_gen = train_ts_generator(
                self._model,
                self._ts_obj,
                self._batch_size,
                self._train_window,
                val_set=True,
            )
        else:
            val_gen = None

        # Iterate over epochs.
        return self._training_loop(
            filepath,
            train_gen,
            val_gen,
            epochs=epochs,
            steps_per_epoch=steps_per_epoch,
            early_stopping=early_stopping,
            stopping_patience=stopping_patience,
            stopping_delta=stopping_delta,
        )

    def _add_prev_target(
        self, x_test: tf.Variable, prev_target: tf.Tensor
    ) -> typing.Tuple[tf.Variable, tf.Tensor]:

        """ private util function that replaces the previous target column in input data with 
            dynamically generated last target  
        """
        x_test_new = x_test[0][:, :1, -1:].assign(prev_target)
        return [x_test_new, *x_test[1:]]

    def _squeeze(
        self, mu: tf.Tensor, scale: tf.Tensor, squeeze_dims: typing.List[int] = [2]
    ) -> typing.Tuple[tf.Tensor, tf.Tensor]:

        """
        private util function that squeezes predictions along certain dimensions depending on whether
            we are predicting in-sample or out-of-sample
        """
        return tf.squeeze(mu, squeeze_dims), tf.squeeze(scale, squeeze_dims)

    def _negative_binomial(
        self, mu: tf.Tensor, scale: tf.Tensor, samples: int = 1
    ) -> np.ndarray:

        """
        private util function that draws n samples from a negative binomial distribution parameterized 
            by mu and scale parameters

            based on implementation from GluonTS library: 
            https://gluon-ts.mxnet.io/_modules/gluonts/distribution/neg_binomial.html#NegativeBinomial
        """
        tol = 1e-5
        r = 1.0 / scale
        theta = scale * mu
        r = tf.math.minimum(tf.math.maximum(tol, r), 1e10)
        theta = tf.math.minimum(tf.math.maximum(tol, theta), 1e10)
        p = 1 / (theta + 1)
        return np.random.negative_binomial(r, p, samples)

    def _draw_samples(
        self,
        mu_tensor: tf.Tensor,
        scale_tensor: tf.Tensor,
        point_estimate: int = False,
        samples: int = 1,
    ) -> typing.List[typing.List[np.ndarray]]:

        """
        private util function that draws samples from appropriate distribution
        """
        # shape : [# test groups, samples]
        if point_estimate:
            return [np.repeat(mu, samples) for mu in mu_tensor]
        elif self._ts_obj.count_data:
            return [
                list(self._negative_binomial(mu, scale, samples))
                for mu, scale in zip(mu_tensor, scale_tensor)
            ]
        else:
            return [
                list(np.random.normal(mu, scale, samples))
                for mu, scale in zip(mu_tensor, scale_tensor)
            ]

    def _reset_lstm_states(self):
        """ private util function to reset lstm states, dropout mask, and recurrent dropout mask
        """

    def predict(
        self,
        test_ts_obj: TimeSeriesTest,
        point_estimate: bool = True,
        horizon: int = None,
        samples: int = 100,
        include_all_training: bool = False,
        return_in_sample_predictions: bool = True,
    ) -> np.ndarray:

        """ predict horizon steps into the future
        
        Arguments:
            test_ts_obj {TimeSeriesTest} -- time series object for prediction
        
        Keyword Arguments:
            point_estimate {bool} -- if True, always sample mean of distributions, 
                otherwise sample from (mean, scale) parameters (default: {True})
            horizon {int} -- optional, can specify prediction horizon into future (default: {None})
            samples {int} -- how many samples to draw to calculate confidence intervals  (default: {100})
            include_all_training {bool} -- whether to start calculating hidden states from beginning 
                of training data, alternative is from t_0 - train_window (default: {False})
            return_in_sample_predictions {bool} -- 
                whether to return in sample predictions as well as future predictions (default: {True})
        
        Returns:
            np.ndarray -- predictions, shape is (# unique test groups, horizon, # samples)
        """

        assert samples > 0, "The number of samples to draw must be positive"

        # test generator
        test_gen = test_ts_generator(
            self._model,
            test_ts_obj,
            self._batch_size,
            self._train_window,
            include_all_training=include_all_training,
        )

        # reset lstm states before prediction
        self._model.reset_lstm_states()

        # make sure horizon is legitimate value
        if horizon is None or horizon > test_ts_obj.horizon:
            horizon = test_ts_obj.horizon

        # forward
        test_samples = [[] for _ in range(len(test_ts_obj.test_groups))]
        start_time = time.time()
        prev_iteration_index = 0

        for batch_idx, batch in enumerate(test_gen):

            x_test, scale_values, horizon_idx, iteration_index = batch
            if iteration_index is None:
                break
            if horizon_idx == horizon:
                test_ts_obj.batch_idx = 0
                test_ts_obj.iterations += 1
                continue

            # reset lstm states for new sequence of predictions through time
            if iteration_index != prev_iteration_index:
                self._model.reset_lstm_states()

            # don't need to replace for first test batch bc have tgt from last training example
            if horizon_idx > 1:
                # add one sample from previous predictions to test batches
                # all dim 0, first row of dim 1, last col of dim 3
                x_test = self._add_prev_target(x_test, mu[:, :1, :])

            # make predictions
            mu, scale = self._model(x_test)

            # unscale
            scaled_mu, scaled_scale = unscale(
                mu[: len(scale_values)],
                scale[: len(scale_values)],
                scale_values,
            )

            # in-sample predictions (ancestral sampling)
            if horizon_idx <= 0:
                if horizon_idx % 5 == 0:
                    logger.info(
                        f"Making in-sample predictions with ancestral sampling. {-horizon_idx} batches remaining"
                    )

                # squeeze 2nd dim - output_dim
                scaled_mu, scaled_scale = self._squeeze(scaled_mu, scaled_scale)
                for mu, scale, sample_list in zip(
                    scaled_mu,
                    scaled_scale,
                    test_samples[
                        iteration_index
                        * self._batch_size : (iteration_index + 1)
                        * self._batch_size
                    ],
                ):
                    draws = self._draw_samples(
                        mu, scale, point_estimate=point_estimate, samples=samples
                    )
                    sample_list.extend(draws)

            # draw samples from learned distributions for test samples
            else:
                if horizon_idx % 5 == 0:
                    logger.info(
                        f"Making future predictions. {horizon-horizon_idx} batches remaining"
                    )

                # get first column predictions (squeeze 1st dim - horizon)
                scaled_mu, scaled_scale = scaled_mu[:, :1, :], scaled_scale[:, :1, :]

                # slice at number of unique ts and squeeze 1st dim - horizon, 2nd dim - output_dim)
                squeezed_mu, squeezed_scale = self._squeeze(
                    scaled_mu, scaled_scale, squeeze_dims=[1, 2],
                )

                # concatenate
                draws = self._draw_samples(
                    squeezed_mu,
                    squeezed_scale,
                    point_estimate=point_estimate,
                    samples=samples,
                )
                for draw_list, sample_list in zip(
                    draws,
                    test_samples[
                        iteration_index
                        * self._batch_size : (iteration_index + 1)
                        * self._batch_size
                    ],
                ):
                    sample_list.append(draw_list)

        # reset batch idx and iterations_index so we can call predict() multiple times
        test_ts_obj.batch_idx = 0
        test_ts_obj.iterations = 0
        test_ts_obj.batch_test_data_prepared = False

        # filter test_samples depending on return_in_sample_predictions param
        if return_in_sample_predictions:
            pred_samples = np.array(test_samples)[
                :, -(self._ts_obj.max_age + horizon) :, :
            ]
        else:
            pred_samples = np.array(test_samples)[:, -horizon:, :]

        logger.info(
            f"Inference ({samples} sample(s), {horizon} timesteps) took {round(time.time() - start_time, 0)}s"
        )

        # Shape [# test_groups, horizon, samples]
        # TODO [# test_groups, horizon, samples, output_dim] not supported yet
        return pred_samples
