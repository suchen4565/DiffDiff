from typing import Optional, Union

import numpy as np


def mqloss(
    y: np.ndarray,
    y_hat: np.ndarray,
    quantiles: np.ndarray,
    weights: Optional[np.ndarray] = None,
    axis: Optional[int] = None,
) -> Union[float, np.ndarray]:
    """Multi-quantile (pinball) loss.

    Discretized CRPS via a left-Riemann sum over uniformly spaced quantiles.
    """
    if weights is None:
        weights = np.ones(y.shape)

    n_q = len(quantiles)
    y_rep = np.expand_dims(y, axis=-1)
    error = y_hat - y_rep
    sq = np.maximum(-error, np.zeros_like(error))
    s1_q = np.maximum(error, np.zeros_like(error))
    loss = quantiles * sq + (1 - quantiles) * s1_q

    weights = np.repeat(np.expand_dims(weights, axis=-1), repeats=n_q, axis=-1)
    return np.average(loss, weights=weights, axis=axis)


def mae(y: np.ndarray, y_hat: np.ndarray, axis: Optional[int] = None):
    return np.nanmean(np.abs(y - y_hat), axis=axis)


def mse(y: np.ndarray, y_hat: np.ndarray, axis: Optional[int] = None):
    return np.nanmean(np.square(y - y_hat), axis=axis)


def calculate_metrics(y_pred, y_real, quantiles=(np.arange(9) + 1) / 10):
    """Return (MAE, MSE, MQL) plus the quantile / point predictions used."""
    y_pred = y_pred.cpu().numpy()
    y_real = y_real.cpu().numpy()

    assert y_pred.ndim == 4
    assert y_pred[0].shape == y_real.shape

    y_pred_point = np.mean(y_pred, axis=0)
    y_pred_q = np.quantile(y_pred, quantiles, axis=0)
    y_pred_q = np.transpose(y_pred_q, (1, 2, 3, 0))

    MAE = mae(y_real, y_pred_point)
    MSE = mse(y_real, y_pred_point)
    MQL = mqloss(y_real, y_pred_q, quantiles=np.array(quantiles))
    return (MAE, MSE, MQL), y_pred_q, y_pred_point
