# Probabilistic LSTM Forecasting for SDE-Driven Time Series

This project demonstrates a methodology for probabilistic time series forecasting using a Long Short-Term Memory (LSTM) network, specifically applied to data simulated from a Stochastic Differential Equation (SDE) akin to a Wiener process with drift (Arithmetic Brownian Motion). The goal is to predict not just the next value in the series but also the uncertainty associated with that prediction.

## Methodology Overview

The core approach involves the following steps:

1.  **Data Simulation:**
    A time series is generated using the Euler-Maruyama method to approximate an SDE of the form $dX_t = \mu dt + \sigma dW_t$. This serves as the ground truth data with known underlying stochastic properties (constant drift $\mu$ and diffusion $\sigma$).

2.  **Data Preprocessing:**
    The simulated time series is:
    * Scaled (e.g., using Min-Max scaling) to a normalized range.
    * Transformed into input sequences (of a fixed length) and corresponding target values (the next point in the series).
    * Split into training and testing sets.

3.  **Probabilistic LSTM Model:**
    An LSTM network is designed to output two parameters for each input sequence:
    * $\hat{\mu}'$: The predicted mean of the (scaled) next value in the time series.
    * $\log(\hat{\sigma}'^2)$: The predicted log-variance of the (scaled) next value.
    These parameters define a Gaussian distribution $N(\hat{\mu}', \hat{\sigma}'^2)$ for the prediction.

4.  **Hybrid Loss Function for Training:**
    The model is trained by minimizing a composite loss function that includes:
    * **Gaussian Negative Log-Likelihood (NLL):** Encourages the predicted distribution to match the observed data.
        $L_{\text{NLL}} = \frac{1}{2}\left(\log(2\pi\hat{\sigma}'^2) + \frac{(Y' - \hat{\mu}')^2}{\hat{\sigma}'^2}\right)$
    * **Mean Squared Error (MSE) on the Mean:** Penalizes errors in the point prediction of the mean.
        $L_{\text{MSE}} = (Y' - \hat{\mu}')^2$
    * **SDE Parameter Consistency (PINN-like) Loss:** A physics-informed term that encourages the mean and variance of the *predicted scaled increments* (derived from the LSTM's outputs) to align with the known (scaled) drift and diffusion parameters of the underlying SDE. This component uses MSE to compare the model's implied increment statistics to the target statistics derived from
    
    $\mu_{\text{true}}$ and $\sigma_{\text{true}}$.

    $L_{\text{SDE-Consist}} = \lambda_{\text{drift}} (\hat{m}'_{\Delta} - m'_{\Delta, \text{target}})^2 + \lambda_{\text{diffusion}} (\hat{v}'_{\Delta} - v'_{\Delta, \text{target}})^2$
    The total loss is a weighted sum of these components.

5.  **Evaluation and Uncertainty Quantification:**
    The trained model is evaluated on a test set using:
    * Point forecast metrics (MSE, MAE) on the original data scale.
    * Probabilistic forecast metrics (NLL, Prediction Interval Coverage Probability - PICP, Average Width of Prediction Intervals - AWPI).
    Predictions are visualized with mean forecasts and uncertainty intervals.

## Purpose

This notebook serves as an example of how to build LSTMs for probabilistic time series forecasting and how to incorporate domain knowledge (in this case, the known SDE parameters) into the training process through a hybrid, physics-informed loss function. It highlights a method to achieve both accurate predictions and reliable uncertainty estimates.
