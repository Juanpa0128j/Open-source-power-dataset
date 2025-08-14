# Load Forecasting Benchmark Models

This folder contains a suite of benchmark models for load, wind, and solar forecasting. Each model is implemented as a standalone Python script and shares common utilities via `utils.py`. Models range from statistical baselines to advanced deep learning architectures.

## Model List

- **Statistical Models:**  
  - `arima.py`: ARIMA/VAR (AutoRegressive Integrated Moving Average, Vector AutoRegression)
  - `exponential_smoothing.py`: Simple, Holt, Holt-Winters Exponential Smoothing
  - `naive.py`: Naive persistence baseline
  - `linear_regression.py`: Linear regression

- **Machine Learning Models:**  
  - `svr.py`: Support Vector Regression
  - `random_forest.py`: Random Forest
  - `gradient_boosting.py`: Gradient Boosting

- **Deep Learning Models:**  
  - `FNN.py`: Feedforward Neural Network
  - `EML.py`: Extreme Learning Machine (last layer tuning)
  - `cnn.py`: 1D Convolutional Neural Network
  - `rnn.py`: RNN/GRU/LSTM
  - `LSTNet.py`: LSTNet (CNN+RNN+Skip+Highway)
  - `DeepAR.py`: Probabilistic LSTM (DeepAR)
  - `WaveNet.py`: Dilated CNN (WaveNet)
  - `NBeats.py`: N-BEATS (backcast/forecast blocks)
  - `NeuralODE.py`: Neural ODE for time series
  - `tcn.py`: Temporal Convolutional Network
  - `transformer.py`: Vanilla Transformer
  - `informer.py`: Informer (efficient transformer for long sequences)

## Theoretical Overview

- **Statistical Models:**  
  - *ARIMA/VAR*: Classical time series models for univariate/multivariate forecasting. ARIMA captures autocorrelation, trend, and seasonality; VAR extends to multiple variables.
  - *Exponential Smoothing*: Weighted averages of past observations, with variants for trend and seasonality.
  - *Naive*: Uses the last observed value as the forecast.

---

### ARIMA Model (`arima.py`)

#### Theoretical Background

ARIMA (AutoRegressive Integrated Moving Average) is a foundational statistical model for time series forecasting. It models a time series using three components:
- **AR (p)**: Autoregressive part, using past values.
- **I (d)**: Integrated part, differencing to achieve stationarity.
- **MA (q)**: Moving average part, using past forecast errors.

When `d=0`, ARIMA reduces to ARMA. For multivariate forecasting, the code uses VAR (Vector AutoRegression), which models multiple time series jointly.

Features:
- Supports both univariate and multivariate forecasting.
- Can incorporate external features (exogenous variables).
- Uses grid search to select optimal hyperparameters (`p`, `d`, `q`, sliding window, external features).
- Handles multiple prediction horizons (e.g., short-term, long-term).
- Selects hyperparameters based on metrics like RMSE, MAE, or MAPE.

#### Code Implementation

- **Parameter Grid Setup**: Uses `sklearn.model_selection.ParameterGrid` to generate combinations of hyperparameters.
- **Validation**: For each parameter set, trains ARIMA/VAR models on training data, evaluates on validation set, and records metrics.
- **Hyperparameter Selection**: Chooses the best parameter set for each prediction horizon based on the lowest error metric.
- **Testing**: Applies the selected models to the test set, generates forecasts, and computes prediction intervals.
- **External Features**: If enabled, includes additional columns as exogenous variables in ARIMA/VAR.
- **Multi-Task Support**: Handles load, wind, and solar forecasting in a unified workflow.
- **Output**: Saves predictions and configuration to the logs directory for later analysis.
- **CLI Usage**: Accepts arguments for all key parameters; merges with YAML config for reproducibility.

**Typical Workflow:**
1. Load configuration and data.
2. Generate parameter grid.
3. For each file and parameter set:
   - Train and validate models.
   - Select best hyperparameters.
   - Test and save predictions.
4. Evaluate results and save metrics.

**Example Command:**
```bash
python models/arima.py --sliding_window 24 --p_values [1,2,3] --d_values [0,1] --q_values [0,1] --variate ['uni','multi'] --external_feature_flag True --external_features ['temp','humidity']
```

**References:**
- [Statsmodels ARIMA documentation](https://www.statsmodels.org/stable/generated/statsmodels.tsa.arima.model.ARIMA.html)

---

### CNN Model (`cnn.py`)

#### Theoretical Background

The 1D Convolutional Neural Network (CNN) is a deep learning architecture designed for time series forecasting. It applies convolutional filters across temporal windows to extract local patterns and features from historical and external data. Key components include:
- **Convolutional Layers**: Learn filters to capture short-term dependencies and local trends.
- **Pooling Layers**: Downsample feature maps (max, average, adaptive, lp pooling) to reduce dimensionality and focus on salient features.
- **Nonlinear Activation**: LeakyReLU introduces nonlinearity for complex pattern modeling.
- **Dropout**: Regularizes the network to prevent overfitting.
- **Fully Connected Layers**: Map extracted features to output predictions for each forecast horizon.
- **Normalization**: Optional min-max normalization for output scaling.

Features:
- Supports multi-layer, multi-channel convolutions.
- Flexible pooling strategies.
- Handles external features and multiple targets.
- Grid search for architecture and training hyperparameters.

#### Code Implementation

- **CNNNet Class**: Defines the CNN architecture, including convolutional, pooling, activation, dropout, and fully connected layers.
- **CNN_exp Class**: Experiment wrapper for training, validation, and testing, inheriting from FNN_exp.
- **Parameter Grid Setup**: Uses `sklearn.model_selection.ParameterGrid` to explore combinations of dropout, hidden layers, kernel size, stride, pooling, batch size, learning rate, normalization, and sliding window.
- **Validation/Testing**: For each parameter set, trains the CNN on training data, validates on validation set, and selects the best configuration based on error metrics.
- **Hyperparameter Selection**: Chooses the best model using validation performance and saves parameters.
- **Testing**: Applies the selected model to the test set and saves predictions.
- **CLI Usage**: Accepts arguments for all key parameters; merges with YAML config for reproducibility.

**Typical Workflow:**
1. Load configuration and data.
2. Generate parameter grid.
3. For each file and parameter set:
   - Train and validate CNN models.
   - Select best hyperparameters.
   - Test and save predictions.
4. Evaluate results and save metrics.

**Example Command:**
```bash
python models/cnn.py --sliding_window 24 --hidden_layers "[32,64]" --kernel_size "[3,3]" --stride "[1,1]" --pooling "max" --dropout 0.2 --batch_size 64 --learning_rate 0.001 --normalization "minmax" --external_feature_flag True --external_features "['temp','humidity']"
```

**References:**
- [PyTorch 1D Convolution documentation](https://pytorch.org/docs/stable/generated/torch.nn.Conv1d.html)
- [Deep Learning for Time Series Forecasting](https://arxiv.org/abs/1809.10712)

---

### DeepAR Model (`DeepAR.py`)

#### Theoretical Background

DeepAR is a probabilistic forecasting model based on LSTM recurrent neural networks. It predicts the parameters (mean and standard deviation) of a Gaussian distribution for each forecasted value, enabling uncertainty quantification. Key features:
- **LSTM Layers**: Capture long-term temporal dependencies in time series data.
- **Distributional Output**: Predicts both mean (mu) and standard deviation (sigma) for each target, allowing interval forecasts.
- **Covariates/External Features**: Supports additional input features (e.g., weather, calendar) alongside historical values.
- **Multi-task Support**: Can forecast multiple targets (load, wind, solar) in a unified framework.
- **Probabilistic Loss**: Trains by maximizing the likelihood of observed values under the predicted distribution.

Features:
- Grid search for architecture and training hyperparameters.
- Handles normalization and inverse transformation for outputs.
- Supports both univariate and multivariate forecasting.

#### Code Implementation

- **DeepAR_model Class**: Defines the LSTM architecture, output layers for mu/sigma, and probabilistic loss function.
- **DeepAR_exp Class**: Experiment wrapper for training, validation, and testing, including data loading, model instantiation, and evaluation.
- **Parameter Grid Setup**: Uses `sklearn.model_selection.ParameterGrid` to explore combinations of sliding window, batch size, learning rate, normalization, dropout, hidden dimension, hidden layers, and external flag.
- **Validation/Testing**: For each parameter set, trains DeepAR on training data, validates on validation set, and selects the best configuration based on error metrics.
- **Hyperparameter Selection**: Chooses the best model using validation performance and saves parameters.
- **Testing**: Applies the selected model to the test set, saves mean, upper, and lower interval predictions.
- **CLI Usage**: Accepts arguments for all key parameters; merges with YAML config for reproducibility.

**Typical Workflow:**
1. Load configuration and data.
2. Generate parameter grid.
3. For each file and parameter set:
   - Train and validate DeepAR models.
   - Select best hyperparameters.
   - Test and save predictions (mean, upper/lower intervals).
4. Evaluate results and save metrics.

**Example Command:**
```bash
python models/DeepAR.py --sliding_window 24 --hidden_dim 64 --hidden_layers 2 --dropout 0.1 --batch_size 64 --learning_rate 0.001 --normalization "minmax" --external_flag True --external_features "['temp','humidity']"
```

**References:**
- [DeepAR: Probabilistic Forecasting with Autoregressive Recurrent Networks](https://arxiv.org/abs/1704.04110)
- [PyTorch LSTM documentation](https://pytorch.org/docs/stable/generated/torch.nn.LSTM.html)

---

### EML Model (`EML.py`)

#### Theoretical Background

Extreme Learning Machine (ELM) is a variant of feedforward neural networks where all layers except the last are frozen (not trainable), and only the final layer's weights are updated during training. This approach enables fast training and efficient regression, as the hidden layers act as random feature extractors and only the output layer adapts to the data.

Features:
- Only the last layer is trained; all previous layers are frozen.
- Fast training and low computational cost.
- Suitable for regression and time series forecasting tasks.
- Grid search for architecture and training hyperparameters.

#### Code Implementation

- **ELM_exp Class**: Inherits from FNN_exp, modifies the model so only the last layer is trainable (using `requires_grad`).
- **Parameter Grid Setup**: Uses `sklearn.model_selection.ParameterGrid` to explore combinations of sliding window, hidden size, number of layers, batch size, learning rate, dropout, normalization.
- **Validation/Testing**: For each parameter set, trains the ELM on training data, validates on validation set, and selects the best configuration based on error metrics.
- **Hyperparameter Selection**: Chooses the best model using validation performance and saves parameters.
- **Testing**: Applies the selected model to the test set and saves predictions.
- **CLI Usage**: Accepts arguments for all key parameters; merges with YAML config for reproducibility.

**Typical Workflow:**
1. Load configuration and data.
2. Generate parameter grid.
3. For each file and parameter set:
   - Train and validate ELM models.
   - Select best hyperparameters.
   - Test and save predictions.
4. Evaluate results and save metrics.

**Example Command:**
```bash
python models/EML.py --sliding_window 24 --hidden_size 128 --num_layers 3 --dropout 0.2 --batch_size 64 --learning_rate 0.001 --normalization "minmax"
```

**References:**
- [Extreme Learning Machine: Theory and Applications](https://ieeexplore.ieee.org/document/1380068)
- [PyTorch Feedforward documentation](https://pytorch.org/docs/stable/generated/torch.nn.Linear.html)

---

### Exponential Smoothing Model (`exponential_smoothing.py`)

#### Theoretical Background

Exponential smoothing is a family of time series forecasting methods that apply exponentially decreasing weights to past observations. This implementation includes:
- **Simple Exponential Smoothing**: Suitable for series with no trend or seasonality.
- **Holt’s Linear Trend Method**: Extends simple smoothing to capture linear trends.
- **Holt-Winters’ Seasonal Method**: Adds support for seasonality (additive or multiplicative).

Limitations:
- Does not support external features or multiple locations.
- Only univariate forecasting.

Features:
- Fast, interpretable statistical baselines.
- Grid search for method and sliding window hyperparameters.

#### Code Implementation

- **Validation/Testing**: For each parameter set (method, sliding window), trains the selected exponential smoothing model on training data, validates on validation set, and selects the best configuration based on error metrics.
- **Hyperparameter Selection**: Chooses the best model using validation performance and saves parameters.
- **Testing**: Applies the selected model to the test set and saves predictions (mean, upper/lower intervals).
- **CLI Usage**: Accepts arguments for all key parameters; merges with YAML config for reproducibility.

**Typical Workflow:**
1. Load configuration and data.
2. Generate parameter grid (method, sliding window).
3. For each file and parameter set:
   - Train and validate exponential smoothing models.
   - Select best hyperparameters.
   - Test and save predictions (mean, upper/lower intervals).
4. Evaluate results and save metrics.

**Example Command:**
```bash
python models/exponential_smoothing.py --sliding_window 24 --selection_metric "RMSE"
```

**References:**
- [Simple Exponential Smoothing](https://otexts.com/fpp2/ses.html)
- [Holt’s Linear Trend Method](https://otexts.com/fpp2/holt.html)
- [Holt-Winters’ Seasonal Method](https://otexts.com/fpp2/holt-winters.html)
- [Statsmodels Exponential Smoothing documentation](https://www.statsmodels.org/stable/generated/statsmodels.tsa.holtwinters.ExponentialSmoothing.html)

---

### FNN Model (`FNN.py`)

#### Theoretical Background

Feedforward Neural Networks (FNN), also known as multilayer perceptrons, are universal function approximators for regression and classification. In time series forecasting, FNNs learn nonlinear mappings from historical values and external features to future targets. They support multi-task forecasting and flexible architectures.

Features:
- Multiple hidden layers and nonlinear activations (LeakyReLU).
- Support for external features and multi-task outputs.
- Grid search for architecture and training hyperparameters.
- Optional normalization (minmax, standard).

#### Code Implementation

- **RegressorNet Class**: Defines the FNN architecture with configurable layers, hidden size, dropout, and normalization.
- **FNN_exp Class**: Experiment wrapper for training, validation, and testing, including data loading, model instantiation, and evaluation.
- **Parameter Grid Setup**: Uses `sklearn.model_selection.ParameterGrid` to explore combinations of sliding window, hidden size, number of layers, batch size, learning rate, dropout, normalization.
- **Validation/Testing**: For each parameter set, trains the FNN on training data, validates on validation set, and selects the best configuration based on error metrics.
- **Hyperparameter Selection**: Chooses the best model using validation performance and saves parameters.
- **Testing**: Applies the selected model to the test set and saves predictions.
- **CLI Usage**: Accepts arguments for all key parameters; merges with YAML config for reproducibility.

**Typical Workflow:**
1. Load configuration and data.
2. Generate parameter grid.
3. For each file and parameter set:
   - Train and validate FNN models.
   - Select best hyperparameters.
   - Test and save predictions.
4. Evaluate results and save metrics.

**Example Command:**
```bash
python models/FNN.py --sliding_window 24 --hidden_size 128 --num_layers 3 --dropout 0.2 --batch_size 64 --learning_rate 0.001 --normalization "minmax"
```

**References:**
- [PyTorch Feedforward documentation](https://pytorch.org/docs/stable/generated/torch.nn.Linear.html)
- [Universal Approximation Theorem](https://en.wikipedia.org/wiki/Universal_approximation_theorem)

---

### Gradient Boosting Model (`gradient_boosting.py`)

#### Theoretical Background

Gradient Boosting is an ensemble learning method that builds a series of decision trees, where each tree corrects the errors of the previous ones. It is highly effective for regression tasks and can handle multivariate outputs when multiple locations are considered. Compared to SVR, gradient boosting supports multivariate output natively.

Features:
- Ensemble of decision trees for robust regression.
- Supports multivariate output for multiple locations.
- Grid search for number of estimators, learning rate, and other hyperparameters.
- Can incorporate external features.

#### Code Implementation

- **grid_search_gradient_boosting Function**: Sets up parameter grid, trains and validates models, selects best hyperparameters, and tests on unseen data.
- **Parameter Grid Setup**: Uses `sklearn.model_selection.ParameterGrid` to explore combinations of sliding window, variate, external feature flag, number of estimators, and learning rate.
- **Validation/Testing**: For each parameter set, trains the gradient boosting model on training data, validates on validation set, and selects the best configuration based on error metrics.
- **Hyperparameter Selection**: Chooses the best model using validation performance and saves parameters.
- **Testing**: Applies the selected model to the test set and saves predictions (mean, upper/lower intervals).
- **CLI Usage**: Accepts arguments for all key parameters; merges with YAML config for reproducibility.

**Typical Workflow:**
1. Load configuration and data.
2. Generate parameter grid.
3. For each file and parameter set:
   - Train and validate gradient boosting models.
   - Select best hyperparameters.
   - Test and save predictions (mean, upper/lower intervals).
4. Evaluate results and save metrics.

**Example Command:**
```bash
python models/gradient_boosting.py --sliding_window 24 --n_estimators 100 --learning_rate 0.1 --external_feature_flag True --variate "multi"
```

**References:**
- [Gradient Boosting documentation (scikit-learn)](https://scikit-learn.org/stable/modules/ensemble.html#gradient-boosting)
- [Friedman, J.H. (2001). Greedy Function Approximation: A Gradient Boosting Machine](https://statweb.stanford.edu/~jhf/ftp/trebst.pdf)

---

### Informer Model (`informer.py`)

#### Theoretical Background

Informer is an efficient transformer-based architecture designed for long sequence time series forecasting. It introduces probabilistic attention to reduce computational cost and memory usage, enabling scalable modeling of long input sequences. Informer also uses distillation and multi-head attention for improved performance.

Features:
- Probabilistic attention for efficient long sequence modeling.
- Supports external features, time features, and multi-task outputs.
- Flexible encoder-decoder architecture with embedding layers.
- Grid search for architecture and training hyperparameters.

#### Code Implementation

- **Informer Class**: Implements the full Informer architecture, including token/positional/time embeddings, encoder/decoder layers, attention mechanisms, and output projection.
- **informer_exp Class**: Experiment wrapper for training, validation, and testing, including data loading, model instantiation, and evaluation.
- **Parameter Grid Setup**: Uses `sklearn.model_selection.ParameterGrid` to explore combinations of sliding window, batch size, learning rate, dropout, normalization, autoregressive flag, label length, attention type, model dimension, factor, number of heads/layers, activation, distillation, and mix.
- **Validation/Testing**: For each parameter set, trains Informer on training data, validates on validation set, and selects the best configuration based on error metrics.
- **Hyperparameter Selection**: Chooses the best model using validation performance and saves parameters.
- **Testing**: Applies the selected model to the test set and saves predictions.
- **CLI Usage**: Accepts arguments for all key parameters; merges with YAML config for reproducibility.

**Typical Workflow:**
1. Load configuration and data.
2. Generate parameter grid.
3. For each file and parameter set:
   - Train and validate Informer models.
   - Select best hyperparameters.
   - Test and save predictions.
4. Evaluate results and save metrics.

**Example Command:**
```bash
python models/informer.py --sliding_window 96 --batch_size 32 --learning_rate 0.001 --dropout 0.1 --normalization "minmax" --autoregressive True --label_len 48 --attn "prob" --d_model 512 --factor 5 --n_heads 8 --e_layers 3 --d_layers 2 --d_ff 512 --activation "gelu" --distil True --mix True --time_features "['month','day','hour']" --external_features "['temp','humidity']"
```

**References:**
- [Informer: Beyond Efficient Transformer for Long Sequence Time-Series Forecasting](https://arxiv.org/abs/2012.07436)
- [Official Informer code](https://github.com/zhouhaoyi/Informer2020)

---

### Linear Regression Model (`linear_regression.py`)

#### Theoretical Background

Linear regression is a classical statistical method for modeling the relationship between input features and target variables. In this benchmark, linear regression can produce multivariate output when multiple locations are considered, similar to gradient boosting and SVR.

Features:
- Fast, interpretable baseline for regression.
- Supports multivariate output for multiple locations.
- Can incorporate external features.
- Grid search for normalization and sliding window hyperparameters.

#### Code Implementation

- **grid_search_linear_regression Function**: Sets up parameter grid, trains and validates models, selects best hyperparameters, and tests on unseen data.
- **Parameter Grid Setup**: Uses `sklearn.model_selection.ParameterGrid` to explore combinations of sliding window, variate, external feature flag, and normalization.
- **Validation/Testing**: For each parameter set, trains the linear regression model on training data, validates on validation set, and selects the best configuration based on error metrics.
- **Hyperparameter Selection**: Chooses the best model using validation performance and saves parameters.
- **Testing**: Applies the selected model to the test set and saves predictions (mean, upper/lower intervals).
- **CLI Usage**: Accepts arguments for all key parameters; merges with YAML config for reproducibility.

**Typical Workflow:**
1. Load configuration and data.
2. Generate parameter grid.
3. For each file and parameter set:
   - Train and validate linear regression models.
   - Select best hyperparameters.
   - Test and save predictions (mean, upper/lower intervals).
4. Evaluate results and save metrics.

**Example Command:**
```bash
python models/linear_regression.py --sliding_window 24 --normalize True --external_feature_flag True --variate "multi"
```

**References:**
- [Linear Regression documentation (scikit-learn)](https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LinearRegression.html)
- [Regression Analysis](https://en.wikipedia.org/wiki/Regression_analysis)

---

### LSTNet Model (`LSTNet.py`)

#### Theoretical Background

LSTNet is a deep learning architecture for multivariate time series forecasting that combines convolutional neural networks (CNN), recurrent neural networks (RNN), skip connections, and highway networks. This hybrid design enables LSTNet to capture both short-term local patterns and long-term dependencies, as well as periodicity and trend.

Features:
- CNN layers extract local temporal features.
- RNN layers (GRU) capture long-term dependencies.
- Skip connections model periodic patterns.
- Highway component enables direct mapping of recent observations to outputs.
- Supports external features and multi-task outputs.
- Grid search for architecture and training hyperparameters.

#### Code Implementation

- **LSTNetModel Class**: Implements CNN, RNN, skip-RNN, and highway components, with configurable hidden sizes, kernel size, skip length, and highway window.
- **LSTNet_exp Class**: Experiment wrapper for training, validation, and testing, including data loading, model instantiation, and evaluation.
- **Parameter Grid Setup**: Uses `sklearn.model_selection.ParameterGrid` to explore combinations of sliding window, batch size, learning rate, normalization, dropout, CNN/RNN/skip hidden sizes, kernel size, skip length, and highway window.
- **Validation/Testing**: For each parameter set, trains LSTNet on training data, validates on validation set, and selects the best configuration based on error metrics.
- **Hyperparameter Selection**: Chooses the best model using validation performance and saves parameters.
- **Testing**: Applies the selected model to the test set and saves predictions.
- **CLI Usage**: Accepts arguments for all key parameters; merges with YAML config for reproducibility.

**Typical Workflow:**
1. Load configuration and data.
2. Generate parameter grid.
3. For each file and parameter set:
   - Train and validate LSTNet models.
   - Select best hyperparameters.
   - Test and save predictions.
4. Evaluate results and save metrics.

**Example Command:**
```bash
python models/LSTNet.py --sliding_window 24 --batch_size 64 --learning_rate 0.001 --dropout 0.2 --normalization "minmax" --hidRNN 100 --hidCNN 50 --hidSkip 10 --cnn_kernel 6 --skip 24 --highway_window 12 --external_features "['temp','humidity']"
```

**References:**
- [LSTNet: Modeling Long- and Short-Term Temporal Patterns with Deep Neural Networks](https://arxiv.org/abs/1703.07015)
- [Official LSTNet code](https://github.com/laiguokun/LSTNet)

---

### Naive Model (`naive.py`)

#### Theoretical Background

The Naive model is a simple persistence baseline for time series forecasting. It predicts future values by repeating the last observed value for all forecast horizons. This approach is fast, interpretable, and serves as a lower bound for model performance.

Features:
- No training required; simply copies the last value forward.
- Supports both single and multiple locations.
- Computes prediction intervals using residuals from the training set.
- Handles multiple prediction horizons (e.g., short-term, long-term).
- Fast and robust baseline for comparison.

#### Code Implementation

- **single_naive / multiple_naive Functions**: Generate predictions by repeating the last observed value for each horizon and location.
- **perform_naive_V3 Function**: Unified implementation for multi-task forecasting (load, wind, solar).
- **run_naive_V3 Function**: Loads configuration, iterates over files, generates predictions, computes intervals, and saves results.
- **Prediction Intervals**: Standard deviation of residuals is used to compute upper/lower bounds.
- **CLI Usage**: Accepts arguments for configuration and number of files; merges with YAML config for reproducibility.

**Typical Workflow:**
1. Load configuration and data.
2. For each file:
   - Generate naive predictions for all horizons.
   - Compute prediction intervals.
   - Save predictions.
3. Evaluate results and save metrics.

**Example Command:**
```bash
python models/naive.py --manual_seed 42 --num_files 3 --gpus "[1]"
```

**References:**
- [Persistence Forecasting](https://otexts.com/fpp2/simple-methods.html)
- [Naive Baseline for Time Series](https://en.wikipedia.org/wiki/Forecasting#Na%C3%AFve_forecast)

---

### N-BEATS Model (`NBeats.py`)

#### Theoretical Background

N-BEATS is a deep learning architecture for time series forecasting that uses stacked blocks with backcast/forecast decomposition. Each block refines the residuals from previous blocks, enabling interpretable and flexible modeling of complex temporal patterns. N-BEATS supports both univariate and multivariate forecasting, and can incorporate external features.

Features:
- Stacked blocks for iterative backcast/forecast refinement.
- Generic basis functions for flexible output decomposition.
- Supports external features and multi-task outputs.
- Grid search for architecture and training hyperparameters.

#### Code Implementation

- **NBeatsBlock Class**: Defines a block with configurable layers and basis function for backcast/forecast decomposition.
- **GenericBasis Class**: Implements the basis function for splitting parameters into backcast and forecast.
- **NBeats Class**: Stacks multiple blocks, refines residuals, and produces final forecast.
- **NBeats_exp Class**: Experiment wrapper for training, validation, and testing, including data loading, model instantiation, and evaluation.
- **Parameter Grid Setup**: Uses `sklearn.model_selection.ParameterGrid` to explore combinations of sliding window, batch size, learning rate, normalization, stacks, layers, layer size, and external flag.
- **Validation/Testing**: For each parameter set, trains N-BEATS on training data, validates on validation set, and selects the best configuration based on error metrics.
- **Hyperparameter Selection**: Chooses the best model using validation performance and saves parameters.
- **Testing**: Applies the selected model to the test set and saves predictions.
- **CLI Usage**: Accepts arguments for all key parameters; merges with YAML config for reproducibility.

**Typical Workflow:**
1. Load configuration and data.
2. Generate parameter grid.
3. For each file and parameter set:
   - Train and validate N-BEATS models.
   - Select best hyperparameters.
   - Test and save predictions.
4. Evaluate results and save metrics.

**Example Command:**
```bash
python models/NBeats.py --sliding_window 24 --batch_size 64 --learning_rate 0.001 --normalization "minmax" --stacks 3 --layers 4 --layer_size 256 --external_flag True --external_features "['temp','humidity']"
```

**References:**
- [N-BEATS: Neural Basis Expansion Analysis for Time Series Forecasting](https://arxiv.org/abs/1905.10437)
- [Official N-BEATS code](https://github.com/ElementAI/N-BEATS)

---

### Neural ODE Model (`NeuralODE.py`)

#### Theoretical Background

Neural Ordinary Differential Equations (Neural ODEs) model time series as solutions to ordinary differential equations parameterized by neural networks. This approach enables continuous-time modeling and flexible latent dynamics, capturing complex temporal dependencies beyond discrete-time architectures.

Features:
- Models latent dynamics with ODEs parameterized by neural networks.
- Recognition RNN encodes initial latent state from observed data.
- Decoder maps latent trajectories to observed outputs.
- Supports external features and multi-task outputs.
- Grid search for architecture and training hyperparameters.

#### Code Implementation

- **LatentODEfunc Class**: Defines the neural network parameterizing the ODE dynamics in latent space.
- **RecognitionRNN Class**: Encodes observed data into initial latent state.
- **Decoder Class**: Maps latent trajectories to observed outputs.
- **LatentODENet Class**: Integrates all components, performs ODE integration, and outputs predictions.
- **LatentODE_exp Class**: Experiment wrapper for training, validation, and testing, including data loading, model instantiation, and evaluation.
- **Parameter Grid Setup**: Uses `sklearn.model_selection.ParameterGrid` to explore combinations of sliding window, batch size, learning rate, normalization, latent dimension, hidden sizes, noise standard deviation.
- **Validation/Testing**: For each parameter set, trains NeuralODE on training data, validates on validation set, and selects the best configuration based on error metrics.
- **Hyperparameter Selection**: Chooses the best model using validation performance and saves parameters.
- **Testing**: Applies the selected model to the test set and saves predictions.
- **CLI Usage**: Accepts arguments for all key parameters; merges with YAML config for reproducibility.

**Typical Workflow:**
1. Load configuration and data.
2. Generate parameter grid.
3. For each file and parameter set:
   - Train and validate NeuralODE models.
   - Select best hyperparameters.
   - Test and save predictions.
4. Evaluate results and save metrics.

**Example Command:**
```bash
python models/NeuralODE.py --sliding_window 24 --batch_size 64 --learning_rate 0.001 --normalization "minmax" --latent_dim 4 --nhidden 20 --rnn_nhidden 25 --noise_std 0.1 --external_features "['temp','humidity']"
```

**References:**
- [Neural Ordinary Differential Equations](https://arxiv.org/abs/1806.07366)
- [Official torchdiffeq code](https://github.com/rtqichen/torchdiffeq)

---

### Random Forest Model (`random_forest.py`)

#### Theoretical Background

Random Forest is an ensemble learning method that builds multiple decision trees and averages their predictions for regression tasks. It supports multivariate output when forecasting for multiple locations and can incorporate external features. Random Forest is robust to overfitting and effective for nonlinear relationships.

Features:
- Ensemble of decision trees for robust regression.
- Supports multivariate output for multiple locations.
- Can incorporate external features.
- Grid search for number of estimators, criterion, and other hyperparameters.

#### Code Implementation

- **grid_search_random_foreast Function**: Sets up parameter grid, trains and validates models, selects best hyperparameters, and tests on unseen data.
- **Parameter Grid Setup**: Uses `sklearn.model_selection.ParameterGrid` to explore combinations of sliding window, variate, external feature flag, number of estimators, and criterion.
- **Validation/Testing**: For each parameter set, trains the random forest model on training data, validates on validation set, and selects the best configuration based on error metrics.
- **Hyperparameter Selection**: Chooses the best model using validation performance and saves parameters.
- **Testing**: Applies the selected model to the test set and saves predictions (mean, upper/lower intervals).
- **CLI Usage**: Accepts arguments for all key parameters; merges with YAML config for reproducibility.

**Typical Workflow:**
1. Load configuration and data.
2. Generate parameter grid.
3. For each file and parameter set:
   - Train and validate random forest models.
   - Select best hyperparameters.
   - Test and save predictions (mean, upper/lower intervals).
4. Evaluate results and save metrics.

**Example Command:**
```bash
python models/random_forest.py --sliding_window 24 --n_estimators 100 --criterion "mse" --external_feature_flag True --variate "multi"
```

**References:**
- [Random Forest documentation (scikit-learn)](https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.RandomForestRegressor.html)
- [Breiman, L. (2001). Random Forests](https://www.stat.berkeley.edu/~breiman/randomforest2001.pdf)

---

### RNN Model for Event Classification (`rnn.py`)

#### Theoretical Background

Recurrent Neural Networks (RNN), including GRU and LSTM variants, are widely used for sequence modeling and event classification in time series. These architectures capture temporal dependencies and can process sequences in unidirectional or bidirectional modes. Multi-layer RNNs and dropout improve model capacity and generalization.

Features:
- Supports RNN, GRU, and LSTM cells.
- Unidirectional and bidirectional processing.
- Multi-layer architecture with configurable hidden size and dropout.
- Embedding and fully connected layers for feature extraction and classification.
- Grid search for architecture and training hyperparameters.

#### Code Implementation

- **RNNNet Class**: Defines the RNN/GRU/LSTM architecture, embedding, and output layers for classification.
- **RNN_exp Class**: Experiment wrapper for training, validation, and testing, including data loading, model instantiation, and evaluation.
- **grid_search_RNN Function**: Sets up parameter grid, trains and validates models, selects best hyperparameters, and tests on unseen data.
- **Parameter Grid Setup**: Uses `sklearn.model_selection.ParameterGrid` to explore combinations of hidden size, number of layers, batch size, learning rate, dropout, direction, normalization, target name, and label constraints.
- **Validation/Testing**: For each parameter set, trains the RNN on training data, validates on validation set, and selects the best configuration based on error metrics.
- **Hyperparameter Selection**: Chooses the best model using validation performance and saves parameters.
- **Testing**: Applies the selected model to the test set and saves predictions.
- **CLI Usage**: Accepts arguments for all key parameters; merges with YAML config for reproducibility.

**Typical Workflow:**
1. Load configuration and data.
2. Generate parameter grid.
3. For each parameter set:
   - Train and validate RNN/GRU/LSTM models.
   - Select best hyperparameters.
   - Test and save predictions.
4. Evaluate results and save metrics.

**Example Command:**
```bash
python EventClassification/models/rnn.py --hidden_size 128 --num_layers 2 --batch_size 64 --learning_rate 0.001 --dropout 0.2 --direction "bi" --model_name "GRU" --target_name "fault"
```

**References:**
- [PyTorch RNN documentation](https://pytorch.org/docs/stable/generated/torch.nn.RNN.html)
- [PyTorch LSTM documentation](https://pytorch.org/docs/stable/generated/torch.nn.LSTM.html)
- [PyTorch GRU documentation](https://pytorch.org/docs/stable/generated/torch.nn.GRU.html)

---

### SVR Model (`svr.py`)

#### Theoretical Background

Support Vector Regression (SVR) is a kernel-based machine learning method for regression tasks. SVR finds a function that deviates from actual targets by a value no greater than epsilon for all training data, and is as flat as possible. SVR is typically used for univariate regression, but this implementation supports forecasting for multiple locations and can incorporate external features.

Features:
- Kernel-based regression (linear, polynomial, RBF, etc.).
- Typically univariate, but supports multiple locations.
- Can incorporate external features.
- Grid search for kernel type, sliding window, variate, and external feature flag.

#### Code Implementation

- **grid_search_svr Function**: Sets up parameter grid, trains and validates models, selects best hyperparameters, and tests on unseen data.
- **Parameter Grid Setup**: Uses `sklearn.model_selection.ParameterGrid` to explore combinations of sliding window, kernel, variate, and external feature flag.
- **Validation/Testing**: For each parameter set, trains the SVR model on training data, validates on validation set, and selects the best configuration based on error metrics.
- **Hyperparameter Selection**: Chooses the best model using validation performance and saves parameters.
- **Testing**: Applies the selected model to the test set and saves predictions (mean, upper/lower intervals).
- **CLI Usage**: Accepts arguments for all key parameters; merges with YAML config for reproducibility.

**Typical Workflow:**
1. Load configuration and data.
2. Generate parameter grid.
3. For each file and parameter set:
   - Train and validate SVR models.
   - Select best hyperparameters.
   - Test and save predictions (mean, upper/lower intervals).
4. Evaluate results and save metrics.

**Example Command:**
```bash
python models/svr.py --sliding_window 24 --kernel "rbf" --variate "uni" --external_feature_flag True
```

**References:**
- [Support Vector Regression documentation (scikit-learn)](https://scikit-learn.org/stable/modules/generated/sklearn.svm.SVR.html)
- [Smola, A.J., & Schölkopf, B. (2004). A tutorial on support vector regression](https://www.csie.ntu.edu.tw/~cjlin/papers/support_vector_regression.pdf)

---

### TCN Model (`tcn.py`)

#### Theoretical Background

Temporal Convolutional Network (TCN) is a deep learning architecture for sequence modeling and time series forecasting. TCN uses causal convolutions (no future leakage), dilation for long-range dependencies, and residual connections for stable training. It supports multi-layer, multi-channel architectures and flexible output mapping.

Features:
- Causal convolutions for temporal order preservation.
- Dilation for capturing long-term dependencies.
- Residual connections for improved gradient flow.
- Supports external features and multi-task outputs.
- Grid search for architecture and training hyperparameters.

#### Code Implementation

- **Chomp1d, TemporalBlock, TemporalConvNet Classes**: Define the core TCN architecture, including causal/dilated convolutions and residual connections.
- **TCNModel Class**: Wraps the TCN backbone and output decoder, supports various output strategies (first, last, all tokens).
- **TCN_exp Class**: Experiment wrapper for training, validation, and testing, including data loading, model instantiation, and evaluation.
- **grid_search_TCN Function**: Sets up parameter grid, trains and validates models, selects best hyperparameters, and tests on unseen data.
- **Parameter Grid Setup**: Uses `sklearn.model_selection.ParameterGrid` to explore combinations of sliding window, batch size, learning rate, normalization, dropout, hidden channels, levels, kernel size, and classification token.
- **Validation/Testing**: For each parameter set, trains the TCN on training data, validates on validation set, and selects the best configuration based on error metrics.
- **Hyperparameter Selection**: Chooses the best model using validation performance and saves parameters.
- **Testing**: Applies the selected model to the test set and saves predictions.
- **CLI Usage**: Accepts arguments for all key parameters; merges with YAML config for reproducibility.

**Typical Workflow:**
1. Load configuration and data.
2. Generate parameter grid.
3. For each file and parameter set:
   - Train and validate TCN models.
   - Select best hyperparameters.
   - Test and save predictions.
4. Evaluate results and save metrics.

**Example Command:**
```bash
python models/tcn.py --sliding_window 24 --batch_size 64 --learning_rate 0.001 --dropout 0.2 --nhid 64 --levels 3 --kernel_size 2 --classification_token "last" --normalization "minmax" --external_features "['temp','humidity']"
```

**References:**
- [TCN: An Empirical Evaluation of Generic Convolutional and Recurrent Networks for Sequence Modeling](https://arxiv.org/abs/1803.01271)
- [Official TCN code](https://github.com/locuslab/TCN)

---

### Transformer Model (`transformer.py`)

#### Theoretical Background

The vanilla Transformer is a deep learning architecture based on self-attention mechanisms, originally designed for sequence modeling in NLP but now widely used for time series forecasting. Transformers use multi-head self-attention to capture dependencies across all positions in the input sequence, enabling flexible modeling of temporal relationships. Positional encoding is used to inject order information.

Features:
- Multi-head self-attention for global dependency modeling.
- Stacked encoder layers for hierarchical feature extraction.
- Positional encoding for temporal order.
- Supports external features and multi-task outputs.
- Grid search for architecture and training hyperparameters.

#### Code Implementation

- **TransformerNet Class**: Defines the Transformer encoder architecture, embedding, and output layers for regression.
- **Transformer_exp Class**: Experiment wrapper for training, validation, and testing, including data loading, model instantiation, and evaluation.
- **grid_search_transformer Function**: Sets up parameter grid, trains and validates models, selects best hyperparameters, and tests on unseen data.
- **Parameter Grid Setup**: Uses `sklearn.model_selection.ParameterGrid` to explore combinations of sliding window, hidden size, number of layers, batch size, learning rate, dropout, number of heads, classification token, and normalization.
- **Validation/Testing**: For each parameter set, trains the Transformer on training data, validates on validation set, and selects the best configuration based on error metrics.
- **Hyperparameter Selection**: Chooses the best model using validation performance and saves parameters.
- **Testing**: Applies the selected model to the test set and saves predictions.
- **CLI Usage**: Accepts arguments for all key parameters; merges with YAML config for reproducibility.

**Typical Workflow:**
1. Load configuration and data.
2. Generate parameter grid.
3. For each file and parameter set:
   - Train and validate Transformer models.
   - Select best hyperparameters.
   - Test and save predictions.
4. Evaluate results and save metrics.

**Example Command:**
```bash
python models/transformer.py --sliding_window 24 --hidden_size 128 --num_layers 2 --batch_size 64 --learning_rate 0.001 --dropout 0.1 --num_heads 4 --classification_token "last" --normalization "minmax" --external_features "['temp','humidity']"
```

**References:**
- [Attention Is All You Need (Transformer)](https://arxiv.org/abs/1706.03762)
- [PyTorch Transformer documentation](https://pytorch.org/docs/stable/generated/torch.nn.TransformerEncoder.html)

---

### WaveNet Model (`WaveNet.py`)

#### Theoretical Background

WaveNet is a deep learning architecture originally designed for audio generation, now widely used for time series forecasting. It uses dilated causal convolutions to capture long-range dependencies without future leakage, and employs residual and skip connections for stable training and efficient gradient flow.

Features:
- Dilated causal convolutions for long-range temporal modeling.
- Residual and skip connections for improved training.
- Supports external features and multi-task outputs.
- Grid search for architecture and training hyperparameters.

#### Code Implementation

- **constant_pad_1d, dilate, DilatedQueue Functions/Classes**: Utilities for padding, dilation, and efficient queue management in convolutions.
- **WaveNetModel Class**: Defines the full WaveNet architecture, including dilated convolutions, residual/skip connections, and output layers.
- **WaveNetpy_exp Class**: Experiment wrapper for training, validation, and testing, including data loading, model instantiation, and evaluation.
- **grid_search_WaveNetpy Function**: Sets up parameter grid, trains and validates models, selects best hyperparameters, and tests on unseen data.
- **Parameter Grid Setup**: Uses `sklearn.model_selection.ParameterGrid` to explore combinations of sliding window, batch size, learning rate, normalization, layers, blocks, dilation/residual/skip/end channels, kernel size.
- **Validation/Testing**: For each parameter set, trains the WaveNet on training data, validates on validation set, and selects the best configuration based on error metrics.
- **Hyperparameter Selection**: Chooses the best model using validation performance and saves parameters.
- **Testing**: Applies the selected model to the test set and saves predictions.
- **CLI Usage**: Accepts arguments for all key parameters; merges with YAML config for reproducibility.

**Typical Workflow:**
1. Load configuration and data.
2. Generate parameter grid.
3. For each file and parameter set:
   - Train and validate WaveNet models.
   - Select best hyperparameters.
   - Test and save predictions.
4. Evaluate results and save metrics.

**Example Command:**
```bash
python models/WaveNet.py --sliding_window 24 --batch_size 64 --learning_rate 0.001 --layers 10 --blocks 4 --dilation_channels 32 --residual_channels 32 --skip_channels 256 --end_channels 256 --kernel_size 2 --normalization "minmax" --external_features "['temp','humidity']"
```

**References:**
- [WaveNet: A Generative Model for Raw Audio](https://arxiv.org/abs/1609.03499)
- [Official PyTorch WaveNet code](https://github.com/vincentherrmann/pytorch-wavenet)

- **Machine Learning Models:**  
  - *SVR, Random Forest, Gradient Boosting, Linear Regression*: Supervised regression algorithms using historical windows and optional external features.

- **Deep Learning Models:**  
  - *FNN, CNN, RNN, LSTM, GRU*: Neural architectures for sequence modeling. CNNs capture local patterns; RNNs/LSTMs/GRUs model temporal dependencies.
  - *LSTNet*: Combines CNN, RNN, skip connections, and highway networks for multivariate time series.
  - *DeepAR*: Probabilistic forecasting with LSTM, outputs mean and variance.
  - *WaveNet*: Dilated convolutions for long-range dependencies.
  - *NBeats*: Stacked blocks for interpretable backcast/forecast decomposition.
  - *NeuralODE*: Models time series as solutions to ODEs in latent space.
  - *TCN*: Causal convolutions with residual connections.
  - *Transformer/Informer*: Attention-based models for long sequence modeling.

## Code Structure

- Each model script defines:
  - Model class (architecture, forward pass, loss function)
  - Experiment class (training/testing logic)
  - Grid search or training function (`grid_search_*`)
  - Main entry point for argument parsing and configuration loading

- Shared utilities (`utils.py`):
  - Data loading, preprocessing
  - Evaluation metrics (RMSE, MAE, MAPE, MSIS)
  - Training/testing routines
  - Hyperparameter merging

- Configuration:
  - Each model uses a YAML config file for hyperparameters and experiment settings.

## Usage

1. **Install dependencies:**  
   See `requirements.txt` for required packages (PyTorch, scikit-learn, pandas, etc.).

2. **Prepare data:**  
   Place CSV files in the specified data folder. Each file should include historical values, external features, and flags.

3. **Run a model:**  
   ```bash
   python models/arima.py --sliding_window 24 --p_values [1,2,3] --d_values [0,1] --q_values [0,1] --variate ['uni','multi']
   ```
   Replace with the appropriate script and arguments for each model.

4. **Evaluate results:**  
   Outputs are saved in the `logs/` directory. Use provided evaluation functions to compute metrics.

## References

- ARIMA: [Statsmodels documentation](https://www.statsmodels.org/)
- DeepAR: [Salinas et al., 2020](https://arxiv.org/abs/1704.04110)
- LSTNet: [Lai et al., 2018](https://arxiv.org/abs/1703.07015)
- N-BEATS: [Oreshkin et al., 2020](https://arxiv.org/abs/1905.10437)
- WaveNet: [Oord et al., 2016](https://arxiv.org/abs/1609.03499)
- Informer: [Zhou et al., 2021](https://arxiv.org/abs/2012.07436)
- NeuralODE: [Chen et al., 2018](https://arxiv.org/abs/1806.07366)

## Notes

- All models support grid search for hyperparameter tuning.
- External features and multi-variate forecasting are supported where applicable.
- For details on each model, see the docstring and comments in the corresponding `.py` file.
