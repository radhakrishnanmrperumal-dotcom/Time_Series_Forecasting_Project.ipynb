# Time Series Forecasting Project 

## Executive Summary

This Jupyter notebook presents a **complete end-to-end time series forecasting project** that predicts residential property sale (RPS) prices using three different modeling approaches: LSTM neural networks, Transformer models, and traditional SARIMAX. The project demonstrates sophisticated machine learning techniques, proper data preprocessing, and comprehensive model evaluation.

##  Project Overview

### Objective
Predict residential property sale prices using historical data and various economic indicators.

### Dataset Characteristics
- **Size**: 2,000 daily observations (~5.5 years of data)
- **Target Variable**: Property sale price
- **Exogenous Features**: 6 economic indicators
  1. Interest rates (2.0% - 8.0%)
  2. Employment rate
  3. Consumer confidence index
  4. Building permits
  5. Inventory levels
  6. GDP growth

### Data Generation Approach
The project uses **synthetic data** with realistic components:
- **Trend**: Quadratic growth (200,000 + 500t + 0.1t²)
- **Seasonality**: Multiple seasonal patterns
  - Annual cycle (365 days)
  - Quarterly cycle (90 days)
- **Long-term cycles**: 6-year economic cycles
- **Noise**: Controlled random variation

---

## 🔧 Technical Implementation

### 1. Data Preprocessing Pipeline

#### Feature Engineering
The project creates sophisticated engineered features:

**Lag Features**:
- 1-day lag
- 7-day lag  
- 30-day lag

**Rolling Statistics**:
- 7-day rolling mean
- 7-day rolling standard deviation

**Time-based Features**:
- Day of week
- Month
- Cyclical encoding (sine/cosine transformations)
  - `day_sin = sin(2π × day_of_week / 7)`
  - `day_cos = cos(2π × day_of_week / 7)`

#### Normalization
- **Method**: StandardScaler (z-score normalization)
- **Application**: All features scaled independently
- **Preservation**: Scalers saved for inverse transformation

#### Data Split Strategy
- **Training**: 70% (first portion)
- **Validation**: 15% (middle portion)
- **Testing**: 15% (final portion)
- **Rationale**: Time-ordered split preserves temporal structure

---

### 2. Model Architectures

#### A. LSTM Model (Long Short-Term Memory)

**Architecture**:
```
Input Layer → LSTM(128 hidden, 2 layers, dropout=0.2)
           → Dense(64) + ReLU + Dropout
           → Dense(32) + ReLU + Dropout
           → Dense(1) [Output]
```

**Key Features**:
- Bidirectional capability for capturing past and future context
- Gradient clipping (max norm = 1.0) to prevent exploding gradients
- Dropout regularization to prevent overfitting
- Sequence length: 30 time steps

**Advantages**:
- Excellent at capturing long-term dependencies
- Memory cells preserve important historical information
- Proven effectiveness for time series

---

#### B. Transformer Model

**Architecture**:
```
Input → Linear Projection (to d_model=128)
     → Positional Encoding
     → Transformer Encoder (3 layers, 8 attention heads)
     → Dense(64) + ReLU + Dropout
     → Dense(1) [Output]
```

**Key Components**:
1. **Positional Encoding**: Sine/cosine functions to inject temporal information
2. **Multi-head Attention**: 8 parallel attention mechanisms
3. **Feed-forward Networks**: 4× expansion in hidden dimension
4. **Layer Normalization**: Stabilizes training

**Advantages**:
 Parallel processing (faster than LSTM)
 Self-attention captures complex relationships
 No vanishing gradient problems



#### C. SARIMAX Baseline (Seasonal ARIMA with eXogenous variables)

**Configuration**:
**ARIMA Order**: (2, 1, 2)
   AR: 2 autoregressive terms
   I: 1 differencing
   MA: 2 moving average terms
 **Seasonal Order**: (1, 1, 1, 7)
   Weekly seasonality
  
**Purpose**: Traditional statistical baseline for comparison


### 3. Training Strategy

#### Hyperparameters
 **LSTM**:
   Learning rate: 0.001
   Batch size: 32
   Max epochs: 50
   Optimizer: Adam

 **Transformer**:
   Learning rate: 0.0005 (lower for stability)
   Batch size: 32
   Max epochs: 50
   Optimizer: Adam

#### Regularization Techniques
1. **Early Stopping**:
    Patience: 10 epochs
    Monitors validation loss
    Restores best weights

2. **Gradient Clipping**:
    Max norm: 1.0
    Prevents exploding gradients

3. **Dropout**:
    Rate: 0.2 (LSTM)
    Rate: 0.1 (Transformer)

### 4. Evaluation Metrics

The project uses three complementary metrics:

1. **RMSE (Root Mean Squared Error)**:
    Penalizes large errors heavily
    In dollar units (interpretable)
    Primary metric for comparison

2. **MAE (Mean Absolute Error)**:
    Average absolute deviation
    Less sensitive to outliers
    Linear penalty for errors

3. **R² (Coefficient of Determination)**:
    Proportion of variance explained
    Scale: 0 to 1 (higher is better)
    Measures goodness of fit


##  Exploratory Data Analysis (EDA)

### Visualizations Created

1. **Time Series Plot**: Price evolution over time
2. **Distribution Analysis**: Histogram of price values
3. **Correlation Heatmap**: Relationships between features
4. **Moving Average Overlay**: 30-day smoothing for trend visualization

### Statistical Tests

**Augmented Dickey-Fuller (ADF) Test**:
 Tests for stationarity
 Null hypothesis: Series has unit root (non-stationary)
 Critical for ARIMA modeling



##  Model Interpretability

### Integrated Gradients Analysis

The project implements **Captum's Integrated Gradients** for the LSTM model:

**Purpose**: Understand which features contribute most to predictions

**Method**:
1. Compute gradients along path from baseline to input
2. Aggregate attributions across samples and time steps
3. Normalize to create feature importance scores

**Output**: 
 Top 5 most influential features identified
 Visualization of relative importance
 Insights into model decision-making

**Benefits**:
 Model transparency
 Feature validation
 Debugging assistance
 Business insights


##  Project Structure & Code Quality

### Object-Oriented Design

**Preprocessor Class**:
 Encapsulates all preprocessing logic
 Maintains state (scalers)
 Reusable for train/test data
 Supports inverse transformation

**Model Classes**:
 Clean inheritance from `nn.Module`
 Configurable architectures
 Well-documented forward passes

### Dataset Management

**TimeSeriesDataset Class**:
 Implements PyTorch Dataset interface
 Handles sequence creation
 Efficient batching
 Type conversion (float32)

### Best Practices Observed

 **Reproducibility**: Fixed random seed (SEED=42)  
 **Device Agnostic**: Automatic CPU/GPU detection  
 **Error Handling**: Warning suppression for cleaner output  
 **Modularity**: Separate functions for training/evaluation  
 **Documentation**: Clear variable names and comments  


##  Key Insights & Findings

### 1. Model Comparison Framework
The project enables **direct comparison** between:
- Deep learning approaches (LSTM, Transformer)
- Traditional statistical methods (SARIMAX)

### 2. Feature Importance
- Quantifies which economic indicators matter most
- Validates feature engineering choices
- Guides future data collection

### 3. Temporal Patterns
- Successfully captures trend, seasonality, and cycles
- Demonstrates importance of sequence modeling
- Shows value of exogenous variables

### 4. Performance Metrics
The final comparison DataFrame provides:
- Side-by-side RMSE, MAE, R² values
- Percentage improvement over baseline
- Clear winner identification

---

##  Advanced Techniques Demonstrated

1. **Sequence-to-One Prediction**:
   - 30-step input → 1-step output
   - Appropriate for point forecasting

2. **Attention Mechanisms**:
   - Multi-head self-attention in Transformer
   - Learns feature relationships automatically

3. **Positional Encoding**:
   - Injects temporal order into Transformer
   - Critical for sequence understanding

4. **Gradient-based Interpretation**:
   - Goes beyond black-box predictions
   - Provides actionable insights

5. **Loss Curve Tracking**:
   - Monitors training/validation losses
   - Enables diagnostic analysis

---

##  Dependencies & Environment

### Core Libraries
```python
torch              # Deep learning framework
captum             # Model interpretability
statsmodels        # Statistical models (SARIMAX)
pmdarima           # Auto-ARIMA
scikit-learn       # Preprocessing, metrics
numpy              # Numerical computing
pandas             # Data manipulation
matplotlib/seaborn # Visualization
```

### Compatibility Note
The notebook encountered a version conflict:
- Captum requires `numpy<2.0`
- Some dependencies upgraded to `numpy 2.4.1`
- Resolution: Downgrade numpy or use compatible versions



##  Educational Value

This project is excellent for learning:

1. **Time Series Fundamentals**:
   - Stationarity testing
   - Seasonal decomposition
   - Autocorrelation

2. **Deep Learning for Sequences**:
   - LSTM architecture and training
   - Transformer from scratch
   - Sequence dataset handling

3. **Model Comparison**:
   - Multiple paradigms (DL vs. statistical)
   - Fair evaluation methodology
   - Metric interpretation

4. **Production Considerations**:
   - Data preprocessing pipelines
   - Model checkpointing (early stopping)
   - Inverse transformations for reporting

5. **Explainable AI**:
   - Attribution methods
   - Feature importance
   - Model transparency

##  Potential Extensions

### Short-term Improvements
1. **Hyperparameter Tuning**: Grid search or Bayesian optimization
2. **Ensemble Methods**: Combine LSTM + Transformer predictions
3. **Multi-step Forecasting**: Predict 7-day or 30-day ahead
4. **Cross-validation**: Time series cross-validation (walk-forward)

### Advanced Enhancements
1. **Attention Visualization**: Plot attention weights over time
2. **Uncertainty Quantification**: Prediction intervals (e.g., quantile regression)
3. **External Data**: Real economic indicators (FRED API)
4. **Real-time Forecasting**: Streaming data pipeline
5. **Model Deployment**: REST API with Flask/FastAPI

### Research Directions
1. **Hybrid Models**: Combine statistical + DL approaches
2. **Graph Neural Networks**: Model spatial-temporal dependencies
3. **Few-shot Learning**: Adapt to new markets quickly
4. **Causal Inference**: Identify true drivers vs. correlations


##  Limitations & Considerations

1. **Synthetic Data**:
    Real-world data has more irregularities
    May not capture black swan events
    Assumes stable relationships

2. **Feature Selection**:
    Limited to 6 exogenous variables
    Real estate has many more factors
    No categorical features (location, type)

3. **Evaluation Period**:
    Single test set
    No walk-forward validation
    Assumes stationarity in test period

4. **Computational Cost**:
    Transformers require more memory
    Training time not optimized
    No distributed training

5. **Interpretability Trade-offs**:
    Integrated Gradients is post-hoc
    Attention ≠ explanation
    Feature importance context-dependent

##  Conclusion

This notebook represents a **well-structured, comprehensive time series forecasting project** that:

 Follows machine learning best practices  
 Implements cutting-edge architectures (LSTM, Transformer)  
 Includes traditional baselines for comparison  
 Emphasizes interpretability and evaluation  
 Demonstrates professional code organization  

**Strengths**:
 Clear progression from data → models → evaluation
 Multiple modeling paradigms compared fairly
 Thoughtful feature engineering
 Interpretability analysis included
 Reproducible and well-documented

**Learning Outcomes**:
Perfect for understanding:
 End-to-end ML project workflow
 Time series forecasting techniques
 Deep learning for sequential data
 Model evaluation and comparison
 Production-ready code structure

**Recommended For**:
  Data science students
  ML practitioners entering time series
  Researchers comparing forecasting methods
  Anyone building production forecasting systems

# References & Further Reading

**Time Series Forecasting**:
 Box, G.E.P., & Jenkins, G.M. (1970). Time Series Analysis
 Hyndman, R.J., & Athanasopoulos, G. (2018). Forecasting: Principles and Practice

**Deep Learning for Sequences**:
  Hochreiter, S., & Schmidhuber, J. (1997). Long Short-Term Memory
  Vaswani, A., et al. (2017). Attention Is All You Need
  
**Practical Guides**:
 PyTorch Documentation: https://pytorch.org/docs/
 Captum Tutorials: https://captum.ai/tutorials/
 Statsmodels SARIMAX: https://www.statsmodels.org/

