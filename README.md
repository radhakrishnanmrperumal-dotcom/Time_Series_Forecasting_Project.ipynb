# Time Series Forecasting Project 

## Executive Summary

This Jupyter notebook implements a complete **Real Property Sales (RPS) time series forecasting pipeline** that compares three different modeling approaches: LSTM (deep learning), Transformer (attention-based), and SARIMAX (traditional statistical). The project demonstrates advanced forecasting techniques with synthetic but realistic housing price data spanning approximately 5.5 years.

---

## Project Overview

### Objective
Predict real estate property prices using multiple time series modeling techniques and compare their performance on a synthetic dataset that mimics real-world housing market dynamics.

### Key Highlights
- **Dataset**: 2,000 daily observations (~5.5 years) of property prices
- **Models**: LSTM, Transformer, and SARIMAX
- **Features**: 6 exogenous variables + engineered time-based features
- **Approach**: Comprehensive pipeline including data generation, preprocessing, modeling, evaluation, and interpretability analysis

---

## Detailed Component Analysis

### 1. **Environment Setup & Dependencies** (Cells 1-2)

The project uses modern machine learning and time series libraries:

**Core Libraries:**
- `PyTorch` - Deep learning framework for LSTM and Transformer models
- `scikit-learn` - Data preprocessing and metrics
- `statsmodels` - Traditional time series modeling (SARIMAX)
- `captum` - Model interpretability (Integrated Gradients)
- `pandas/numpy` - Data manipulation
- `matplotlib/seaborn` - Visualization

**Note**: The notebook encounters some dependency conflicts (particularly with numpy version 2.4.1 vs older packages), which is common in complex ML environments but appears to be resolved in later cells.

---

### 2. **Synthetic Data Generation** (Cell 4)

This is a sophisticated data generation process that creates realistic property price time series.

**Data Components:**

**a) Trend Component**
```
trend = 200000 + 500 * t + 0.1 * t²
```
- Base price: $200,000
- Linear growth: $500 per day
- Quadratic acceleration: accounts for compounding market growth

**b) Seasonality**
```
15000 * sin(2π * t / 365) + 8000 * sin(2π * t / 90)
```
- Annual cycle: ±$15,000 (yearly housing market patterns)
- Quarterly cycle: ±$8,000 (seasonal demand variations)

**c) Economic Cycle**
```
25000 * sin(2π * t / (365 * 6))
```
- 6-year business cycle: ±$25,000 (long-term economic cycles)

**d) Exogenous Features** (6 variables):

1. **Interest Rates** (2.0% - 8.0%)
   - Cyclical pattern with random noise
   - Strong inverse relationship with prices

2. **Employment Rate** (75% - 95%)
   - Long-term wave pattern
   - Positive correlation with housing demand

3. **Housing Supply**
   - Growing supply over time
   - Seasonal variations

4. **GDP Growth** (-1% to +6%)
   - Economic health indicator
   - Direct impact on purchasing power

5. **Consumer Confidence** (70-130 index)
   - Annual cyclical pattern
   - Reflects buyer sentiment

**e) Autoregressive Component**
```
AR(1) process with coefficient 0.7
```
- Captures price momentum and market memory
- Standard deviation: $5,000

**Feature Impact Model:**
The final price integrates all these factors with realistic economic relationships:
- -$3,000 per 1% increase in interest rates
- +$800 per 10% increase in employment
- -$0.50 per unit increase in housing supply
- +$2,000 per 1% GDP growth
- +$150 per 1-point confidence increase

**Output:**
- 2,000 daily observations
- Price range: $100,000 minimum (floor constraint)
- Realistic market dynamics with multiple cyclical patterns

---

### 3. **Exploratory Data Analysis (EDA)** (Cell 5)

**Visualizations Created:**

1. **Price Over Time Plot**
   - Shows trend, seasonality, and cycles
   - Reveals overall growth pattern

2. **Price Distribution Histogram**
   - Examines price distribution shape
   - Checks for outliers

3. **Correlation Heatmap**
   - Analyzes relationships between all variables
   - Identifies multicollinearity issues

4. **30-Day Moving Average**
   - Smooths out noise
   - Highlights underlying trends

**Stationarity Test:**
- Uses Augmented Dickey-Fuller (ADF) test
- Determines if differencing is needed
- Critical for model selection (especially SARIMAX)

---

### 4. **Data Preprocessing Pipeline** (Cell 6)

The `Preprocessor` class implements sophisticated feature engineering:

**Lag Features:**
```python
price_lag_1  # Yesterday's price
price_lag_7  # Last week's price
price_lag_30 # Last month's price
```
- Captures short-term, weekly, and monthly patterns
- Essential for autoregressive behavior

**Rolling Statistics:**
```python
rolling_mean_7  # 7-day average
rolling_std_7   # 7-day volatility
```
- Momentum indicators
- Volatility measures

**Time-Based Features:**
```python
day_of_week    # 0-6 (Monday-Sunday)
month          # 1-12
day_sin        # Cyclical encoding (sine)
day_cos        # Cyclical encoding (cosine)
```
- Captures weekly and monthly patterns
- Sin/cos encoding preserves cyclical nature

**Normalization:**
- StandardScaler applied to all features
- Each feature scaled independently
- Maintains scalers for inverse transformation

**Train/Val/Test Split:**
- Training: 70% (1,400 samples)
- Validation: 15% (300 samples)
- Testing: 15% (300 samples)

---

### 5. **PyTorch Dataset Creation** (Cell 7)

**TimeSeriesDataset Class:**

```python
SEQ_LEN = 30  # 30-day lookback window
BATCH_SIZE = 32
```

**Architecture:**
- Creates sliding windows of 30 days
- Each window predicts the next day's price
- Separates features from target (price)
- Converts to PyTorch tensors

**Data Loaders:**
- Train: Shuffled batches for learning
- Val/Test: Sequential for evaluation
- Efficient batch processing

**Feature Count:**
- Original: 6 exogenous variables
- Engineered: ~10 additional features
- Total: ~16 input features per timestep

---

### 6. **LSTM Model Architecture** (Cell 8)

**Model Structure:**

```
Input (batch, 30, 16) → LSTM Layers → Fully Connected → Output (batch, 1)
```

**LSTM Component:**
- Hidden size: 128 units
- Layers: 2 stacked LSTM layers
- Dropout: 20% (between layers)
- Bidirectional: No (unidirectional for causal prediction)

**Fully Connected Head:**
```
Linear(128 → 64) → ReLU → Dropout(0.2)
    ↓
Linear(64 → 32) → ReLU → Dropout(0.2)
    ↓
Linear(32 → 1)  # Final prediction
```

**Key Characteristics:**
- Captures long-term dependencies (30-day sequences)
- Regularization through dropout
- Deep architecture with multiple non-linearities
- Parameter count: ~100K-200K parameters

**Why LSTM?**
- Handles variable-length sequences
- Maintains memory of past patterns
- Mitigates vanishing gradient problem
- Standard for time series tasks

---

### 7. **Training Procedure** (Cell 9)

**Training Configuration:**

```python
epochs = 50
learning_rate = 0.001
criterion = MSELoss  # Mean Squared Error
optimizer = Adam     # Adaptive learning rate
```

**Training Loop:**

1. **Forward Pass:**
   - Batch through model
   - Compute predictions

2. **Loss Calculation:**
   - Compare predictions to actual prices
   - MSE penalizes large errors quadratically

3. **Backward Pass:**
   - Compute gradients
   - Update weights via Adam optimizer

4. **Validation:**
   - Evaluate on validation set
   - No gradient updates

**Early Stopping:**
- Patience: 10 epochs
- Monitors validation loss
- Prevents overfitting
- Saves best model state

**Loss Tracking:**
- Records train loss each epoch
- Records validation loss
- Enables learning curve visualization

---

### 8. **Transformer Model** (Cell 10)

**Architecture Components:**

**Positional Encoding:**
```python
PE(pos, 2i) = sin(pos / 10000^(2i/d_model))
PE(pos, 2i+1) = cos(pos / 10000^(2i/d_model))
```
- Injects time information into embeddings
- Enables attention mechanism to understand sequence order
- Learnable or fixed (typically fixed)

**Transformer Encoder:**
```
Input Embedding → Positional Encoding → 
    Multi-Head Self-Attention (×3 layers) → 
    Feed-Forward Network → 
    Output Projection
```

**Key Parameters:**
- d_model: 64 (embedding dimension)
- nhead: 8 (parallel attention heads)
- dim_feedforward: 256
- num_layers: 3
- dropout: 0.2

**Advantages over LSTM:**
- Parallel processing (faster training)
- Captures long-range dependencies better
- Attention mechanism provides interpretability
- No sequential bottleneck

**Training:**
- Same procedure as LSTM
- Same hyperparameters
- Typically converges faster

---

### 9. **Model Evaluation** (Cell 11)

**Evaluation Metrics:**

1. **RMSE (Root Mean Squared Error)**
   ```
   RMSE = √(Σ(predicted - actual)² / n)
   ```
   - In original price scale ($)
   - Penalizes large errors more
   - Most interpretable metric

2. **MAE (Mean Absolute Error)**
   ```
   MAE = Σ|predicted - actual| / n
   ```
   - Average absolute deviation
   - More robust to outliers

3. **R² Score (Coefficient of Determination)**
   ```
   R² = 1 - (SS_res / SS_tot)
   ```
   - Proportion of variance explained
   - Range: (-∞, 1], perfect fit = 1

**Evaluation Process:**
1. Set model to evaluation mode (disable dropout)
2. Run inference on test set (no gradient computation)
3. Inverse transform predictions to original scale
4. Compute all metrics
5. Store results for comparison

**Visualization:**
- Actual vs Predicted plots
- Residual analysis
- Error distribution

---

### 10. **SARIMAX Baseline Model** (Cell 12)

**Model Specification:**

```python
SARIMAX(p=2, d=1, q=2, seasonal=(1,1,1,7))
```

**Parameters Explained:**

- **p=2**: 2 autoregressive terms (AR)
  - Uses past 2 values to predict
  
- **d=1**: 1 order of differencing
  - Makes series stationary
  
- **q=2**: 2 moving average terms (MA)
  - Uses past 2 error terms
  
- **Seasonal (1,1,1,7)**:
  - P=1: Seasonal AR term
  - D=1: Seasonal differencing
  - Q=1: Seasonal MA term
  - Period=7: Weekly seasonality

**Why SARIMAX?**
- Industry-standard baseline
- Interpretable coefficients
- Handles seasonality explicitly
- Well-suited for economic data

**Training:**
- Maximum iterations: 200
- Uses training data only
- Forecasts test period steps

**Comparison Value:**
- Establishes baseline performance
- Deep learning should outperform
- Validates data quality

---

### 11. **Comprehensive Model Comparison** (Cell 13)

**Results Table Format:**

```
Model        | RMSE    | MAE     | R²
-------------|---------|---------|-------
LSTM         | $X,XXX  | $Y,YYY  | 0.9XX
Transformer  | $X,XXX  | $Y,YYY  | 0.9XX
SARIMAX      | $X,XXX  | $Y,YYY  | 0.9XX
```

**Visualization Dashboard:**

1. **Learning Curves** (2 subplots)
   - LSTM: Train vs Val loss over epochs
   - Transformer: Train vs Val loss over epochs
   - Shows convergence and overfitting

2. **Predictions vs Actuals** (2 subplots)
   - LSTM: Overlaid actual and predicted prices
   - Transformer: Overlaid actual and predicted prices
   - Visual fit assessment

**Expected Insights:**

- **LSTM**: Good for sequential patterns
- **Transformer**: Better for complex dependencies
- **SARIMAX**: Competitive but less flexible

**Statistical Tests:**
- Can perform Diebold-Mariano test
- Compare forecast accuracy
- Statistical significance of differences

---

### 12. **Model Interpretability Analysis** (Cell 14)

**Integrated Gradients:**

**Concept:**
- Attribute prediction to input features
- Measures feature importance
- Specific to each prediction

**Process:**
```python
1. Define baseline (typically zeros)
2. Interpolate between baseline and input
3. Compute gradients at each step
4. Integrate gradients (sum)
```

**Output:**
- Attribution score per feature
- Positive: Feature increases prediction
- Negative: Feature decreases prediction
- Magnitude: Importance level

**Analysis:**
- Top 5 most important features
- Feature importance across time steps
- Temporal patterns in importance

**Visualization:**
- Heatmap of feature attributions
- Bar chart of average importance
- Time series of attribution patterns

**Business Value:**
- Explains model decisions
- Validates economic intuitions
- Builds stakeholder trust
- Identifies data quality issues

---

### 13. **Final Summary Report** (Cell 15)

**Report Structure:**

**1. Dataset Summary:**
- 2,000 daily samples (~5.5 years)
- 6 exogenous features + engineered features
- Train/Val/Test split details

**2. Models Trained:**
- LSTM: 2 layers, 128 hidden units, dropout
- Transformer: 3 layers, 8 attention heads
- SARIMAX: (2,1,2)(1,1,1,7) specification

**3. Performance Metrics:**
- Complete comparison table
- Winner identification
- Performance margins

**4. Key Findings:**
- Best performing model
- Feature importance insights
- Temporal patterns discovered
- Forecasting horizon capabilities

**5. Recommendations:**
- Model deployment suggestions
- Data collection improvements
- Feature engineering opportunities
- Future research directions

---

## Technical Architecture Summary

### Data Flow Pipeline

```
Raw Data Generation
    ↓
Feature Engineering
    ↓
Normalization
    ↓
Sequence Creation (30-day windows)
    ↓
Model Training (LSTM, Transformer, SARIMAX)
    ↓
Evaluation & Comparison
    ↓
Interpretability Analysis
    ↓
Results & Recommendations
```

### Model Complexity Comparison

| Aspect | LSTM | Transformer | SARIMAX |
|--------|------|-------------|---------|
| Parameters | ~150K | ~200K | ~20 |
| Training Time | Moderate | Fast (parallel) | Very Fast |
| Interpretability | Low | Medium (attention) | High |
| Flexibility | High | Very High | Low |
| Data Requirements | High | High | Moderate |

---

## Strengths of This Project

1. **Comprehensive Pipeline**: End-to-end implementation
2. **Multiple Approaches**: Compares traditional vs modern methods
3. **Realistic Data**: Sophisticated synthetic data generation
4. **Feature Engineering**: Extensive temporal features
5. **Proper Validation**: Train/val/test split with early stopping
6. **Interpretability**: Integrated Gradients analysis
7. **Visualization**: Clear, informative plots
8. **Documentation**: Well-structured code

---

## Potential Improvements

1. **Hyperparameter Tuning**:
   - Grid search or Bayesian optimization
   - Learning rate scheduling
   - Architecture search

2. **Ensemble Methods**:
   - Combine LSTM and Transformer predictions
   - Weighted averaging
   - Stacking

3. **Additional Models**:
   - Prophet (Facebook's forecasting tool)
   - XGBoost with lag features
   - Neural Prophet
   - Temporal Fusion Transformer

4. **Feature Engineering**:
   - Technical indicators (RSI, MACD)
   - Fourier features for seasonality
   - External data sources

5. **Evaluation**:
   - Multiple horizon forecasting (1-day, 7-day, 30-day)
   - Prediction intervals (uncertainty quantification)
   - Directional accuracy

6. **Production Considerations**:
   - Model serving infrastructure
   - Monitoring and retraining
   - A/B testing framework

---

## Real-World Applications

This methodology is applicable to:

1. **Real Estate**: Property price forecasting
2. **Finance**: Stock price prediction, portfolio optimization
3. **Retail**: Demand forecasting, inventory management
4. **Energy**: Load forecasting, renewable generation prediction
5. **Healthcare**: Patient admission forecasting
6. **Transportation**: Traffic flow prediction

---

## Key Takeaways

1. **Deep learning excels** when sufficient data and complex patterns exist
2. **Traditional methods** remain competitive for interpretability and simple patterns
3. **Feature engineering** is critical regardless of model choice
4. **Proper evaluation** requires multiple metrics and visualization
5. **Interpretability** is increasingly important for production systems
6. **Time series forecasting** requires domain knowledge and statistical rigor

---

## Conclusion

This notebook demonstrates a professional-grade time series forecasting pipeline that balances statistical rigor, modern deep learning techniques, and practical considerations. The comparison of LSTM, Transformer, and SARIMAX models provides valuable insights into the strengths and trade-offs of each approach. The inclusion of interpretability analysis via Integrated Gradients adds significant value for real-world deployment where model transparency is crucial.

The synthetic data generation process is particularly noteworthy, as it creates realistic economic relationships that mirror actual housing market dynamics. This makes the project both educational and practically relevant for anyone working in forecasting or quantitative analysis.

Overall, this project serves as an excellent template for time series forecasting tasks and demonstrates best practices in model development, evaluation, and interpretation.
