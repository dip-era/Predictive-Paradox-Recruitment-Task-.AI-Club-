# Power Grid Demand Forecasting (Summary report):

**Author's Note:** The Jupyter Notebook provided in this repository represents the finalized, production-ready pipeline. Please note that extensive behind-the-scenes exploratory data analysis (EDA), iterative hyperparameter testing, and feature experimentation were conducted in separate environments. I deliberately omitted those scratchpad iterations from the final `.ipynb` file to ensure the submitted deliverable remains clean, linear, and strictly focused on the final mathematical architecture.

This repository contains a production-ready machine learning pipeline designed to predict hourly power grid demand. The architecture prioritizes causal purity and strictly avoids temporal data lekage.

## 1. Rationale for Handling Missing Data and Outliers

Backward-Looking Anomaly Detection: Applying a standard 1.5 * IQR multiplier would risk deleting valid peak-demand events (like heatwaves), effectively blinding the model to extreme scenarios. Instead, a highly conservative 2.5 * IQR threshold was applied using a strictly backward-looking 7-day rolling window (center=False). This ensures only impossible physical limits or actual hardware sensor failures are flagged as NaN while preserving natural peaks.

Intraday Imputation (Forward fill): For hourly gaps caused by sensor downtime or removed anomalies, missing values were strictly imputed using .ffill() (forward-fill). Interpolating with future data to guess today's missing value introduces leakage, making forward-fill the most mathematically suitable approach.

Macroeconomic Zero-Leakage Dropping: When merging annual macroeconomic data (GDP, Population, Industry) with the hourly grid, some early years lacked corresponding economic records. Rather than using a backward-fill (bfill) which would leak future economic trends into past power data, I actively chose to enforce a zero-leakage policy by using .dropna(). Sacrificing early historical volume ensures a 100% pure causal timeline.

## 2. Temporal Feature Engineering Strategy

The timeline was mathematically translated into supervised learning columns to allow the models to recognize patterns as models (here Random Forests regressor) do not inherently understand the sequential flow of time.

**The Target variable (shift(-1)):** The core supervised setup involved shifting the target demand backward by one hour. This aligns "current weather and time" with "next hour's demand" on the exact same row, teaching the model to predict the future based on present conditions.

**Human BehaVior & Seasonality (Time Flags):** Columns for the hour of the day, month, and weekend status (is_weekend) were extracted. These flags teach the model about human-driven seasonality, such as offices closing on weekends or AC usage spiking during summer months.

**Autoregressive Grid memory (Lags):** To give the model historical context, lag features were engineered to show the grid state 1 hour ago (lag_1h), 24 hours ago (lag_24h), and exactly 1 week ago (lag_168h).

**Grid Momentum (Rolling trends):** Moving averages (3-hour and 24-hour rolling means) were added to help the model quantify momentum—specifically, whether the current grid load is actively ramping up or stabilizing.

## 3. Modeling Choice & Feature Importance

A Random Forest Regressor was selected as the forecasting baseline. Power demand exhibits highly non-linear relationships with weather (e.g., demand stays relatively flat until the temperature crosses a specific AC-trigger threshold, at which point it spikes non-linearly). Tree-based models natively capture these thresholds without complex mathematical scaling.

As i have said before as to why i have used random forest regressor instead of models like XGboost, catboost, LightGBM etc. I found that they are not always the optimal starting point for raw time-series forecasting. Power grid data is inherently noisy due to unpredictable weather anomalies. Boosting algorithms sequentially hunt down residual errors, making them highly prone to overfitting this noise without exhaustive hyperparameter tuning. Random Forest utilizes parallel bagging. If one tree memorizes a random anomaly, the other trees out-vote it. This provides a mathematically robust, overfitting-resistant baseline. Future iterations of this pipeline may introduce boosting models, using this Random Forest as the benchmark to ensure the added model complexity actually yields genuine predictive value. Also my MAPE score was prety good which also overall impressed me with this model chosen.

**Hyperparameter Strategy:** To ensure the model generalized well to unseen data, I ran an empirical depth-optimization loop to find the exact inflection point of the bias-variance tradeoff. Testing across multiple magnitudes revealed that a shallow tree (depth 5) severely underfit the data, while an unconstrained tree (infinite depth) began to overfit and memorize historical noise. Hard-capping the pipeline at max_depth=20 hit the mathematical sweet spot, locking in an optimal baseline performance of 1.94% MAPE on the unseen test data.

## 4.Key Insights & Drivers

To ensure model explainability, the Feature Importance is analyzed to identify what factors drive the power grid.

**Analysis of Drivers:**
The model acts as a highly efficient autoregressive engine. Because grid demand is a strictly persistent time-series, the strongest predictor for the load at 3:00 PM is the known load at 2:00 PM. The model heavily weights the current, known demand (demand_mw at 96.53%) to establish a massive mathematical baseline.

It then utilizes the remaining relative weight to adjust for non-linear variance and fine-tune the forecast for the upcoming hour, specifically looking at:<br>Time of Day (hour - 1.64%): Capturing daily human routines and industrial schedules.<br>Weather (temperature_2m - 0.36%): Accounting for immediate meteorological impacts like AC usage.<br>Weather (temperature_2m - 0.36%): Accounting for immediate meteorological impacts like AC usage.



