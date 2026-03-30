# Task
The dataset was extracted from kaggle.
The project involved developing a time series forecasting model to predict 'Global_active_power' for energy management. The `household_power_consumption.csv` dataset was used, processed by reducing it to the last 1000 rows, handling missing values through mean imputation and linear interpolation, converting data types, and creating a 'datetime' index with engineered features (year, month, day, day_of_week, hour). Exploratory Data Analysis (EDA) revealed the distribution of 'Global_active_power', strong correlations (e.g., positive with 'Global_intensity', negative with 'Voltage'), and daily/weekly consumption patterns. Outliers were detected using Z-score analysis in several columns, though no explicit treatment was applied. Autocorrelation analysis indicated strong persistence in the series.

An initial Prophet model, trained solely on 'Global_active_power', showed an MAE of 1.073, MSE of 1.781, and RMSE of 1.335. A second Prophet model, significantly improved by incorporating external regressors ('Global_intensity', 'Sub_metering_3', 'Voltage', 'hour', 'day_of_week', 'month'), achieved much better performance with an MAE of 0.025, MSE of 0.002, and RMSE of 0.041. An Exponential Smoothing (ETS) model was also trained, but its performance (MAE: 1.110, MSE: 2.059, RMSE: 1.435) was inferior to the regressor-enhanced Prophet model. Hyperparameter tuning for the Prophet model (without regressors) using cross-validation identified optimal parameters (changepoint_prior_scale=0.5, seasonality_prior_scale=0.1, daily_seasonality=True), resulting in an average RMSE of 0.563 and MAE of 0.371. The best-performing Prophet model (with regressors) was saved as `prophet_model_with_regressors.pkl`, and a Streamlit application was created for real-time predictions. The project highlights the critical role of external regressors in enhancing time series forecasting accuracy.

## Dataset Description and Preprocessing Summary

This project began with the `household_power_consumption.csv` dataset. To focus on recent data and manage computational load, the dataset was initially reduced to its **last 1000 rows**.

### Handling of Missing Values:

2.  **'?' to NaN Conversion**: During the conversion of several power-related columns to numeric data types, non-numeric entries (such as '?') were automatically coerced into `NaN` (Not a Number) values. This was achieved by using `pd.to_numeric` with the `errors='coerce'` argument.
3.  **Linear Interpolation**: After resampling the data to a regular minute frequency, any remaining `NaN` values that emerged or persisted were filled using linear interpolation. This method estimates missing values based on the values before and after them, ensuring a continuous time series.

### Data Type Conversion:
The core power consumption columns, specifically 'Global_active_power', 'Global_reactive_power', 'Voltage', 'Global_intensity', 'Sub_metering_1', 'Sub_metering_2', and 'Sub_metering_3', were converted from their initial object type to **numeric (float64)** types. This step was crucial for performing mathematical operations and time series analysis.

### Datetime Index and Feature Engineering:
1.  **Datetime Index Creation**: The 'Date' and 'Time' columns were combined into a new column named `datetime_col`. This combined string was then converted into proper `datetime` objects using `pd.to_datetime` (with `dayfirst=True` to handle the date format). This `datetime_col` was subsequently set as the DataFrame's index, making the dataset a time series.
2.  **Time-Based Feature Engineering**: Several useful time-based features were extracted from the new `datetime` index:
    *   `year`: The year of each observation.
    *   `month`: The month of each observation.
    *   `day`: The day of the month of each observation.
    *   `day_of_week`: The day of the week (Monday=0, Sunday=6) for each observation.
    *   `hour`: The hour of the day for each observation.

These preprocessing steps ensured the data was clean, in the correct format, and enriched with temporal features essential for time series forecasting.

## Exploratory Data Analysis (EDA) Summary

Based on the performed EDA, here are the key insights regarding the power consumption data:

1.  **Time Series Plot of Global Active Power :**
    *   The initial time series plot of 'Global_active_power' over time (before resampling) showed fluctuations with no immediately apparent long-term trend due to the limited timeframe of the sampled data (around 1 day). Some distinct spikes in power consumption were observed. The plot after cleaning and resampling provided a clearer, more continuous representation of these fluctuations, indicating varying power demands throughout the observed period. However, the data is very short, spanning only about 16 hours.

2.  **Distribution of Global Active Power :**
    *   The histogram of 'Global_active_power'  reveals a right-skewed distribution, indicating that lower power consumption values are more frequent, while higher consumption levels occur less often. The descriptive statistics  confirm this, with a mean of approximately 0.779 kW and a median of 0.420 kW, showing that the mean is pulled towards the higher values by some larger readings. The standard deviation of 0.776 kW indicates a notable spread in the data, ranging from a minimum of 0.100 kW to a maximum of 3.848 kW.

3.  **Correlation Analysis **
    *   The correlation matrix heatmap shows strong relationships between 'Global_active_power' and other electrical measurements:
        *   **Global_intensity:** There is a very strong positive correlation (0.99) between 'Global_active_power' and 'Global_intensity'. This is expected as higher power consumption directly translates to higher electrical current.
        *   **Voltage:** A strong negative correlation (-0.70) exists between 'Global_active_power' and 'Voltage'. This suggests that as global active power increases, voltage tends to decrease, possibly indicating increased load on the system.
        *   **Sub_metering_1, Sub_metering_2, Sub_metering_3:** These sub-metering values also show positive correlations with 'Global_active_power' (0.69, 0.28, 0.04 respectively), as they represent components of the total active power.

4.  **Daily and Weekly Patterns:**
    *   **Daily Pattern:** The plot of 'Average Global Active Power by Hour of Da reveals a clear daily cycle. Power consumption is generally lower during late night/early morning hours (e.g., 0-3 AM), starts to increase in the morning, peaks around mid-morning (e.g., 9-10 AM) and then decreases towards the afternoon, before potentially rising again in the evening. The observed data covers only a portion of two days.
    *   **Weekly Pattern:** The plot of 'Average Global Active Power by Day of Week' (code cell `4c6f9410`) is based on very limited data (only Tuesday and Wednesday from the sampled data). Thus, it's difficult to draw definitive conclusions about weekly patterns, but it appears to show a slight decrease in average power from Tuesday to Wednesday in the observed short period.

**Consolidated Summary:**

The 'Global_active_power' data exhibits a right-skewed distribution with most consumption occurring at lower levels, punctuated by occasional high-demand spikes. There's a very strong direct relationship with 'Global_intensity' and an inverse relationship with 'Voltage'. Although the dataset is limited to approximately 16 hours, a noticeable daily consumption pattern emerges, with lower usage overnight and a peak in the mid-morning. Due to the limited time frame, weekly patterns cannot be reliably assessed from this subset of the data.

### Outlier Detection Summary

Outliers were detected using **Z-score analysis**. A **Z-score threshold of 3** was applied, meaning any data point with an absolute Z-score greater than 3 was considered an outlier.

Outliers were identified in the following columns:
- `Global_active_power`
- `Global_reactive_power`
- `Voltage`
- `Global_intensity`
- `Sub_metering_1`

No outliers were found in `Sub_metering_2` and `Sub_metering_3` based on this threshold.

It is important to note that **no explicit treatment** (e.g., removal, capping, or transformation) was applied to these detected outliers in the subsequent steps of the analysis. The data was kept as is after identification.

### Insights from ACF and PACF Plots for Global Active Power

**Autocorrelation Function (ACF) Analysis:**

The ACF plot shows a very slow decay, indicating strong persistence in the 'Global_active_power' time series. This slow decay suggests that observations at earlier time steps have a lasting influence on future observations, which is a common characteristic of non-stationary time series. The high autocorrelation values persist for many lags, implying that the series might need differencing to achieve stationarity.

**Partial Autocorrelation Function (PACF) Analysis:**

The PACF plot shows a significant spike at Lag 1, which then quickly drops off to non-significant levels. This strong, direct dependency at Lag 1 is highly indicative of an AutoRegressive (AR) process of order 1, or AR(1). It means that the current observation is directly and strongly correlated with the immediately preceding observation, after removing the effects of correlations at other lags.

**Summary:**

Both plots together suggest that the 'Global_active_power' series is likely non-stationary due to the strong persistence observed in the ACF. The pronounced spike at Lag 1 in the PACF further supports the presence of a strong AR(1) characteristic, meaning that a value from the previous time step is a good direct predictor of the current value. These insights are crucial for selecting appropriate time series forecasting models, such as ARIMA/SARIMA, where differencing might be required to handle non-stationarity and the AR(1) component can be explicitly modeled.

## Model Training and Evaluation (Prophet without Regressors)

### Subtask:
Detail the training of an initial Prophet model using only the target variable, its evaluation metrics (MAE, MSE, RMSE), and the visual performance against actuals.

#### Instructions
1.  **Data Splitting and Preparation:**
    -   The `Global_active_power` column was selected as the target variable for the Prophet model. 
    -   A new DataFrame (`df_model`) was created containing only this target variable. 
    -   The data was then chronologically split into training and testing sets, with 80% for training and 20% for testing. 
    -   For Prophet compatibility, the training data's index (`datetime_col`) was renamed to `ds` and `Global_active_power` to `y`. This prepared `prophet_df_train` for model fitting.

2.  **Prophet Model Training (No Regressors):**
    -   A Prophet model was instantiated using `model = Prophet()`.
    -   Crucially, at this stage, no additional regressors were added to the model to assess its baseline performance using only the time series data itself.
    -   The model was then fitted to the prepared training data (`prophet_df_train`).

3.  **Model Evaluation and Metrics:**
    -   Predictions were generated on the test set (`test_df`) using the trained Prophet model.
    -   The predicted values (`yhat`) were compared against the actual `Global_active_power` values from the test set.
    -   The following evaluation metrics were calculated:
        -   **Mean Absolute Error (MAE):** 1.073
        -   **Mean Squared Error (MSE):** 1.781
        -   **Root Mean Squared Error (RMSE):** 1.335
    -   These metrics indicate a relatively high error, suggesting that a model relying solely on the time series trend and seasonality, without other influencing factors, may not be sufficiently accurate for this dataset.

4.  **Visual Performance Analysis:**
    -   A plot was generated to visualize the Prophet forecast against the actual `Global_active_power` values on the test set.
    -   **Visual Observations:** The plot (from previous output) shows that while the Prophet model captures some general trend, its predictions are significantly smoother than the actual data and fail to capture minute-to-minute fluctuations or sharper changes. The prediction interval (pink shaded area) is quite wide, reflecting the model's uncertainty and the high error metrics. This visually confirms that the model's performance without additional regressors is limited, leading to a relatively poor fit to the actual power consumption patterns.

    ## Model Training and Evaluation (Prophet with Regressors)

### Subtask:
Describe the improved Prophet model incorporating external regressors ('Global_intensity', 'Sub_metering_3', 'Voltage', 'hour', 'day_of_week', 'month'), its significantly better evaluation metrics (MAE, MSE, RMSE), and the enhanced visual fit to actual data.

#### Instructions

1.  **Data Preparation with Regressors**

    To enhance the Prophet model's predictive power, we incorporated external regressors. First, a new DataFrame `df_model` was created, containing the target variable `Global_active_power` and a set of selected input features: `Global_intensity`, `Sub_metering_3`, `Voltage`, `hour`, `day_of_week`, and `month`. These features were chosen due to their potential influence on power consumption.

    The data was then chronologically split into training and testing sets, with an 80/20 ratio, ensuring that the model is trained on past data and evaluated on future, unseen data. Specifically, `train_df` received the first 80% of the data, and `test_df` received the remaining 20%.

    For Prophet's specific input requirements, the training data was further prepared. The DataFrame's `datetime_col` (which was previously set as the index) was reset as a regular column and renamed to `ds` (for 'datestamp'), and the `Global_active_power` column was renamed to `y` (for 'target variable'). This `prophet_df_train` DataFrame, now including the renamed time series and the selected regressors, was ready for model training.

    2.  **Prophet Model Training with Regressors**

    The Prophet model was initialized without any predefined seasonalities (yearly, weekly, and daily seasonality were explicitly disabled, likely due to the limited time frame of the dataset or to allow the regressors to capture these patterns). For each of the `input_features` (`Global_intensity`, `Sub_metering_3`, `Voltage`, `hour`, `day_of_week`, `month`), the `model.add_regressor()` method was called. This step is crucial as it informs Prophet to consider these external variables when modeling the time series, allowing it to capture more complex relationships and improve predictive accuracy. Finally, the model was fitted using the `prophet_df_train` DataFrame, which contains both the target variable `y` and the specified regressors, enabling the model to learn from these additional features.

    3.  **Model Evaluation and Metrics**

    Predictions were made on the test set by first preparing a `future` DataFrame that included the `ds` column (for the datestamp) and all the specified `input_features` (`Global_intensity`, `Sub_metering_3`, `Voltage`, `hour`, `day_of_week`, `month`) for the corresponding test period. This ensured that the model had all the necessary regressor values to generate accurate forecasts.

    The evaluation metrics for this Prophet model, incorporating external regressors, showed a significant improvement:

    *   **Mean Absolute Error (MAE):** 0.025
    *   **Mean Squared Error (MSE):** 0.002
    *   **Root Mean Squared Error (RMSE):** 0.041

    Comparing these to the metrics of the Prophet model *without* regressors (MAE: 1.073, MSE: 1.781, RMSE: 1.335), it is evident that the inclusion of regressors led to a drastic reduction in prediction errors. The MAE decreased by approximately 97.7%, the MSE by about 99.8%, and the RMSE by approximately 96.9%, demonstrating a substantially more accurate and robust forecasting model.

    4.  **Visual Performance Analysis**

    The visual comparison of the Prophet forecast (with regressors) against the actual `Global_active_power` values clearly demonstrates a significantly enhanced fit. The red dashed line representing the Prophet forecast closely tracks the blue line of the actual data, indicating that the model is now much better at capturing the fluctuations and trends in power consumption. Furthermore, the pink shaded area, which denotes the prediction interval (`yhat_lower` to `yhat_upper`), is noticeably narrower compared to the model without regressors. This reduced width signifies increased confidence in the predictions and reflects the model's improved accuracy and reduced uncertainty, largely attributed to the valuable information provided by the external regressors. This visual evidence strongly supports the quantitative improvements observed in the MAE, MSE, and RMSE metrics.


    ## Model Training and Evaluation (ETS Model)

### Subtask:
Summarize the Exponential Smoothing (ETS) model's configuration, its performance metrics (MAE, MSE, RMSE), and a comparison noting its lower accuracy compared to the regressor-enhanced Prophet.

### ETS Model Summary

1.  **Configuration**: The Exponential Smoothing (ETS) model was configured with an **additive trend** (`trend='add'`), **no seasonal component** (`seasonal=None`), and **no damped trend** (`damped_trend=False`). It was trained on the `Global_active_power` from the training dataset.

2.  **Performance Metrics (ETS Model)**:
    *   **Mean Absolute Error (MAE)**: 1.110
    *   **Mean Squared Error (MSE)**: 2.059
    *   **Root Mean Squared Error (RMSE)**: 1.435

### Comparison with Regressor-Enhanced Prophet Model

The Prophet model, when enhanced with regressors, demonstrated significantly better performance compared to the ETS model:

*   **Prophet (with Regressors) MAE**: 0.025
*   **Prophet (with Regressors) MSE**: 0.002
*   **Prophet (with Regressors) RMSE**: 0.041

**Conclusion**: The ETS model, while providing a baseline for time series forecasting, exhibited considerably lower accuracy (higher MAE, MSE, and RMSE) than the Prophet model which incorporated additional relevant features (regressors). This suggests that the exogenous variables (e.g., Global_intensity, Sub_metering_3, Voltage, hour, day_of_week, month) played a crucial role in improving the prediction accuracy for `Global_active_power`.


### Subtask
Summarize the Exponential Smoothing (ETS) model's configuration, its performance metrics (MAE, MSE, RMSE), and a comparison noting its lower accuracy compared to the regressor-enhanced Prophet.

#### Instructions
1. Define the training and testing sets for the ETS model, using `y_train` and `y_test` which contain the 'Global_active_power' values.
2. Instantiate the `ExponentialSmoothing` model from `statsmodels.tsa.holtwinters` with `trend='add'`, `seasonal=None`, and `damped_trend=False`.
3. Fit the ETS model to the training data (`y_train`).
4. Generate predictions using the fitted ETS model for the length of the test set (`len(y_test)`).
5. Calculate the Mean Absolute Error (MAE), Mean Squared Error (MSE), and Root Mean Squared Error (RMSE) by comparing the ETS forecast with the actual `y_test` values.
6. Create a plot to visualize the actual training data, actual test data, and the ETS forecast on the test set, including appropriate labels and titles for clarity.

## Hyperparameter Tuning for Prophet

### Explanation of Prophet Cross-Validation

1.  **Purpose of Hyperparameter Tuning and Cross-Validation**: In Prophet models, hyperparameter tuning is crucial for optimizing forecasting performance. Parameters like `changepoint_prior_scale` (controlling the flexibility of trend changes) and `seasonality_prior_scale` (controlling the strength of seasonality) significantly impact how the model fits the data and generalizes to future observations. Cross-validation is used to systematically evaluate different combinations of these hyperparameters by simulating historical forecasts. This helps us find the set of parameters that yield the best performance on unseen data, preventing overfitting and ensuring reliable predictions.

2.  **Cross-Validation Setup**: We utilized Prophet's built-in `cross_validation` and `performance_metrics` functions. The `cross_validation` function was configured as follows:
    *   `initial='360 minutes'`: The first training period included 360 minutes of data.
    *   `period='60 minutes'`: The model was re-trained every 60 minutes.
    *   `horizon='120 minutes'`: Each forecast extended 120 minutes into the future.
    *   `parallel='processes'`: Cross-validation was parallelized using multiple processes for faster computation.

    The `performance_metrics` function then calculated `RMSE` (Root Mean Squared Error) and `MAE` (Mean Absolute Error) for each forecast generated by the cross-validation process.

3.  **Parameter Grid Explored**:
    The following hyperparameters and their ranges were explored:
    *   `changepoint_prior_scale`: `[0.001, 0.01, 0.1, 0.5]` - This parameter adjusts the flexibility of the trend. A smaller value makes the trend less flexible (underfitting), while a larger value allows for more trend changes (potential overfitting).
    *   `seasonality_prior_scale`: `[0.01, 0.1, 1.0, 10.0]` - This controls the strength of the seasonality components. A smaller value shrinks the seasonal components, and a larger value allows them to be more prominent.
    *   `daily_seasonality`: `[False, True]` - Given that our data is recorded at minute intervals, daily seasonality is highly relevant for capturing intra-day patterns. We tested both including and excluding it.

    `weekly_seasonality` and `yearly_seasonality` were explicitly set to `False`. This decision was made because the dataset contains a very limited time range (a few days from the full dataset), which is insufficient to reliably capture weekly or yearly patterns.

4.  **Best Parameters Found**:
    Through the cross-validation process, the best performing Prophet model (without regressors) was identified with the following parameters:
    *   `best_params_prophet`: `{'changepoint_prior_scale': 0.5, 'seasonality_prior_scale': 0.1, 'daily_seasonality': True}`
    *   `best_rmse_prophet`: `0.563`
    *   `best_mae_prophet`: `0.371`

    These parameters resulted in the lowest average RMSE and MAE across the simulated forecasts, indicating a good balance between model complexity and generalization ability for this specific dataset and forecast horizon.

5.  **Plot Interpretation (RMSE and MAE over Forecast Horizon)**:
    The generated plots (`plot_cross_validation_metric`) visualize the RMSE and MAE values as a function of the forecast horizon. These plots are crucial for understanding how the model's performance degrades or improves as we predict further into the future. Ideally, we look for a relatively stable or slowly increasing trend in both RMSE and MAE, indicating consistent performance. Any sharp increases would suggest a rapid decline in prediction accuracy beyond a certain horizon. The plots for the best model show the RMSE and MAE across the 120-minute forecast horizon, providing a visual assessment of its predictive reliability over short-term predictions.


    ## Model Saving

### Subtask:
Document the saving of the best-performing Prophet model (with regressors) as a pickle file for deployment.

### Explanation of Model Saving

After identifying the best-performing Prophet model (which includes additional regressors for improved accuracy), it is essential to save this trained model for future use. The model is saved as a Python pickle file named `prophet_model_with_regressors.pkl`. 

The primary purpose of saving the model is to enable its deployment without requiring retraining. This saved file can be loaded into other applications for real-time predictions, batch forecasting, or further analysis, significantly streamlining the operationalization of the forecasting solution and ensuring consistency in predictions.


## Streamlit Deployment Summary

This Streamlit application serves as an interactive interface for users to obtain real-time predictions for 'Global_active_power' using the previously trained Prophet model with regressors. 

**Key features of the application include:**

1.  **Efficient Model Loading:** The application loads the `prophet_model_with_regressors.pkl` file, which contains our trained Prophet model, using Python's `pickle` library. The `@st.cache_resource` decorator from Streamlit ensures that the model is loaded only once when the application starts, optimizing performance.

2.  **Interactive Feature Input:** Users can easily adjust various input features through interactive sliders:
    *   `Global_intensity` (range: 1.0 - 22.4, default: 6.8)
    *   `Sub_metering_3` (range: 0.0 - 30.0, default: 19.0)
    *   `Voltage` (range: 235.65 - 246.65, default: 244.0)
    *   `hour` (range: 0-23, default: 14)
    *   `day_of_week` (range: 0=Monday, 6=Sunday, default: 3)
    
    The `month` feature is fixed to `10` to align with the characteristics of the training data.

3.  **Real-Time Prediction:** Upon clicking the "Predict" button, the application constructs a `future` DataFrame. This DataFrame includes the user-specified feature values and a placeholder `ds` (datestamp) close to the training data's timeline to ensure consistent predictions. The `predict` method of the loaded Prophet model is then invoked with this `future` DataFrame to generate the forecast.

4.  **Result Display:** The predicted 'Global_active_power' (`yhat`) is extracted from the forecast. To maintain physical realism, any negative `yhat` values are clipped to `0`. The application also displays the prediction interval, showing the `yhat_lower` and `yhat_upper` bounds (also clipped to `0` if negative), providing a measure of uncertainty around the prediction.


## Summary:

### Data Analysis Key Findings

*   **Data Preprocessing**: The `household_power_consumption.csv` dataset was reduced to its last 1000 rows. Missing values were handled via mean imputation for 'Sub_metering_3', '?' to NaN conversion, and linear interpolation. Key power-related columns were converted to numeric types, and a 'datetime' index was created with engineered features including year, month, day, day\_of\_week, and hour.
*   **Exploratory Data Analysis (EDA)**:
    *   'Global\_active\_power' exhibited a right-skewed distribution (mean: 0.779 kW, median: 0.420 kW), ranging from 0.100 kW to 3.848 kW.
    *   Strong correlations were observed: 0.99 positive with 'Global\_intensity', -0.70 negative with 'Voltage', and positive with sub-metering values.
    *   A clear daily consumption pattern was identified, with lower usage overnight and a peak in mid-morning. Weekly patterns could not be reliably assessed due to the limited 16-hour data subset.
*   **Outlier Detection**: Z-score analysis (threshold of 3) identified outliers in 'Global\_active\_power', 'Global\_reactive\_power', 'Voltage', 'Global\_intensity', and 'Sub\_metering\_1', though no explicit treatment was applied.
*   **Autocorrelation Analysis**: ACF plots indicated strong persistence and likely non-stationarity, while PACF plots showed a significant spike at Lag 1, suggesting an AR(1) characteristic.
*   **Prophet Model Performance**:
    *   **Without Regressors**: An initial Prophet model, trained solely on 'Global\_active\_power', yielded an MAE of 1.073, MSE of 1.781, and RMSE of 1.335.
    *   **With Regressors**: Incorporating external regressors ('Global\_intensity', 'Sub\_metering\_3', 'Voltage', 'hour', 'day\_of\_week', 'month') drastically improved performance, achieving an MAE of 0.025, MSE of 0.002, and RMSE of 0.041. This represents approximately a 97.7% reduction in MAE, 99.8% in MSE, and 96.9% in RMSE compared to the model without regressors.
*   **ETS Model Performance**: An Exponential Smoothing (ETS) model (configured with additive trend, no seasonal/damped trend) resulted in an MAE of 1.110, MSE of 2.059, and RMSE of 1.435, performing significantly worse than the regressor-enhanced Prophet model.
*   **Hyperparameter Tuning (Prophet without Regressors)**: Cross-validation identified optimal parameters: `changepoint_prior_scale=0.5`, `seasonality_prior_scale=0.1`, and `daily_seasonality=True`, resulting in an average RMSE of 0.563 and MAE of 0.371.
*   **Deployment**: The best-performing Prophet model (with regressors) was saved as `prophet_model_with_regressors.pkl`, and a Streamlit application was developed for real-time predictions based on user-input features.

### Insights or Next Steps

*   The inclusion of external regressors is paramount for achieving high accuracy in forecasting 'Global\_active\_power', dramatically outperforming models reliant only on time series components or simpler statistical methods like ETS.
*   Future enhancements could involve exploring additional domain-specific regressors (e.g., temperature, holiday indicators), expanding the dataset to capture longer-term seasonalities, and applying advanced hyperparameter tuning specifically to the regressor-enhanced Prophet model.
