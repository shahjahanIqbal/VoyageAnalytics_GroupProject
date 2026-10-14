**Voyage Analytics — Flight Price Prediction**

Voyage Analytics is an end-to-end Machine Learning project that predicts the estimated price of a flight from historical flight data. The project covers data preprocessing, feature selection, categorical encoding, regression model training, model evaluation, model serialization, and Streamlit deployment.

**Objectives**

- Understand and preprocess flight data.
- Select relevant features for price prediction.
- Encode categorical variables.
- Split data into training and testing sets.
- Train regression models.
- Evaluate model performance using MAE, MSE, RMSE, and R².
- Save the trained model and encoders with Joblib.
- Build a simple and professional Streamlit prediction interface.

**Final Model**

**Random Forest Regressor**

The final application uses a Random Forest regression model. The saved model is loaded by the Streamlit application to predict the price of new flight inputs.

**Dataset**

The project uses flight.csv.

Important features include:

<div class="joplin-table-wrapper"><table><thead><tr><th><p><strong>Feature</strong></p></th><th><p><strong>Description</strong></p></th></tr></thead><tbody><tr><td><pre><code>from</code></pre></td><td><p>Source/departure location</p></td></tr><tr><td><pre><code>to</code></pre></td><td><p>Destination location</p></td></tr><tr><td><pre><code>flightType</code></pre></td><td><p>Flight type/class</p></td></tr><tr><td><pre><code>time</code></pre></td><td><p>Travel time</p></td></tr><tr><td><pre><code>distance</code></pre></td><td><p>Flight distance</p></td></tr><tr><td><pre><code>agency</code></pre></td><td><p>Agency information</p></td></tr><tr><td><pre><code>year</code></pre></td><td><p>Travel year</p></td></tr><tr><td><pre><code>month</code></pre></td><td><p>Travel month</p></td></tr><tr><td><pre><code>day</code></pre></td><td><p>Travel day</p></td></tr><tr><td><pre><code>price</code></pre></td><td><p>Target variable — flight price</p></td></tr></tbody></table></div>

The exact feature set used by the saved model must match the features used during training.

**Machine Learning Workflow**

```
flight.csv
   ↓
Data Loading
   ↓
Data Cleaning
   ↓
Exploratory Data Analysis
   ↓
Feature Selection
   ↓
Categorical Encoding
   ↓
Train/Test Split
   ↓
Model Training
   ↓
Model Evaluation
   ↓
Random Forest Selection
   ↓
Save Model + Encoders
   ↓
Streamlit Deployment
   ↓
Flight Price Prediction
```

**Data Preprocessing**

The preprocessing stage includes:

- Checking dataset shape and data types.
- Checking missing values.
- Checking duplicate records.
- Separating features and target.
- Encoding categorical variables.
- Preparing the final feature set for model training.

Example:

```
X = df.drop("price", axis=1)
y = df["price"]
```

**Categorical Encoding**

Categorical features such as from, to, flightType, and agency are converted into numerical values using encoders.

The trained encoders are saved and reused by Streamlit so that new user inputs receive the same numerical representation used during model training.

Example:

```
from sklearn.preprocessing import LabelEncoder
le_from = LabelEncoder()
df["from"] = le_from.fit_transform(df["from"])
```

**Train/Test Split**

The dataset is split into training and testing sets.

```
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42
)
```

- Training data: 80%
- Testing data: 20%

**Feature Scaling**

The final Random Forest model is trained on **unscaled features**.

Random Forest and Decision Tree models generally do not require StandardScaler because their split decisions are not based on feature magnitude in the same way as distance-based or gradient-based algorithms.

Therefore, the Streamlit application does **not** apply StandardScaler before prediction for this Random Forest model.

**Random Forest Training**

```
from sklearn.ensemble import RandomForestRegressor
rf_model = RandomForestRegressor(
    n_estimators=100,
    random_state=42
)
rf_model.fit(X_train, y_train)
```

**Model Evaluation**

The following regression metrics can be used:

**MAE — Mean Absolute Error**

```
from sklearn.metrics import mean_absolute_error
mae = mean_absolute_error(y_test, predictions)
```

**MSE — Mean Squared Error**

```
from sklearn.metrics import mean_squared_error
mse = mean_squared_error(y_test, predictions)
```

**RMSE — Root Mean Squared Error**

```
import numpy as np
rmse = np.sqrt(
    mean_squared_error(y_test, predictions)
)
```

**R² Score**

```
from sklearn.metrics import r2_score
r2 = r2_score(y_test, predictions)
```

**Model Saving**

The trained model and preprocessing encoders are saved using Joblib.

```
import joblib
joblib.dump(rf_model, "flight_price_model.joblib")
joblib.dump(le_from, "from_encoder.joblib")
joblib.dump(le_to, "to_encoder.joblib")
joblib.dump(le_flightType, "flightType_encoder.joblib")
joblib.dump(le_agency, "agency_encoder.joblib")
```

Only one serialization format is required. This project uses .joblib consistently.

**Streamlit Application**

The Streamlit interface allows the user to enter:

- From
- To
- Flight Type
- Travel Time
- Distance
- Agency
- Year
- Month
- Day
- Day of Week

The application then:

```
User Input
    ↓
Load Saved Encoders
    ↓
Encode Categorical Values
    ↓
Arrange Model Features
    ↓
Load Random Forest Model
    ↓
Predict Flight Price
    ↓
Display Estimated Price
```

**Project Structure**

```
Voyage-Analytics/
│
├── flight.csv
├── flight_price_prediction.ipynb
├── app.py
│
├── flight_price_model.joblib
├── from_encoder.joblib
├── to_encoder.joblib
├── flightType_encoder.joblib
├── agency_encoder.joblib
│
├── requirements.txt
└── README.md
```

**File Description**

<div class="joplin-table-wrapper"><table><thead><tr><th><p><strong>File</strong></p></th><th><p><strong>Purpose</strong></p></th></tr></thead><tbody><tr><td><pre><code>flight.csv</code></pre></td><td><p>Flight dataset</p></td></tr><tr><td><pre><code>flight_price_prediction.ipynb</code></pre></td><td><p>EDA, preprocessing, training and evaluation</p></td></tr><tr><td><pre><code>app.py</code></pre></td><td><p>Streamlit application</p></td></tr><tr><td><pre><code>flight_price_model.joblib</code></pre></td><td><p>Saved Random Forest model</p></td></tr><tr><td><pre><code>from_encoder.joblib</code></pre></td><td><p>Saved source encoder</p></td></tr><tr><td><pre><code>to_encoder.joblib</code></pre></td><td><p>Saved destination encoder</p></td></tr><tr><td><pre><code>flightType_encoder.joblib</code></pre></td><td><p>Saved flight-type encoder</p></td></tr><tr><td><pre><code>agency_encoder.joblib</code></pre></td><td><p>Saved agency encoder</p></td></tr><tr><td><pre><code>requirements.txt</code></pre></td><td><p>Python dependencies</p></td></tr><tr><td><pre><code>README.md</code></pre></td><td><p>Project documentation</p></td></tr></tbody></table></div>

**Installation**

Create a virtual environment:

```
python -m venv venv
```

**Windows**

```
venv\Scripts\activate
```

**Linux/macOS**

```
source venv/bin/activate
```

Install dependencies:

```
pip install -r requirements.txt
```

If requirements.txt is not available:

```
pip install pandas numpy scikit-learn matplotlib seaborn joblib streamlit
```

**Run the Application**

Open a terminal in the project folder:

```
streamlit run app.py
```

Then open the local Streamlit address shown in the terminal, commonly:

```
http://localhost:8501
```