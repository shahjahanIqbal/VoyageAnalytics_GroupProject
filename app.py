import streamlit as st
import pandas as pd
import joblib


# ============================================================
# PAGE CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="Voyage Analytics",
    page_icon="✈️",
    layout="wide"
)


# ============================================================
# CUSTOM CSS
# ============================================================

st.markdown(
    """
    <style>

    .main {
        background-color: #f5f7fb;
    }

    .title {
        font-size: 42px;
        font-weight: 700;
        color: #1f3c88;
        margin-bottom: 5px;
    }

    .subtitle {
        font-size: 18px;
        color: #6c757d;
        margin-bottom: 25px;
    }

    .prediction-box {
        padding: 25px;
        border-radius: 15px;
        background-color: #ffffff;
        text-align: center;
        box-shadow: 0px 4px 15px rgba(0, 0, 0, 0.10);
        margin-top: 20px;
        margin-bottom: 20px;
    }

    .price {
        font-size: 42px;
        font-weight: bold;
        color: #008000;
    }

    .section-title {
        font-size: 25px;
        font-weight: 600;
        color: #1f3c88;
    }

    </style>
    """,
    unsafe_allow_html=True
)


# ============================================================
# LOAD MODEL AND ENCODERS
# ============================================================

@st.cache_resource
def load_files():

    model = joblib.load("flight_price_model.joblib")

    from_encoder = joblib.load("from_encoder.joblib")

    to_encoder = joblib.load("to_encoder.joblib")

    flightType_encoder = joblib.load(
        "flightType_encoder.joblib"
    )

    agency_encoder = joblib.load(
        "agency_encoder.joblib"
    )

    return (
        model,
        from_encoder,
        to_encoder,
        flightType_encoder,
        agency_encoder
    )


# ============================================================
# LOAD FILES
# ============================================================

try:

    (
        model,
        from_encoder,
        to_encoder,
        flightType_encoder,
        agency_encoder

    ) = load_files()

except Exception as e:

    st.error("❌ Model or encoder files could not be loaded.")

    st.write(
        """
        Make sure these files are present in the same folder as app.py:

        1. flight_price_model.joblib
        2. from_encoder.joblib
        3. to_encoder.joblib
        4. flightType_encoder.joblib
        5. agency_encoder.joblib
        """
    )

    st.exception(e)

    st.stop()


# ============================================================
# GET MODEL FEATURES
# ============================================================

try:

    model_features = list(model.feature_names_in_)

except AttributeError:

    # If feature_names_in_ is not available,
    # use the features used during our training.

    model_features = [
        "from",
        "to",
        "flightType",
        "time",
        "distance",
        "agency",
        "year",
        "month",
        "day"
    ]


# ============================================================
# HEADER
# ============================================================

st.markdown(
    '<div class="title">✈️ Voyage Analytics</div>',
    unsafe_allow_html=True
)

st.markdown(
    '<div class="subtitle">'
    'Flight Price Prediction using Machine Learning'
    '</div>',
    unsafe_allow_html=True
)


# ============================================================
# SIDEBAR
# ============================================================

st.sidebar.title("✈️ Voyage Analytics")

st.sidebar.markdown("### About Project")

st.sidebar.info(
    """
    Voyage Analytics is a Machine Learning application
    that predicts the estimated price of a flight.

    **Machine Learning Model**

    Random Forest Regressor

    **Problem Type**

    Regression

    **Target Variable**

    Flight Price
    """
)

st.sidebar.markdown("---")

st.sidebar.markdown("### How It Works")

st.sidebar.write(
    "1. Enter flight information"
)

st.sidebar.write(
    "2. Select travel date"
)

st.sidebar.write(
    "3. Click Predict Flight Price"
)

st.sidebar.write(
    "4. Get estimated price"
)


# ============================================================
# FLIGHT INFORMATION
# ============================================================

st.markdown(
    '<div class="section-title">🛫 Flight Information</div>',
    unsafe_allow_html=True
)

st.write("Enter the details of your flight.")


# ============================================================
# SOURCE / DESTINATION / FLIGHT TYPE
# ============================================================

col1, col2, col3 = st.columns(3)


with col1:

    source = st.selectbox(
        "From",
        options=list(from_encoder.classes_)
    )


with col2:

    destination = st.selectbox(
        "To",
        options=list(to_encoder.classes_)
    )


with col3:

    flight_type = st.selectbox(
        "Flight Type",
        options=list(flightType_encoder.classes_)
    )


# ============================================================
# TIME / DISTANCE / AGENCY
# ============================================================

col1, col2, col3 = st.columns(3)


with col1:

    time = st.number_input(
        "Travel Time",
        min_value=0.0,
        value=2.0,
        step=0.5
    )


with col2:

    distance = st.number_input(
        "Distance",
        min_value=0.0,
        value=500.0,
        step=10.0
    )


with col3:

    agency = st.selectbox(
        "Agency",
        options=list(agency_encoder.classes_)
    )


# ============================================================
# TRAVEL DATE
# ============================================================

st.markdown(
    '<div class="section-title">📅 Travel Date</div>',
    unsafe_allow_html=True
)

st.write("Select the travel date.")


col1, col2, col3 = st.columns(3)


with col1:

    year = st.number_input(
        "Year",
        min_value=2000,
        max_value=2100,
        value=2023,
        step=1
    )


with col2:

    month = st.number_input(
        "Month",
        min_value=1,
        max_value=12,
        value=5,
        step=1
    )


with col3:

    day = st.number_input(
        "Day",
        min_value=1,
        max_value=31,
        value=15,
        step=1
    )


# ============================================================
# DAY OF WEEK
# ============================================================

day_names = [
    "Monday",
    "Tuesday",
    "Wednesday",
    "Thursday",
    "Friday",
    "Saturday",
    "Sunday"
]

day_name = st.selectbox(
    "Day of Week",
    day_names
)

dayofweek = day_names.index(day_name)


# ============================================================
# PREDICT BUTTON
# ============================================================

st.write("")

predict_button = st.button(
    "🔮 Predict Flight Price",
    type="primary",
    use_container_width=True
)


# ============================================================
# PREDICTION
# ============================================================

if predict_button:

    try:

        # ----------------------------------------------------
        # ENCODE CATEGORICAL VALUES
        # ----------------------------------------------------

        source_encoded = from_encoder.transform(
            [source]
        )[0]

        destination_encoded = to_encoder.transform(
            [destination]
        )[0]

        flight_type_encoded = flightType_encoder.transform(
            [flight_type]
        )[0]

        agency_encoded = agency_encoder.transform(
            [agency]
        )[0]


        # ----------------------------------------------------
        # CREATE ALL POSSIBLE INPUT FEATURES
        # ----------------------------------------------------

        input_data = pd.DataFrame({

            "from": [source_encoded],

            "to": [destination_encoded],

            "flightType": [flight_type_encoded],

            "time": [time],

            "distance": [distance],

            "agency": [agency_encoded],

            "year": [year],

            "month": [month],

            "day": [day],

            "dayofweek": [dayofweek]

        })


        # ----------------------------------------------------
        # CHECK MODEL FEATURES
        # ----------------------------------------------------

        st.write("")

        # Keep only features that the model was trained with
        input_data = input_data.reindex(
            columns=model_features
        )


        # ----------------------------------------------------
        # CHECK FOR MISSING FEATURES
        # ----------------------------------------------------

        if input_data.isnull().any().any():

            missing_features = input_data.columns[
                input_data.isnull().any()
            ].tolist()

            st.error(
                "Some required model features are missing:"
            )

            st.write(missing_features)

            st.stop()


        # ----------------------------------------------------
        # MAKE PREDICTION
        # ----------------------------------------------------

        prediction = model.predict(
            input_data
        )

        predicted_price = prediction[0]


        # ----------------------------------------------------
        # DISPLAY PREDICTED PRICE
        # ----------------------------------------------------

        st.markdown(
            
            f"""
            <div class="prediction-box">
                <h2>💰 Estimated Flight Price</h2>
                <div class="price">₹ {predicted_price:,.2f}</div>
            </div>
            """,
            unsafe_allow_html=True
        )
        


        # ----------------------------------------------------
        # SUCCESS MESSAGE
        # ----------------------------------------------------

        st.success(
            "✅ Flight price predicted successfully!"
        )


        # ====================================================
        # DISPLAY INPUT DETAILS
        # ====================================================

        st.markdown(
            '<div class="section-title">'
            '📋 Flight Details'
            '</div>',
            unsafe_allow_html=True
        )


        details = pd.DataFrame({

            "Feature": [

                "From",

                "To",

                "Flight Type",

                "Travel Time",

                "Distance",

                "Agency",

                "Year",

                "Month",

                "Day",

                "Day of Week"

            ],

            "Value": [

                source,

                destination,

                flight_type,

                time,

                distance,

                agency,

                year,

                month,

                day,

                day_name

            ]

        })


        st.table(details)


        # ====================================================
        # MODEL INFORMATION
        # ====================================================

        st.markdown(
            '<div class="section-title">'
            '🤖 Model Information'
            '</div>',
            unsafe_allow_html=True
        )


        col1, col2, col3 = st.columns(3)


        with col1:

            st.metric(
                "Model",
                "Random Forest"
            )


        with col2:

            st.metric(
                "Problem",
                "Regression"
            )


        with col3:

            st.metric(
                "Target",
                "Flight Price"
            )


    # ========================================================
    # ERROR HANDLING
    # ========================================================

    except Exception as e:

        st.error(
            "❌ An error occurred while making the prediction."
        )

        st.exception(e)


# ============================================================
# MODEL FEATURES - OPTIONAL DEBUG INFORMATION
# ============================================================

with st.expander("🔍 View Model Features"):

    st.write(
        "The saved model expects these features:"
    )

    st.write(model_features)


# ============================================================
# FOOTER
# ============================================================

st.markdown("---")

st.markdown(
    """
    <center>

    <b>✈️ Voyage Analytics</b>

    <br>

    Flight Price Prediction using Machine Learning

    <br><br>

    Built with Python • Scikit-learn • Streamlit

    </center>
    """,
    unsafe_allow_html=True
)