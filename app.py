import streamlit as st
import pandas as pd
import joblib


# ============================================================
# PAGE CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="Voyage Analytics",
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
# APP NAVIGATION + DESTINATION PLANNER
# ============================================================

from pathlib import Path
import numpy as np

BASE_DIR = Path(__file__).resolve().parent
OUTPUT_DIR = BASE_DIR / "travel_recommender_outputs"

st.sidebar.markdown("---")
app_page = st.sidebar.radio(
    "Navigate",
    ["Flight Price Predictor", "Destination Planner"],
    key="app_page",
)


@st.cache_data
def load_planner_data():
    users_path = BASE_DIR / "users.csv"
    journeys_path = OUTPUT_DIR / "journeys_train.csv"
    hotels_path = BASE_DIR / "hotels.csv"
    if not users_path.exists():
        raise FileNotFoundError("users.csv was not found beside app.py.")
    if not journeys_path.exists():
        raise FileNotFoundError(
            f"{journeys_path} was not found. Run the journey data-preparation "
            "pipeline first so travel_recommender_outputs/journeys_train.csv exists."
        )
    if not hotels_path.exists():
        raise FileNotFoundError("hotels.csv was not found beside app.py.")

    users = pd.read_csv(users_path)
    history = pd.read_csv(journeys_path)
    hotels = pd.read_csv(hotels_path)
    required_users = {"code", "name"}
    required_history = {"userCode", "origin", "destination", "departure_date"}
    required_hotels = {"name", "place", "days", "price", "total"}
    if not required_users.issubset(users.columns):
        raise ValueError(f"users.csv must contain columns: {sorted(required_users)}")
    if not required_history.issubset(history.columns):
        raise ValueError(
            "journeys_train.csv must contain columns: "
            f"{sorted(required_history)}"
        )
    if not required_hotels.issubset(hotels.columns):
        raise ValueError(f"hotels.csv must contain columns: {sorted(required_hotels)}")

    users["code"] = users["code"].astype(str)
    history["userCode"] = history["userCode"].astype(str)
    history["origin"] = history["origin"].astype(str)
    history["destination"] = history["destination"].astype(str)
    history["departure_date"] = pd.to_datetime(history["departure_date"], errors="coerce")
    history = history.dropna(subset=["departure_date", "origin", "destination"])
    hotels["place"] = hotels["place"].astype(str)
    for column in ("days", "price", "total"):
        hotels[column] = pd.to_numeric(hotels[column], errors="coerce")
    hotels = hotels.dropna(subset=["place", "name", "price"])
    return users, history, hotels


def baseline_recommendations(user_code, origin, history, candidates, k=5):
    """Personalized count-based baseline, using training history only."""
    user_rows = history[history["userCode"] == str(user_code)]
    global_counts = history["destination"].value_counts().to_dict()
    user_counts = user_rows["destination"].value_counts().to_dict()
    route_counts = user_rows[user_rows["origin"] == origin]["destination"].value_counts().to_dict()
    origin_counts = history[history["origin"] == origin]["destination"].value_counts().to_dict()

    rows = []
    for destination in candidates:
        score = (
            route_counts.get(destination, 0),
            user_counts.get(destination, 0),
            origin_counts.get(destination, 0),
            global_counts.get(destination, 0),
        )
        rows.append({
            "Destination": destination,
            "Route history": score[0],
            "User destination history": score[1],
            "Origin popularity": score[2],
            "Global popularity": score[3],
        })
    ranked = pd.DataFrame(rows).sort_values(
        ["Route history", "User destination history", "Origin popularity", "Global popularity", "Destination"],
        ascending=[False, False, False, False, True],
    ).head(k).reset_index(drop=True)
    ranked.insert(1, "Score", range(len(ranked), 0, -1))
    return ranked[["Destination", "Score"]]


def seasonal_recommendations(user_code, origin, departure_month, history, candidates, k=5):
    """Smoothed user + user-month preference blend used for the seasonal experiment."""
    user_rows = history[history["userCode"] == str(user_code)]
    user_counts = user_rows["destination"].value_counts().to_dict()
    total_user = max(len(user_rows), 1)
    month_rows = user_rows[user_rows["departure_date"].dt.month == int(departure_month)]
    month_counts = month_rows["destination"].value_counts().to_dict()
    route_rows = user_rows[user_rows["origin"] == origin]
    route_counts = route_rows["destination"].value_counts().to_dict()
    global_counts = history["destination"].value_counts().to_dict()
    global_total = max(len(history), 1)
    month_total = len(month_rows)
    route_total = len(route_rows)

    weights = {
        "user_destination": 0.45,
        "user_destination_month": 0.35,
        "user_route": 0.15,
        "global_destination": 0.05,
    }
    smoothing = 5.0
    rows = []
    for destination in candidates:
        user_prior = user_counts.get(destination, 0) / total_user
        month_probability = (
            month_counts.get(destination, 0) + smoothing * user_prior
        ) / (month_total + smoothing)
        route_probability = route_counts.get(destination, 0) / max(route_total, 1)
        global_probability = global_counts.get(destination, 0) / global_total
        score = (
            weights["user_destination"] * user_prior
            + weights["user_destination_month"] * month_probability
            + weights["user_route"] * route_probability
            + weights["global_destination"] * global_probability
        )
        rows.append({
            "Destination": destination,
            "Score": float(score),
        })
    return pd.DataFrame(rows).sort_values(
        ["Score", "Destination"], ascending=[False, True]
    ).head(k).reset_index(drop=True)


def random_forest_recommendations(user_code, origin, departure_date, history, candidates, k=5):
    """Inference for the trained candidate-level Random Forest ranker."""
    import joblib
    model_path = OUTPUT_DIR / "candidate_ranker.joblib"
    if not model_path.exists():
        raise FileNotFoundError(
            "The Random Forest artifact is missing. Add joblib.dump(model, "
            "OUT / 'candidate_ranker.joblib') to train_destination_ranker.py "
            "and run that training script once."
        )
    model = joblib.load(model_path)

    user_rows = history[history["userCode"] == str(user_code)]
    user_destination = user_rows["destination"].value_counts().to_dict()
    user_route = user_rows.groupby(["origin", "destination"]).size().to_dict()
    origin_destination = history.groupby(["origin", "destination"]).size().to_dict()
    destination_counts = history["destination"].value_counts().to_dict()
    user_history_count = len(user_rows)
    user_origin_history_count = int((user_rows["origin"] == origin).sum())
    origin_history_count = int((history["origin"] == origin).sum())

    month = departure_date.month
    weekday = departure_date.weekday()
    records = []
    for candidate in candidates:
        records.append({
            "userCode": str(user_code),
            "origin": str(origin),
            "candidate": str(candidate),
            "month_sin": np.sin(2 * np.pi * month / 12),
            "month_cos": np.cos(2 * np.pi * month / 12),
            "day_of_week": weekday,
            "user_candidate_count": user_destination.get(candidate, 0),
            "user_route_count": user_route.get((origin, candidate), 0),
            "origin_candidate_count": origin_destination.get((origin, candidate), 0),
            "candidate_popularity": destination_counts.get(candidate, 0),
            "user_history_count": user_history_count,
            "user_origin_history_count": user_origin_history_count,
            "origin_history_count": origin_history_count,
        })
    feature_frame = pd.DataFrame(records)
    positive_class = list(model.named_steps["classifier"].classes_).index(1)
    feature_frame["Score"] = model.predict_proba(feature_frame)[:, positive_class]
    feature_frame["Destination"] = feature_frame["candidate"]
    return feature_frame[["Destination", "Score"]].sort_values(
        ["Score", "Destination"], ascending=[False, True]
    ).head(k).reset_index(drop=True)


def render_destination_planner():
    st.markdown('<div class="title">Destination Planner</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="subtitle">Personalized destination recommendations from your travel history</div>',
        unsafe_allow_html=True,
    )

    try:
        users, history, hotels = load_planner_data()
    except Exception as exc:
        st.error("Destination planner data could not be loaded.")
        st.code(str(exc))
        st.info(
            "Keep users.csv and the travel_recommender_outputs folder beside app.py. "
            "The folder must contain journeys_train.csv."
        )
        return

    if "planner_user_code" not in st.session_state:
        st.session_state.planner_user_code = None
    if "planner_user_name" not in st.session_state:
        st.session_state.planner_user_name = None

    if st.session_state.planner_user_code is None:
        st.subheader("1. Log In")
        st.write("Enter your name or select an account below.")

        # Name entry and random-user action are separate because Streamlit
        # does not allow ordinary buttons inside a form.
        with st.form("planner_login_form"):
            entered_name = st.text_input("User name", placeholder="Enter your name")
            continue_clicked = st.form_submit_button("Continue to Plan Trip", type="primary")

        random_col, spacer_col = st.columns([1, 3])
        with random_col:
            random_clicked = st.button("Choose random username", use_container_width=True)

        if random_clicked:
            random_row = users.sample(n=1).iloc[0]
            st.session_state.planner_user_code = str(random_row["code"])
            st.session_state.planner_user_name = str(random_row["name"])
            st.session_state.pop("planner_name_matches", None)
            st.session_state.pop("planner_results", None)
            st.rerun()

        if continue_clicked:
            clean_name = entered_name.strip()
            matches = users[users["name"].astype(str).str.strip().str.casefold() == clean_name.casefold()]
            if not clean_name:
                st.warning("Enter your name to continue.")
            elif matches.empty:
                st.error("That name was not found in users.csv. Check the spelling and try again.")
            elif len(matches) > 1:
                st.session_state.planner_name_matches = matches[["code", "name"]].to_dict("records")
                st.rerun()
            else:
                st.session_state.planner_user_code = str(matches.iloc[0]["code"])
                st.session_state.planner_user_name = str(matches.iloc[0]["name"])
                st.session_state.pop("planner_name_matches", None)
                st.rerun()

        if st.session_state.get("planner_name_matches"):
            matches = pd.DataFrame(st.session_state.planner_name_matches)
            options = matches.apply(lambda row: f"{row['name']} (user code: {row['code']})", axis=1).tolist()
            selected = st.selectbox("Multiple accounts match. Choose your account", options, key="duplicate_user_choice")
            if st.button("Use selected account", type="primary"):
                selected_code = selected.rsplit("user code: ", 1)[-1].rstrip(")")
                selected_row = matches[matches["code"].astype(str) == selected_code].iloc[0]
                st.session_state.planner_user_code = selected_code
                st.session_state.planner_user_name = str(selected_row["name"])
                st.session_state.pop("planner_name_matches", None)
                st.rerun()

        # Compact, five-row paginated username directory.
        st.markdown("#### Available users")
        st.caption("Select a user from the list.")
        directory = users[["code", "name"]].copy()
        directory["code"] = directory["code"].astype(str)
        directory["name"] = directory["name"].fillna("").astype(str)
        directory = directory.sort_values(["name", "code"], key=lambda col: col.str.casefold() if col.name == "name" else col).reset_index(drop=True)
        page_size = 5
        total_pages = max(1, (len(directory) + page_size - 1) // page_size)
        if "planner_user_directory_page" not in st.session_state:
            st.session_state.planner_user_directory_page = 0
        st.session_state.planner_user_directory_page = min(
            max(0, int(st.session_state.planner_user_directory_page)), total_pages - 1
        )
        page_col1, page_col2, page_col3 = st.columns([1, 2, 1])
        with page_col1:
            if st.button("← Previous", disabled=st.session_state.planner_user_directory_page == 0, key="user_directory_prev"):
                st.session_state.planner_user_directory_page -= 1
                st.rerun()
        with page_col2:
            st.caption(f"Page {st.session_state.planner_user_directory_page + 1} of {total_pages} · {len(directory)} users")
        with page_col3:
            if st.button("Next →", disabled=st.session_state.planner_user_directory_page >= total_pages - 1, key="user_directory_next"):
                st.session_state.planner_user_directory_page += 1
                st.rerun()

        start = st.session_state.planner_user_directory_page * page_size
        page_users = directory.iloc[start:start + page_size].copy()
        page_users.insert(0, "#", range(start + 1, start + len(page_users) + 1))
        st.dataframe(page_users, hide_index=True, use_container_width=True, height=225)

        if not page_users.empty:
            choice_options = page_users.apply(
                lambda row: f"{row['name']} (user code: {row['code']})", axis=1
            ).tolist()
            chosen_user = st.selectbox("Choose from these five users", choice_options, key="planner_directory_choice")
            if st.button("Continue with selected username", key="choose_directory_user", type="primary", use_container_width=True):
                selected_code = chosen_user.rsplit("user code: ", 1)[-1].rstrip(")")
                selected_row = page_users[page_users["code"].astype(str) == selected_code].iloc[0]
                st.session_state.planner_user_code = str(selected_row["code"])
                st.session_state.planner_user_name = str(selected_row["name"])
                st.session_state.pop("planner_name_matches", None)
                st.session_state.pop("planner_results", None)
                st.rerun()
        return

    user_code = st.session_state.planner_user_code
    user_name = st.session_state.planner_user_name
    first_name = str(user_name).strip().split()[0] if str(user_name).strip() else "there"
    st.markdown(f"## Hello {first_name}!")
    if st.button("Change user", key="change_planner_user"):
        st.session_state.planner_user_code = None
        st.session_state.planner_user_name = None
        st.session_state.pop("planner_results", None)
        st.session_state.pop("planner_hotel_result", None)
        st.session_state.pop("planner_hotel_destination", None)
        st.rerun()

    st.subheader("Plan your trip")
    user_history = history[history["userCode"] == str(user_code)]
    if user_history.empty:
        st.caption("No previous journeys were found for this user. Recommendations will rely more on general popularity.")

    origins = sorted(history["origin"].dropna().astype(str).unique().tolist())
    with st.form("destination_planner_form"):
        col1, col2 = st.columns(2)
        with col1:
            origin = st.selectbox("Current location / departure city", origins)
        with col2:
            departure_date = st.date_input(
                "Departure date",
                value=pd.Timestamp.today().date(),
                min_value=pd.Timestamp.today().date(),
            )
        model_choice = st.selectbox(
            "Recommendation model",
            ["Personalized baseline", "Candidate Random Forest", "Seasonal personalized"],
        )
        submitted = st.form_submit_button("Recommend destinations", type="primary", use_container_width=True)

    if submitted:
        candidates = sorted(history["destination"].dropna().astype(str).unique().tolist())
        try:
            if model_choice == "Personalized baseline":
                recommendations = baseline_recommendations(user_code, origin, history, candidates)
            elif model_choice == "Candidate Random Forest":
                recommendations = random_forest_recommendations(
                    user_code, origin, departure_date, history, candidates
                )
            else:
                recommendations = seasonal_recommendations(
                    user_code, origin, departure_date.month, history, candidates
                )
            st.session_state.planner_results = recommendations
            st.session_state.pop("planner_hotel_result", None)
            st.session_state.pop("planner_hotel_destination", None)
            st.session_state.planner_result_model = model_choice
            st.session_state.planner_result_inputs = {
                "origin": origin,
                "date": departure_date.strftime("%d %B %Y"),
            }
        except Exception as exc:
            st.error("Could not generate recommendations.")
            st.code(str(exc))

    if "planner_results" in st.session_state:
        recommendations = st.session_state.planner_results
        result_model = st.session_state.get("planner_result_model", "")
        result_inputs = st.session_state.get("planner_result_inputs", {})
        st.markdown("---")
        st.subheader("Recommended destinations")
        st.caption(
            f"Model: {result_model} · Departing from {result_inputs.get('origin', '')} · "
            f"Departure: {result_inputs.get('date', '')}"
        )
        display = recommendations[["Destination", "Score"]].copy()
        display.insert(0, "Rank", range(1, len(display) + 1))
        st.dataframe(display, hide_index=True, use_container_width=True)
        st.caption("Scores rank destinations; they are not probabilities.")
        st.bar_chart(recommendations.set_index("Destination")[["Score"]])

        st.markdown("#### Choose a destination")
        destination_options = recommendations["Destination"].astype(str).tolist()
        selected_destination = st.selectbox(
            "Destination for hotel recommendation",
            destination_options,
            key="planner_selected_destination",
        )
        if st.button("Find hotel", type="primary", key="find_hotel"):
            matching_hotels = hotels[hotels["place"].str.casefold() == selected_destination.casefold()].copy()
            if matching_hotels.empty:
                st.session_state.pop("planner_hotel_result", None)
                st.session_state.pop("planner_hotel_destination", None)
                st.warning("No hotel records were found for this destination.")
            else:
                hotel_summary = (
                    matching_hotels.groupby(["place", "name"], as_index=False)
                    .agg(
                        price_per_day=("price", "median"),
                        typical_stay_days=("days", "median"),
                        typical_total=("total", "median"),
                        bookings=("name", "size"),
                    )
                    .sort_values(["bookings", "name"], ascending=[False, True])
                )
                hotel_summary = hotel_summary.rename(columns={
                    "place": "Destination",
                    "name": "Hotel",
                    "price_per_day": "Median daily price",
                    "typical_stay_days": "Median stay (days)",
                    "typical_total": "Median booking total",
                    "bookings": "Historical bookings",
                })
                st.session_state.planner_hotel_result = hotel_summary
                st.session_state.planner_hotel_destination = selected_destination

        if (
            st.session_state.get("planner_hotel_destination") == selected_destination
            and "planner_hotel_result" in st.session_state
        ):
            st.markdown("#### Hotel recommendation")
            st.dataframe(
                st.session_state.planner_hotel_result,
                hide_index=True,
                use_container_width=True,
            )
            st.caption(
                "Hotel data contains one distinct hotel name per destination. "
                "This result retrieves the recorded hotel and summarizes historical booking prices; "
                "it does not rank alternative hotels."
            )



if app_page == "Destination Planner":
    st.sidebar.info(
        "Enter your name, then provide your departure city and date. "
        "The selected model ranks destinations from the available travel history."
    )
    render_destination_planner()
    st.stop()


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

    st.error("Model or encoder files could not be loaded.")

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
    '<div class="title">Voyage Analytics</div>',
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

st.sidebar.title("Voyage Analytics")

st.sidebar.markdown("### About Project")

st.sidebar.info(
    """
    Predict flight prices using a Random Forest regressor.

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
    '<div class="section-title">Flight Information</div>',
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
    '<div class="section-title">Travel Date</div>',
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
    "Predict Flight Price",
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
                <h2>Estimated Flight Price</h2>
                <div class="price">₹ {predicted_price:,.2f}</div>
            </div>
            """,
            unsafe_allow_html=True
        )
        


        # ----------------------------------------------------
        # SUCCESS MESSAGE
        # ----------------------------------------------------

        st.success(
            "Flight price prediction complete."
        )


        # ====================================================
        # DISPLAY INPUT DETAILS
        # ====================================================

        st.markdown(
            '<div class="section-title">'
            'Flight Details'
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
            'Model Information'
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
            "An error occurred while making the prediction."
        )

        st.exception(e)


# ============================================================
# MODEL FEATURES - OPTIONAL DEBUG INFORMATION
# ============================================================

with st.expander("View Model Features"): 

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

    <b>Voyage Analytics</b>

    <br>

    Flight Price Prediction using Machine Learning

    <br><br>

    Built with Python • Scikit-learn • Streamlit

    </center>
    """,
    unsafe_allow_html=True
)
