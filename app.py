import streamlit as st
import pandas as pd
import numpy as np
import requests
import joblib
import os
from pathlib import Path

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
    .main { background-color: #f5f7fb; }
    .title { font-size: 42px; font-weight: 700; color: #1f3c88; margin-bottom: 5px; }
    .subtitle { font-size: 18px; color: #6c757d; margin-bottom: 25px; }
    .prediction-box { padding: 25px; border-radius: 15px; background-color: #ffffff; 
        text-align: center; box-shadow: 0px 4px 15px rgba(0, 0, 0, 0.10); 
        margin-top: 20px; margin-bottom: 20px; }
    .price { font-size: 42px; font-weight: bold; color: #008000; }
    .section-title { font-size: 25px; font-weight: 600; color: #1f3c88; }
    .fallback-badge { background-color: #fff3cd; color: #856404; padding: 4px 8px; 
        border-radius: 4px; font-size: 12px; font-weight: 600; }
    .best-badge { background-color: #d4edda; color: #155724; padding: 4px 8px; 
        border-radius: 4px; font-size: 12px; font-weight: 600; }
    </style>
    """,
    unsafe_allow_html=True
)

# ============================================================
# CONFIGURATION
# ============================================================
BASE_DIR = Path(__file__).resolve().parent
API_BASE = os.getenv("API_BASE", "http://localhost:5001/api/v1")

# ============================================================
# API CLIENT
# ============================================================
def _http_error_detail(resp):
    try:
        detail = resp.json().get("error", resp.text)
    except Exception:
        detail = resp.text
    detail = str(detail).strip() or resp.reason
    return f"HTTP {resp.status_code}: {detail}"

def api_get(endpoint, fallback_fn=None):
    try:
        resp = requests.get(f"{API_BASE}{endpoint}", timeout=5)
        if not resp.ok:
            raise RuntimeError(_http_error_detail(resp))
        return resp.json(), None
    except Exception as e:
        if fallback_fn:
            return fallback_fn(), f"API unavailable: {e}. Using fallback."
        return None, f"API error: {e}"

def api_post(endpoint, payload, fallback_fn=None):
    try:
        resp = requests.post(f"{API_BASE}{endpoint}", json=payload, timeout=15)
        if not resp.ok:
            raise RuntimeError(_http_error_detail(resp))
        return resp.json(), None
    except Exception as e:
        if fallback_fn:
            return fallback_fn(payload), f"API unavailable: {e}. Using fallback."
        return None, f"API error: {e}"

def get_available_models():
    """Fetch available models from API."""
    result, error = api_get("/models")
    if error or not result:
        # Fallback model list
        return {
            "destination_models": [
                {"key": "logistic_regression_user", "name": "Logistic Regression (User Split) ⭐ BEST", "type": "ml", "split": "user_80_20", "algorithm": "logistic_regression"},
                {"key": "logistic_regression_global", "name": "Logistic Regression (Global Split)", "type": "ml", "split": "global_80_20", "algorithm": "logistic_regression"},
                {"key": "random_forest_user", "name": "Random Forest (User Split)", "type": "ml", "split": "user_80_20", "algorithm": "random_forest"},
                {"key": "random_forest_global", "name": "Random Forest (Global Split)", "type": "ml", "split": "global_80_20", "algorithm": "random_forest"},
                {"key": "gradient_boosting_user", "name": "Gradient Boosting (User Split)", "type": "ml", "split": "user_80_20", "algorithm": "gradient_boosting"},
                {"key": "gradient_boosting_global", "name": "Gradient Boosting (Global Split)", "type": "ml", "split": "global_80_20", "algorithm": "gradient_boosting"},
                {"key": "personalized_history", "name": "Personalized History Baseline", "type": "baseline", "split": "both", "algorithm": "heuristic"},
                {"key": "global_popularity", "name": "Global Popularity Baseline", "type": "baseline", "split": "both", "algorithm": "heuristic"},
            ],
            "hotel_models": [
                {"key": "linear_regression", "name": "Linear Regression ⭐ BEST", "type": "hotel", "algorithm": "linear_regression"},
                {"key": "random_forest", "name": "Random Forest", "type": "hotel", "algorithm": "random_forest"},
                {"key": "gradient_boosting", "name": "Gradient Boosting", "type": "hotel", "algorithm": "gradient_boosting"},
            ]
        }
    return result

# ============================================================
# LOCAL FALLBACK FUNCTIONS (copied from original app.py)
# ============================================================
def baseline_recommendations_local(user_code, origin, history, candidates, k=5):
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
    ranked.insert(0, "Rank", range(1, len(ranked) + 1))
    return ranked[["Rank", "Destination", "Route history", "User destination history", "Origin popularity", "Global popularity"]]

def global_popularity_recommendations_local(history, candidates, k=5):
    """Global popularity baseline."""
    global_counts = history["destination"].value_counts()
    top_dest = global_counts.head(k).index.tolist()
    recs = pd.DataFrame({
        "Rank": range(1, len(top_dest) + 1),
        "Destination": top_dest,
        "Global count": global_counts.head(k).values
    })
    return recs

def seasonal_recommendations_local(user_code, origin, departure_month, history, candidates, k=5):
    """Seasonal recommendations with manual weights (original)."""
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
        rows.append({"Destination": destination, "Score": float(score)})
    return pd.DataFrame(rows).sort_values(
        ["Score", "Destination"], ascending=[False, True]
    ).head(k).reset_index(drop=True)

# ============================================================
# LOAD PLANNER DATA (for local fallbacks)
# ============================================================
@st.cache_data
def load_planner_data():
    OUTPUT_DIR = BASE_DIR / "travel_recommender_outputs"
    users_path = BASE_DIR / "users.csv"
    journeys_path = OUTPUT_DIR / "journeys_train.csv"
    hotels_path = BASE_DIR / "hotels.csv"
    
    users = pd.read_csv(users_path)
    history = pd.read_csv(journeys_path)
    hotels = pd.read_csv(hotels_path)
    
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

# ============================================================
# APP NAVIGATION
# ============================================================
st.sidebar.markdown("---")
app_page = st.sidebar.radio(
    "Navigate",
    ["Flight Price Predictor", "Destination Planner"],
    key="app_page",
)

# ============================================================
# DESTINATION PLANNER (API-backed)
# ============================================================
def render_destination_planner():
    st.markdown('<div class="title">Destination Planner</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="subtitle">Personalized destination recommendations from your travel history</div>',
        unsafe_allow_html=True,
    )

    # Load data for local fallbacks
    try:
        users, history, hotels = load_planner_data()
    except Exception as exc:
        st.error("Destination planner data could not be loaded.")
        st.code(str(exc))
        return

    # Get available models from API
    models_data = get_available_models()
    dest_models = models_data.get("destination_models", [])
    hotel_models = models_data.get("hotel_models", [])

    # Session state
    if "planner_user_code" not in st.session_state:
        st.session_state.planner_user_code = None
    if "planner_user_name" not in st.session_state:
        st.session_state.planner_user_name = None

    # Login flow (unchanged)
    if st.session_state.planner_user_code is None:
        st.subheader("1. Log In")
        st.write("Enter your name or select an account below.")

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
                st.session_state.pop("planner_results", None)
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

    # User logged in
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
    candidates = sorted(history["destination"].dropna().astype(str).unique().tolist())

    # Model selection
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
        
        # Build model options
        ml_models = [m for m in dest_models if m["type"] == "ml"]
        baseline_models = [m for m in dest_models if m["type"] == "baseline"]
        
        model_options = []
        for m in ml_models:
            badge = " ⭐" if "BEST" in m["name"] else ""
            model_options.append((m["key"], f"{m['name']}{badge}"))
        model_options.append(("---", "--- Baselines ---"))
        for m in baseline_models:
            model_options.append((m["key"], m["name"]))
        
        model_choice = st.selectbox(
            "Recommendation model",
            options=[k for k, _ in model_options],
            format_func=lambda k: dict(model_options).get(k, k),
        )
        submitted = st.form_submit_button("Recommend destinations", type="primary", use_container_width=True)

    if submitted:
        try:
            payload = {
                "user_code": user_code,
                "origin": origin,
                "departure_date": departure_date.strftime("%Y-%m-%d"),
                "split": "user_80_20",
                "model": model_choice,
                "top_k": 5
            }
            
            def fallback_dest(payload):
                if payload["model"] == "personalized_history":
                    return baseline_recommendations_local(user_code, origin, history, candidates).to_dict("records")
                elif payload["model"] == "global_popularity":
                    return global_popularity_recommendations_local(history, candidates).to_dict("records")
                else:
                    return baseline_recommendations_local(user_code, origin, history, candidates).to_dict("records")
            
            result, error = api_post("/destinations/recommend", payload, fallback_dest)
            
            if error:
                st.warning(f"⚠️ {error}")
            
            if result and "recommendations" in result:
                st.session_state.planner_results = pd.DataFrame(result["recommendations"])
                st.session_state.pop("planner_hotel_result", None)
                st.session_state.pop("planner_hotel_destination", None)
                st.session_state.planner_result_model = result.get("model", model_choice)
                st.session_state.planner_result_type = result.get("type", "unknown")
                st.session_state.planner_result_inputs = {
                    "origin": origin,
                    "date": departure_date.strftime("%d %B %Y"),
                }
            else:
                st.error("Could not generate recommendations.")
        except Exception as exc:
            st.error("Could not generate recommendations.")
            st.code(str(exc))

    # Display results
    if "planner_results" in st.session_state:
        recommendations = st.session_state.planner_results
        result_model = st.session_state.get("planner_result_model", "")
        result_type = st.session_state.get("planner_result_type", "")
        result_inputs = st.session_state.get("planner_result_inputs", {})
        
        st.markdown("---")
        st.subheader("Recommended destinations")
        
        # Model badge
        badge_html = ""
        if "BEST" in result_model or "best" in result_model.lower():
            badge_html = ' <span class="best-badge">BEST</span>'
        elif result_type == "baseline":
            badge_html = ' <span class="fallback-badge">BASELINE</span>'
        elif "fallback" in str(result_type).lower():
            badge_html = ' <span class="fallback-badge">FALLBACK</span>'
        
        st.caption(
            f"Model: {result_model}{badge_html} · Departing from {result_inputs.get('origin', '')} · "
            f"Departure: {result_inputs.get('date', '')}"
        )
        
        # Ensure proper columns
        if "Destination" in recommendations.columns:
            display = recommendations[["Destination"]].copy()
            if "Score" in recommendations.columns:
                display["Score"] = recommendations["Score"]
            elif "score" in recommendations.columns:
                display["Score"] = recommendations["score"]
            else:
                display["Score"] = None
        elif "destination" in recommendations.columns:
            display = recommendations[["destination"]].copy()
            display.columns = ["Destination"]
            if "score" in recommendations.columns:
                display["Score"] = recommendations["score"]
            else:
                display["Score"] = None
        else:
            display = pd.DataFrame(recommendations)
            if "destination" in display.columns:
                display.columns = ["Destination"] if "Destination" not in display.columns else display.columns
        
        if "Rank" not in display.columns:
            display.insert(0, "Rank", range(1, len(display) + 1))
        
        st.dataframe(display, hide_index=True, use_container_width=True)
        
        if "Score" in display.columns:
            chart_data = display.set_index("Destination")[["Score"]].dropna()
            if not chart_data.empty:
                st.bar_chart(chart_data)
        
        st.caption("Scores rank destinations; they are not probabilities.")
        
        # Hotel selection
        st.markdown("#### Choose a destination for hotel recommendation")
        if "Destination" in display.columns:
            destination_options = display["Destination"].astype(str).tolist()
        elif "destination" in display.columns:
            destination_options = display["destination"].astype(str).tolist()
        else:
            destination_options = candidates
        
        selected_destination = st.selectbox(
            "Destination for hotel recommendation",
            destination_options,
            key="planner_selected_destination",
        )
        
        # Hotel model selection
        hotel_model_options = {m["key"]: m["name"] for m in hotel_models}
        selected_hotel_model = st.selectbox(
            "Hotel expenditure prediction model",
            options=list(hotel_model_options.keys()),
            format_func=lambda k: hotel_model_options[k],
            key="planner_hotel_model"
        )
        
        col1, col2 = st.columns([1, 1])

        # Recommend hotels within the selected destination
        with col1:
            if st.button("Recommend hotels", type="primary", key="recommend_hotel", use_container_width=True):
                payload = {
                    "user_code": user_code,
                    "destination": selected_destination,
                    "departure_date": departure_date.strftime("%Y-%m-%d"),
                    "top_k": 5,
                }
                result, error = api_post("/hotels/recommend", payload)
                if error:
                    st.warning(f"⚠️ {error}")
                else:
                    st.session_state.planner_hotel_result = result
                    st.session_state.planner_hotel_destination = selected_destination
        
        with col2:
            if st.button("Show historical hotel data only", key="show_historical_only", use_container_width=True):
                hist_result, _ = api_post("/hotels/historical", {"destination": selected_destination})
                if hist_result and "hotels" in hist_result:
                    st.session_state.planner_hotel_historical = pd.DataFrame(hist_result["hotels"])
                    st.session_state.planner_hotel_destination = selected_destination
                    st.session_state.pop("planner_hotel_result", None)

    # Display hotel results (side-by-side)
    if "planner_hotel_destination" in st.session_state:
        selected_destination = st.session_state.planner_hotel_destination

        st.markdown("---")
        st.subheader(f"Hotel Recommendation for {selected_destination}")

        col_ml, col_hist = st.columns(2)

        with col_ml:
            st.markdown("### 🏨 Recommended Hotels")
            if "planner_hotel_result" in st.session_state:
                hotel_res = st.session_state.planner_hotel_result
                hotels = hotel_res.get("hotels", [])
                message = hotel_res.get("message")
                if message:
                    st.info(message)
                if hotels:
                    table = pd.DataFrame([
                        {
                            "Rank": h["rank"],
                            "Hotel": h["hotel"],
                            "Score": f"{h['score']:.3f}",
                            "Median price/day": f"${h['median_price_per_day']:.2f}",
                            "Typical stay (days)": h["typical_stay_days"],
                            "Typical total": f"${h['typical_total']:.2f}",
                            "Bookings": h["bookings"],
                        }
                        for h in hotels
                    ])
                    st.dataframe(table, hide_index=True, use_container_width=True)
                    st.caption("Recommended hotels are constrained to the selected destination.")
                else:
                    st.info("No hotels available for this destination.")
            else:
                st.info("Click 'Recommend hotels' to see ML recommendations")

        with col_hist:
            st.markdown("### 📊 Historical Hotel Data")
            if "planner_hotel_historical" in st.session_state:
                hist_df = st.session_state.planner_hotel_historical
                st.dataframe(hist_df, hide_index=True, use_container_width=True)
            else:
                st.info("Historical data will appear here after clicking 'Show historical hotel data only'")

        st.caption(
            "Recommender scores hotels within the destination using user profile, "
            "trip history and hotel statistics. If no hotel data exists for the "
            "destination, no recommendations are shown."
        )


# ============================================================
# FLIGHT PRICE PREDICTOR (unchanged - local model)
# ============================================================
@st.cache_resource
def load_flight_model_files():
    model = joblib.load("flight_price_model.joblib")
    from_encoder = joblib.load("from_encoder.joblib")
    to_encoder = joblib.load("to_encoder.joblib")
    flightType_encoder = joblib.load("flightType_encoder.joblib")
    agency_encoder = joblib.load("agency_encoder.joblib")
    return model, from_encoder, to_encoder, flightType_encoder, agency_encoder

def render_flight_predictor():
    st.markdown(
        '<div class="title">Voyage Analytics</div>',
        unsafe_allow_html=True
    )
    st.markdown(
        '<div class="subtitle">Flight Price Prediction using Machine Learning</div>',
        unsafe_allow_html=True
    )

    # Load files
    try:
        model, from_encoder, to_encoder, flightType_encoder, agency_encoder = load_flight_model_files()
    except Exception as e:
        st.error("Model or encoder files could not be loaded.")
        st.write("Make sure these files are present:")
        st.write("1. flight_price_model.joblib")
        st.write("2. from_encoder.joblib")
        st.write("3. to_encoder.joblib")
        st.write("4. flightType_encoder.joblib")
        st.write("5. agency_encoder.joblib")
        st.exception(e)
        st.stop()

    try:
        model_features = list(model.feature_names_in_)
    except AttributeError:
        model_features = ["from", "to", "flightType", "time", "distance", "agency", "year", "month", "day", "dayofweek"]

    # Sidebar
    st.sidebar.title("Voyage Analytics")
    st.sidebar.markdown("### About Project")
    st.sidebar.info(
        """
        Predict flight prices using a Random Forest regressor.

        **Machine Learning Model:** Random Forest Regressor  
        **Problem Type:** Regression  
        **Target Variable:** Flight Price
        """
    )
    st.sidebar.markdown("---")
    st.sidebar.markdown("### How It Works")
    st.sidebar.write("1. Enter flight information")
    st.sidebar.write("2. Select travel date")
    st.sidebar.write("3. Click Predict Flight Price")
    st.sidebar.write("4. Get estimated price")

    # Flight Information
    st.markdown('<div class="section-title">Flight Information</div>', unsafe_allow_html=True)
    st.write("Enter the details of your flight.")

    col1, col2, col3 = st.columns(3)
    with col1:
        source = st.selectbox("From", options=list(from_encoder.classes_))
    with col2:
        destination = st.selectbox("To", options=list(to_encoder.classes_))
    with col3:
        flight_type = st.selectbox("Flight Type", options=list(flightType_encoder.classes_))

    col1, col2, col3 = st.columns(3)
    with col1:
        time = st.number_input("Travel Time", min_value=0.0, value=2.0, step=0.5)
    with col2:
        distance = st.number_input("Distance", min_value=0.0, value=500.0, step=10.0)
    with col3:
        agency = st.selectbox("Agency", options=list(agency_encoder.classes_))

    st.markdown('<div class="section-title">Travel Date</div>', unsafe_allow_html=True)
    st.write("Select the travel date.")
    col1, col2, col3 = st.columns(3)
    with col1:
        year = st.number_input("Year", min_value=2000, max_value=2100, value=2023, step=1)
    with col2:
        month = st.number_input("Month", min_value=1, max_value=12, value=5, step=1)
    with col3:
        day = st.number_input("Day", min_value=1, max_value=31, value=15, step=1)

    day_names = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
    day_name = st.selectbox("Day of Week", day_names)
    dayofweek = day_names.index(day_name)

    st.write("")
    predict_button = st.button("Predict Flight Price", type="primary", use_container_width=True)

    if predict_button:
        try:
            source_encoded = from_encoder.transform([source])[0]
            destination_encoded = to_encoder.transform([destination])[0]
            flight_type_encoded = flightType_encoder.transform([flight_type])[0]
            agency_encoded = agency_encoder.transform([agency])[0]

            input_data = pd.DataFrame({
                "from": [source_encoded], "to": [destination_encoded],
                "flightType": [flight_type_encoded], "time": [time],
                "distance": [distance], "agency": [agency_encoded],
                "year": [year], "month": [month], "day": [day], "dayofweek": [dayofweek]
            })
            input_data = input_data.reindex(columns=model_features)

            if input_data.isnull().any().any():
                missing_features = input_data.columns[input_data.isnull().any()].tolist()
                st.error("Some required model features are missing:")
                st.write(missing_features)
                st.stop()

            prediction = model.predict(input_data)
            predicted_price = prediction[0]

            st.markdown(
                f"""
                <div class="prediction-box">
                    <h2>Estimated Flight Price</h2>
                    <div class="price">₹ {predicted_price:,.2f}</div>
                </div>
                """, unsafe_allow_html=True
            )
            st.success("Flight price prediction complete.")

            st.markdown('<div class="section-title">Flight Details</div>', unsafe_allow_html=True)
            details = pd.DataFrame({
                "Feature": ["From", "To", "Flight Type", "Travel Time", "Distance", "Agency", "Year", "Month", "Day", "Day of Week"],
                "Value": [source, destination, flight_type, time, distance, agency, year, month, day, day_name]
            })
            st.table(details)

            st.markdown('<div class="section-title">Model Information</div>', unsafe_allow_html=True)
            col1, col2, col3 = st.columns(3)
            with col1: st.metric("Model", "Random Forest")
            with col2: st.metric("Problem", "Regression")
            with col3: st.metric("Target", "Flight Price")

        except Exception as e:
            st.error("An error occurred while making the prediction.")
            st.exception(e)

    with st.expander("View Model Features"):
        st.write("The saved model expects these features:")
        st.write(model_features)

    st.markdown("---")
    st.markdown(
        """
        <center>
        <b>Voyage Analytics</b><br>
        Flight Price Prediction using Machine Learning<br><br>
        Built with Python • Scikit-learn • Streamlit
        </center>
        """, unsafe_allow_html=True
    )


# ============================================================
# MAIN ROUTING
# ============================================================
if app_page == "Destination Planner":
    st.sidebar.info(
        "Enter your name, then provide your departure city and date. "
        "Select a model to rank destinations. Then choose a destination "
        "and hotel model to predict hotel expenditure."
    )
    render_destination_planner()
else:
    render_flight_predictor()