import os
import pandas as pd
import streamlit as st
import streamlit.components.v1 as components

# ============================================================
# CASEFILE - STABLE VERSION
# ============================================================

st.set_page_config(
    page_title="CASEFILE - AI Investigation",
    page_icon="🔎",
    layout="wide"
)

# ============================================================
# PATHS
# ============================================================

CASES_FILE = "data/synthetic/missing_person_cases_demo.csv"
ANOMALY_FILE = "data/processed/anomaly_gps_data.csv"
PRIORITY_FILE = "data/processed/search_priority.csv"
ROUTE_FILE = "data/processed/route_predictions.csv"

# ============================================================
# HELPERS
# ============================================================

def read_csv_safe(path):
    if not os.path.exists(path):
        return pd.DataFrame()

    try:
        return pd.read_csv(path)
    except Exception as e:
        st.error(f"Could not read {path}: {e}")
        return pd.DataFrame()


def get_col(df, names):
    if df.empty:
        return None

    for wanted in names:
        for col in df.columns:
            if str(col).strip().lower() == str(wanted).strip().lower():
                return col

    return None


def get_value(row, col, default="N/A"):
    if col is None:
        return default

    try:
        x = row[col]
        if pd.isna(x):
            return default
        return x
    except Exception:
        return default


def display_report(path, caption):
    if os.path.exists(path):
        st.image(path, caption=caption, use_container_width=True)
    else:
        st.info(f"Report not available: {path}")


# ============================================================
# SIDEBAR
# ============================================================

st.sidebar.title("🔎 CASEFILE")
st.sidebar.markdown("### AI Investigation System")
st.sidebar.markdown("---")

page = st.sidebar.radio(
    "Navigation",
    [
        "Dashboard",
        "Case Simulation",
        "Movement Analysis",
        "Anomaly Detection",
        "Location Prediction",
        "Route Prediction",
        "Search Priority",
        "Interactive Map"
    ]
)

st.sidebar.markdown("---")
st.sidebar.info(
    "Academic simulation using historical GPS data "
    "and fictional case information."
)

# ============================================================
# HEADER
# ============================================================

st.title("🔎 CASEFILE")
st.subheader("AI-Powered Missing Person Investigation Support System")

st.info(
    "Academic simulation using historical GPS movement data. "
    "All cases are fictional and outputs are probabilistic."
)

# ============================================================
# DASHBOARD
# ============================================================

if page == "Dashboard":

    st.header("🏠 Dashboard")

    cases = read_csv_safe(CASES_FILE)

    st.write(
        "An integrated AI/ML application for movement analysis, "
        "anomaly detection, probable-area estimation, route analysis, "
        "search-priority ranking and visualization."
    )

    a, b, c, d = st.columns(4)

    a.metric("Synthetic Cases", len(cases))
    b.metric("AI/ML Modules", "8")
    c.metric("Prediction", "Probabilistic")
    d.metric("Map", "Interactive")

    st.divider()

    st.header("🔄 Investigation Pipeline")

    st.markdown(
        """
        GPS Data → Preprocessing → Feature Engineering → K-Means
        → Isolation Forest → Location Prediction → Route Prediction
        → Search Priority → Explainable AI → Interactive Map
        """
    )

    st.divider()

    st.header("🧠 Technologies Used")

    x1, x2, x3, x4 = st.columns(4)
    x1.write("🔵 K-Means")
    x2.write("🚨 Isolation Forest")
    x3.write("🌲 Random Forest")
    x4.write("🛣️ Markov Chain")

# ============================================================
# CASE SIMULATION - LIGHTWEIGHT VERSION
# ============================================================

elif page == "Case Simulation":

    st.header("🔎 Final Case Simulation")

    st.write(
        "This page runs one fictional case through the integrated "
        "academic investigation-support workflow."
    )

    st.warning(
        "Academic simulation only. A probable area is an analytical "
        "estimate and is not a real-world location determination."
    )

    # --------------------------------------------------------
    # LOAD ONLY THE CASE FILE
    # --------------------------------------------------------

    cases = read_csv_safe(CASES_FILE)

    if cases.empty:
        st.error("❌ missing_person_cases.csv could not be loaded.")
        st.code(CASES_FILE)
        st.stop()

    case_id_col = get_col(
    cases,
    [
        "demo_case_id",
        "case_id",
        "original_case_id",
        "Case_ID",
        "caseID"
    ]
)
    

    if case_id_col is None:
        st.error("❌ case_id column was not found.")
        st.write("Available columns:", list(cases.columns))
        st.stop()

    # --------------------------------------------------------
    # IMPORTANT FIX:
    # DO NOT USE st.selectbox WITH ALL CASE IDs.
    # A huge list can freeze the browser.
    # --------------------------------------------------------

    total_cases = len(cases)

    st.subheader("📁 Select Fictional Case")

    case_number = st.number_input(
        f"Case Number (1 - {total_cases})",
        min_value=1,
        max_value=total_cases,
        value=1,
        step=1
    )

    row_index = int(case_number) - 1
    case = cases.iloc[row_index]

    selected_case_id = str(
        get_value(case, case_id_col)
    )

    st.caption(
        f"Showing case {case_number} of {total_cases}: "
        f"{selected_case_id}"
    )

    # --------------------------------------------------------
    # FIND FIELDS
    # --------------------------------------------------------

    person_col = get_col(cases, ["person_id", "Person_ID", "personID"])
    age_col = get_col(cases, ["age_group", "Age_Group"])
    time_col = get_col(cases, ["last_seen_time", "Last_Seen_Time"])

    lat_col = get_col(
        cases,
        ["last_latitude", "latitude", "lat"]
    )

    lon_col = get_col(
        cases,
        ["last_longitude", "longitude", "lon"]
    )

    previous_col = get_col(
        cases,
        ["previous_area", "Previous_Area"]
    )

    usual_col = get_col(
        cases,
        ["usual_area", "Usual_Area"]
    )

    target_col = get_col(
        cases,
        ["target_area", "Target_Area"]
    )

    hour_col = get_col(cases, ["hour", "Hour"])

    weather_col = get_col(
        cases,
        ["weather", "Weather"]
    )

    speed_col = get_col(
        cases,
        ["average_speed", "Average_Speed"]
    )

    distance_col = get_col(
        cases,
        ["average_distance", "Average_Distance"]
    )

    time_since_col = get_col(
        cases,
        ["time_since_last_seen", "Time_Since_Last_Seen"]
    )

    # --------------------------------------------------------
    # CASE INFORMATION
    # --------------------------------------------------------

    st.subheader("👤 Case Information")

    c1, c2, c3, c4 = st.columns(4)

    c1.metric(
        "Case ID",
        selected_case_id
    )

    c2.metric(
        "Person ID",
        str(get_value(case, person_col, "Synthetic"))
    )

    c3.metric(
        "Age Group",
        str(get_value(case, age_col))
    )

    c4.metric(
        "Last Seen",
        str(get_value(case, time_col))
    )

    # --------------------------------------------------------
    # MOVEMENT DETAILS
    # --------------------------------------------------------

    st.subheader("📋 Case Movement Details")

    details = []

    for title, col in [
        ("Last Latitude", lat_col),
        ("Last Longitude", lon_col),
        ("Previous Area", previous_col),
        ("Usual Area", usual_col),
        ("Target Area", target_col),
        ("Hour", hour_col),
        ("Weather", weather_col),
        ("Average Speed", speed_col),
        ("Average Distance", distance_col),
        ("Time Since Last Seen", time_since_col)
    ]:
        if col is not None:
            details.append(
                {
                    "Feature": title,
                    "Value": str(get_value(case, col))
                }
            )

    if details:
        st.dataframe(
            pd.DataFrame(details),
            use_container_width=True,
            hide_index=True
        )

    st.divider()

    # --------------------------------------------------------
    # RUN BUTTON
    # --------------------------------------------------------

    run = st.button(
        "🚀 Run AI Investigation",
        type="primary",
        use_container_width=True
    )

    if run:

        st.header("🧠 Investigation Results")

        # ----------------------------------------------------
        # PROBABLE AREA
        # ----------------------------------------------------

        probable_area = str(
            get_value(
                case,
                target_col,
                get_value(case, usual_col, "Unknown")
            )
        )

        previous_area = str(
            get_value(
                case,
                previous_col,
                "Unknown"
            )
        )

        st.subheader("📍 Probable Location")

        p1, p2 = st.columns(2)

        p1.metric(
            "Probable Area",
            probable_area
        )

        p2.metric(
            "Prediction Type",
            "Probabilistic"
        )

        # ----------------------------------------------------
        # SEARCH PRIORITY
        # ----------------------------------------------------

        st.subheader("🏆 Search Priority")

        priority = read_csv_safe(PRIORITY_FILE)

        area_col = get_col(
            priority,
            ["area", "cluster", "area_name"]
        )

        score_col = get_col(
            priority,
            [
                "priority_score",
                "score",
                "search_priority"
            ]
        )

        level_col = get_col(
            priority,
            [
                "priority_level",
                "priority",
                "level"
            ]
        )

        if (
            not priority.empty
            and area_col is not None
        ):

            matching = priority[
                priority[area_col].astype(str)
                == probable_area
            ]

            if not matching.empty:

                q1, q2 = st.columns(2)

                if score_col is not None:
                    q1.metric(
                        "Priority Score",
                        str(
                            round(
                                float(
                                    matching.iloc[0][score_col]
                                ),
                                2
                            )
                        )
                    )

                if level_col is not None:
                    q2.metric(
                        "Priority Level",
                        str(
                            matching.iloc[0][level_col]
                        )
                    )

                st.dataframe(
                    matching,
                    use_container_width=True,
                    hide_index=True
                )

            else:
                st.info(
                    "No priority record was found for "
                    "the probable area."
                )

        else:
            st.info(
                "Search-priority data is not available."
            )

        # ----------------------------------------------------
        # ROUTE
        # ----------------------------------------------------

        st.subheader("🛣️ Probable Route")

        route = read_csv_safe(ROUTE_FILE)

        current_col = get_col(
            route,
            [
                "current_area",
                "from_area",
                "source_area"
            ]
        )

        next_col = get_col(
            route,
            [
                "next_area",
                "to_area",
                "target_area"
            ]
        )

        probability_col = get_col(
            route,
            [
                "transition_probability",
                "probability",
                "transition_prob",
                "route_probability"
            ]
        )

        next_area = None

        if (
            not route.empty
            and current_col is not None
            and next_col is not None
        ):

            matches = route[
                route[current_col].astype(str)
                == previous_area
            ].copy()

            if not matches.empty:

                if probability_col is not None:
                    matches = matches.sort_values(
                        probability_col,
                        ascending=False
                    )

                next_area = str(
                    matches.iloc[0][next_col]
                )

        if next_area and next_area != probable_area:

            route_text = (
                previous_area
                + " → "
                + next_area
                + " → "
                + probable_area
            )

        else:

            route_text = (
                previous_area
                + " → "
                + probable_area
            )

        st.success(route_text)

        # ----------------------------------------------------
        # ANOMALIES
        # ----------------------------------------------------

        st.subheader("🚨 Historical Movement Anomalies")

        anomaly = read_csv_safe(ANOMALY_FILE)

        if (
            not anomaly.empty
            and "anomaly_label" in anomaly.columns
        ):

            anomaly_count = int(
                (
                    anomaly["anomaly_label"] == -1
                ).sum()
            )

            normal_count = int(
                (
                    anomaly["anomaly_label"] == 1
                ).sum()
            )

            a1, a2 = st.columns(2)

            a1.metric(
                "Anomalous Records",
                f"{anomaly_count:,}"
            )

            a2.metric(
                "Normal Records",
                f"{normal_count:,}"
            )

        else:

            st.info(
                "Anomaly data is not available."
            )

        st.caption(
            "Anomaly means an unusual historical movement pattern. "
            "It does not indicate criminal or suspicious behaviour."
        )

        # ----------------------------------------------------
        # EXPLANATION
        # ----------------------------------------------------

        st.subheader("💡 Prediction Explanation")

        explanation = []

        for title, col in [
            ("Previous Area", previous_col),
            ("Usual Area", usual_col),
            ("Average Speed", speed_col),
            ("Average Distance", distance_col),
            ("Movement Hour", hour_col),
            ("Weather", weather_col),
            ("Time Since Last Seen", time_since_col)
        ]:
            if col is not None:
                explanation.append(
                    {
                        "Factor": title,
                        "Value": str(
                            get_value(case, col)
                        )
                    }
                )

        if explanation:
            st.dataframe(
                pd.DataFrame(explanation),
                use_container_width=True,
                hide_index=True
            )

        importance_csv = (
            "reports/feature_importance.csv"
        )

        importance_png = (
            "reports/feature_importance.png"
        )

        if os.path.exists(importance_csv):

            try:
                importance = pd.read_csv(
                    importance_csv
                )

                st.markdown(
                    "#### 📊 Model Feature Importance"
                )

                st.dataframe(
                    importance.head(10),
                    use_container_width=True,
                    hide_index=True
                )

            except Exception:
                pass

        if os.path.exists(importance_png):
            st.image(
                importance_png,
                caption="Feature Importance",
                use_container_width=True
            )

        # ----------------------------------------------------
        # FINAL SUMMARY
        # ----------------------------------------------------

        st.divider()

        st.subheader("📄 Final Investigation Summary")

        st.info(
            f"""
**Case ID:** {selected_case_id}

**Probable Area:** {probable_area}

**Previous Area:** {previous_area}

**Probable Route:** {route_text}

This is a probabilistic academic simulation based on
historical movement patterns and fictional case information.
"""
        )

# ============================================================
# MOVEMENT ANALYSIS
# ============================================================

elif page == "Movement Analysis":

    st.header("📊 Movement Analysis")

    display_report(
        "reports/speed_distribution.png",
        "Speed Distribution"
    )

    display_report(
        "reports/movement_by_hour.png",
        "Movement Frequency by Hour"
    )

    display_report(
        "reports/movement_by_weekday.png",
        "Movement by Weekday"
    )

    display_report(
        "reports/gps_movement_pattern.png",
        "Historical GPS Movement"
    )

# ============================================================
# ANOMALY DETECTION
# ============================================================

elif page == "Anomaly Detection":

    st.header("🚨 Anomaly Detection")

    anomaly = read_csv_safe(ANOMALY_FILE)

    if (
        not anomaly.empty
        and "anomaly_label" in anomaly.columns
    ):

        normal = int(
            (anomaly["anomaly_label"] == 1).sum()
        )

        unusual = int(
            (anomaly["anomaly_label"] == -1).sum()
        )

        c1, c2 = st.columns(2)

        c1.metric(
            "Normal Records",
            f"{normal:,}"
        )

        c2.metric(
            "Anomaly Records",
            f"{unusual:,}"
        )

    display_report(
        "reports/anomaly_detection.png",
        "Isolation Forest Anomaly Detection"
    )

    st.info(
        "Anomalies represent unusual historical movement patterns "
        "and do not imply criminal or suspicious activity."
    )

# ============================================================
# LOCATION PREDICTION
# ============================================================

elif page == "Location Prediction":

    st.header("📍 Location Prediction")

    comparison = (
        "reports/model_comparison.csv"
    )

    if os.path.exists(comparison):

        try:
            df = pd.read_csv(comparison)

            st.subheader("🤖 Model Comparison")

            st.dataframe(
                df,
                use_container_width=True,
                hide_index=True
            )

        except Exception as e:
            st.error(str(e))

    display_report(
        "reports/location_prediction_confusion_matrix.png",
        "Location Prediction Confusion Matrix"
    )

    display_report(
        "reports/feature_importance.png",
        "Feature Importance"
    )

    if os.path.exists(
        "reports/feature_importance.csv"
    ):

        try:
            df = pd.read_csv(
                "reports/feature_importance.csv"
            )

            st.dataframe(
                df,
                use_container_width=True,
                hide_index=True
            )

        except Exception:
            pass

# ============================================================
# ROUTE PREDICTION
# ============================================================

elif page == "Route Prediction":

    st.header("🛣️ Route Prediction")

    route = read_csv_safe(ROUTE_FILE)

    if route.empty:

        st.warning(
            "Route prediction data not found."
        )

    else:

        st.dataframe(
            route.head(100),
            use_container_width=True,
            hide_index=True
        )

        current_col = get_col(
            route,
            [
                "current_area",
                "from_area",
                "source_area"
            ]
        )

        next_col = get_col(
            route,
            [
                "next_area",
                "to_area",
                "target_area"
            ]
        )

        probability_col = get_col(
            route,
            [
                "transition_probability",
                "probability",
                "transition_prob",
                "route_probability"
            ]
        )

        if (
            current_col is not None
            and next_col is not None
        ):

            areas = (
                route[current_col]
                .dropna()
                .astype(str)
                .unique()
                .tolist()
            )

            selected = st.selectbox(
                "Select Current Area",
                areas
            )

            matches = route[
                route[current_col].astype(str)
                == selected
            ].copy()

            if not matches.empty:

                if probability_col is not None:
                    matches = matches.sort_values(
                        probability_col,
                        ascending=False
                    )

                st.success(
                    "Most likely next area: "
                    + str(
                        matches.iloc[0][next_col]
                    )
                )

# ============================================================
# SEARCH PRIORITY
# ============================================================

elif page == "Search Priority":

    st.header("⭐ Search Priority")

    priority = read_csv_safe(PRIORITY_FILE)

    if priority.empty:

        st.warning(
            "Search priority data not found."
        )

    else:

        score_col = get_col(
            priority,
            [
                "priority_score",
                "score",
                "search_priority"
            ]
        )

        area_col = get_col(
            priority,
            [
                "area",
                "cluster",
                "area_name"
            ]
        )

        if score_col is not None:
            priority = priority.sort_values(
                score_col,
                ascending=False
            )

        st.dataframe(
            priority,
            use_container_width=True,
            hide_index=True
        )

        if (
            score_col is not None
            and area_col is not None
        ):

            chart = (
                priority[
                    [area_col, score_col]
                ]
                .dropna()
                .set_index(area_col)[score_col]
            )

            st.bar_chart(chart)

    st.info(
        "Priority ranking is an academic analytical output "
        "and not a real-world search decision."
    )

# ============================================================
# INTERACTIVE MAP
# ============================================================

elif page == "Interactive Map":

    st.header("🗺️ Interactive Investigation Map")

    map_file = (
        "reports/movement_investigation_map.html"
    )

    if not os.path.exists(map_file):

        st.error(
            "Interactive map file not found."
        )

    else:

        try:

            with open(
                map_file,
                "r",
                encoding="utf-8"
            ) as f:

                html = f.read()

            components.html(
                html,
                height=700,
                scrolling=True
            )

        except Exception as e:

            st.error(str(e))

# ============================================================
# FOOTER
# ============================================================

st.markdown("---")

st.caption(
    "CASEFILE | Academic AI/ML Investigation Simulation | "
    "Probabilistic outputs only"
)
