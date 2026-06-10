import streamlit as st
import pandas as pd
from sklearn.ensemble import IsolationForest

st.title("AI Log Anomaly Detection System")

st.write("Upload a log file and let AI identify suspicious log entries.")

uploaded_file = st.file_uploader(
    "Upload Log File",
    type=["txt", "log"]
)

if uploaded_file is not None:

    if st.button("Analyze Logs"):

        logs = []

        for line in uploaded_file:
            logs.append(line.decode("utf-8").strip())

        data = pd.DataFrame(logs, columns=["log"])

        data["is_error"] = data["log"].str.contains(
            "error",
            case=False,
            na=False
        )

        data["is_failed"] = data["log"].str.contains(
            "failed",
            case=False,
            na=False
        )

        data["is_error"] = data["is_error"].astype(int)
        data["is_failed"] = data["is_failed"].astype(int)

        data["suspicious_score"] = (
            data["is_error"] +
            data["is_failed"]
        )

        model = IsolationForest(
            contamination=0.3,
            random_state=42
        )

        model.fit(data[["suspicious_score"]])

        data["ai_result"] = model.predict(
            data[["suspicious_score"]]
        )

        total_logs = len(data)

        anomaly_count = len(
            data[data["ai_result"] == -1]
        )

        normal_logs = len(
            data[data["ai_result"] == 1]
        )

        st.success("Analysis Complete!")

        col1, col2, col3 = st.columns(3)

        with col1:
            st.metric("Total Logs", total_logs)

        with col2:
            st.metric("Normal Logs", normal_logs)

        with col3:
            st.metric("Anomalies Found", anomaly_count)

        chart_data = pd.DataFrame(
            {
                "Count": [
                    normal_logs,
                    anomaly_count
                ]
            },
            index=[
                "Normal",
                "Anomaly"
            ]
        )

        st.subheader("Analysis Summary")
        st.bar_chart(chart_data)

        st.subheader("All Log Analysis Results")
        st.dataframe(data)

        st.subheader("Detected Anomalies")

        anomalies = data[
            data["ai_result"] == -1
        ]

        if len(anomalies) > 0:
            st.dataframe(anomalies)
        else:
            st.info("No anomalies detected.")

else:
    st.info("Please upload a log file.")
