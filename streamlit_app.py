# import streamlit as st
# import numpy as np
# import pandas as pd
# import joblib
# st.set_page_config(page_title="DDoS Detection", layout="centered")
# st.title("🚨 DDoS Attack Detection System")
# st.markdown("Upload CICDDoS test samples or manually input features to detect DDoS attacks.")
# # Load model and scaler
# @st.cache_resource
# def load_model_and_scaler():
#     model = joblib.load("xgboost_model.pkl")
#     scaler = joblib.load("scaler.pkl")
#     return model, scaler

# model, scaler = load_model_and_scaler()

# # Update this with your real 50 or 51 selected feature names
# # You can get them from your CSV's column names (excluding the label column)
# feature_names = [
#     'ACK Flag Count',
#     'Fwd Packet Length Mean',
#     'Avg Fwd Segment Size',
#     'Min Packet Length',
#     'Packet Length Mean',
#     'Fwd Packet Length Min',
#     'Average Packet Size',
#     'Protocol',
#     'Fwd Packet Length Max',
#     'Inbound',
#     'Flow Bytes/s',
#     'Unnamed: 0',
#     'Source Port',
#     'Max Packet Length',
#     'Fwd IAT Mean',
#     'Flow IAT Mean',
#     'Down/Up Ratio',
#     'Fwd Packets/s',
#     'Flow Packets/s',
#     'Flow IAT Std',
#     'Fwd IAT Std',
#     'URG Flag Count',
#     'act_data_pkt_fwd',
#     'Idle Max',
#     'Fwd IAT Max',
#     'Idle Mean',
#     'Flow IAT Max',
#     'Idle Min',
#     'Bwd Packet Length Min',
#     'Flow Duration',
#     'Fwd IAT Total',
#     'Idle Std',
#     'Init_Win_bytes_forward',
#     'CWE Flag Count',
#     'Bwd Packet Length Mean',
#     'Avg Bwd Segment Size',
#     'Subflow Fwd Bytes',
#     'Total Length of Fwd Packets',
#     'Fwd PSH Flags',
#     'RST Flag Count',
#     'Packet Length Std',
#     'Destination Port',
#     'Init_Win_bytes_backward',
#     'Fwd Packet Length Std',
#     'Bwd Packet Length Std',
#     'Bwd Packet Length Max',
#     'Bwd IAT Total',
#     'Bwd IAT Max',
#     'Bwd IAT Min',
#     'Bwd IAT Std'
# ] # Replace with actual names from your dataset

# # st.set_page_config(page_title="DDoS Detection", layout="centered")


# # --- File Upload Option ---
# st.subheader("📁 Upload Test Data CSV")
# uploaded_file = st.file_uploader("Upload a CSV file with selected features", type=["csv"])
# if uploaded_file is not None:
#     try:
#         df = pd.read_csv(uploaded_file)
#         st.write("📄 Uploaded Data Sample:")
#         st.dataframe(df.head())

#         if not all(feat in df.columns for feat in feature_names):
#             st.error("❌ Uploaded CSV must contain the correct feature columns.")
#         else:
#             X = df[feature_names]
#             X_scaled = scaler.transform(X)
#             preds = model.predict(X_scaled)
#             df["Prediction"] = ["DDoS 🚨" if p == 1 else "Benign ✅" for p in preds]

#             st.success("✅ Predictions complete!")
#             st.dataframe(df)
#     except Exception as e:
#         st.error(f"Error: {e}")

# # --- Manual Input Option ---
# st.markdown("---")
# st.subheader("📝 Manual Input")
# with st.form("manual_form"):
#     user_input = []
#     for i, feat in enumerate(feature_names):
#         val = st.number_input(f"{feat}:", key=feat)
#         user_input.append(val)
#     submitted = st.form_submit_button("Predict")

#     if submitted:
#         try:
#             scaled_input = scaler.transform([user_input])
#             result = model.predict(scaled_input)[0]
#             label = "DDoS 🚨" if result == 1 else "Benign ✅"
#             st.success(f"🧠 Prediction Result: **{label}**")
#         except Exception as e:
#             st.error(f"Error during prediction: {e}")

#######################################
# import streamlit as st
# import numpy as np
# import pandas as pd
# import joblib

# # ✅ MUST be first Streamlit command
# st.set_page_config(page_title="DDoS Detection", layout="centered")

# # --- Load models and scaler ---
# @st.cache_resource
# def load_models_and_scaler():
#     scaler = joblib.load("scaler.pkl")
    
#     # Load multiple models
#     models = {
#         "Decision Tree - ": joblib.load("decision_tree_model.pkl"),
#         "Random Forest - ": joblib.load("random_forest_model_binary_fixed.pkl"),
#         "XGBoost Classifier - ": joblib.load("xgboost_model.pkl"),
#         "ADABoost Classifier": joblib.load("adaboost_model.pkl"),
#         "K Nearest Neighbours": joblib.load("knn_model.pkl"),
#         "logistic regression classifier": joblib.load("logistic_regression_model.pkl"),
#         "Naive bayes classifier": joblib.load("naive_bayes_model.pkl"),
#         "SVM classifier":joblib.load("svm_model.pkl")
        
#         # Add more models here if you have
#     }
#     return models, scaler

# models, scaler = load_models_and_scaler()

# # Feature names (replace Unnamed: 0 if needed)
# feature_names = [
#     'ACK Flag Count', 'Fwd Packet Length Mean', 'Avg Fwd Segment Size', 'Min Packet Length', 
#     'Packet Length Mean', 'Fwd Packet Length Min', 'Average Packet Size', 'Protocol', 
#     'Fwd Packet Length Max', 'Inbound', 'Flow Bytes/s', 'Unnamed: 0', 'Source Port', 
#     'Max Packet Length', 'Fwd IAT Mean', 'Flow IAT Mean', 'Down/Up Ratio', 'Fwd Packets/s', 
#     'Flow Packets/s', 'Flow IAT Std', 'Fwd IAT Std', 'URG Flag Count', 'act_data_pkt_fwd', 
#     'Idle Max', 'Fwd IAT Max', 'Idle Mean', 'Flow IAT Max', 'Idle Min', 'Bwd Packet Length Min', 
#     'Flow Duration', 'Fwd IAT Total', 'Idle Std', 'Init_Win_bytes_forward', 'CWE Flag Count', 
#     'Bwd Packet Length Mean', 'Avg Bwd Segment Size', 'Subflow Fwd Bytes', 'Total Length of Fwd Packets', 
#     'Fwd PSH Flags', 'RST Flag Count', 'Packet Length Std', 'Destination Port', 
#     'Init_Win_bytes_backward', 'Fwd Packet Length Std', 'Bwd Packet Length Std', 
#     'Bwd Packet Length Max', 'Bwd IAT Total', 'Bwd IAT Max', 'Bwd IAT Min', 'Bwd IAT Std'
# ]

# st.title("🚨 DDoS Attack Detection System")
# st.markdown("Upload CICDDoS samples or manually input features to detect attacks.")

# # --- Choose model ---
# st.sidebar.header("⚙️ Settings")
# selected_model_name = st.sidebar.selectbox("Choose Model for Prediction", list(models.keys()))
# selected_model = models[selected_model_name]

# st.sidebar.success(f"Selected Model: {selected_model_name}")

# # --- File Upload Option ---
# st.subheader("📁 Upload Test Data CSV")
# uploaded_file = st.file_uploader("Upload a CSV file with selected features", type=["csv"])
# if uploaded_file is not None:
#     try:
#         df = pd.read_csv(uploaded_file)
#         st.write("📄 Uploaded Data Sample:")
#         st.dataframe(df.head())

#         if not all(feat in df.columns for feat in feature_names):
#             st.error("❌ Uploaded CSV must contain the correct feature columns.")
#         else:
#             if 'Unnamed: 0' in df.columns:
#                 df = df.drop(columns=['Unnamed: 0'])
#             X = df[feature_names if 'Unnamed: 0' not in feature_names else feature_names.remove('Unnamed: 0')]
#             X_scaled = scaler.transform(X)
#             preds = selected_model.predict(X_scaled)
#             df["Prediction"] = ["DDoS 🚨" if p == 1 else "Benign ✅" for p in preds]

#             st.success("✅ Predictions complete!")
#             st.dataframe(df)
#     except Exception as e:
#         st.error(f"Error: {e}")

# # --- Manual Input Option ---
# st.markdown("---")
# st.subheader("📝 Manual Input")
# with st.form("manual_form"):
#     user_input = []
#     for i, feat in enumerate(feature_names):
#         if feat != 'Unnamed: 0':
#             val = st.number_input(f"{feat}:", key=feat)
#             user_input.append(val)
#     submitted = st.form_submit_button("Predict")

# if submitted:
#     if user_input:  # make sure input exists
#         try:
#             scaled_input = scaler.transform([user_input])
#             result = selected_model.predict(scaled_input)[0]
#             label = "DDoS 🚨" if result == 1 else "Benign ✅"
#             st.success(f"🧠 Prediction Result using {selected_model_name}: **{label}**")
#         except Exception as e:
#             st.error(f"⚠️ Prediction failed due to input mismatch or model issue.")
#             st.info(f"Details: {e}")
#     else:
#         st.warning("⚠️ Please enter all feature values before predicting.")

# with st.form("manual_form"):
#     user_input = []
#     for i, feat in enumerate(feature_names):
#         if feat != 'Unnamed: 0':
#             val = st.number_input(f"{feat}:", key=feat)
#             user_input.append(val)
#     submitted = st.form_submit_button("Predict")

#     if submitted:
#         try:
#             scaled_input = scaler.transform([user_input])
#             result = selected_model.predict(scaled_input)[0]
#             label = "DDoS 🚨" if result == 1 else "Benign ✅"
#             st.success(f"🧠 Prediction Result using {selected_model_name}: **{label}**")
#         except Exception as e:
#             st.error(f"Error during prediction: {e}")
###########################################
import streamlit as st
import numpy as np
import pandas as pd
import joblib

# ✅ Must be first Streamlit command
st.set_page_config(page_title="DDoS Detection", layout="centered")

# --- Load models and scaler ---
@st.cache_resource
def load_models_and_scaler():
    scaler = joblib.load("scaler.pkl")

    models = {
        "Decision Tree - ": joblib.load("decision_tree_model.pkl"),
        "Random Forest - ": joblib.load("random_forest_model_binary_fixed.pkl"),
        "XGBoost Classifier - ": joblib.load("xgboost_model.pkl"),
        "ADABoost Classifier": joblib.load("adaboost_model.pkl"),
        "K Nearest Neighbours": joblib.load("knn_model.pkl"),
        "logistic regression classifier": joblib.load("logistic_regression_model.pkl"),
        "Naive bayes classifier": joblib.load("naive_bayes_model.pkl"),
        "SVM classifier":joblib.load("svm_model.pkl")
        
    }
    return models, scaler

models, scaler = load_models_and_scaler()

# ✅ List of 51 features (excluding label)
feature_names = [
    'ACK Flag Count', 'Fwd Packet Length Mean', 'Avg Fwd Segment Size', 'Min Packet Length',
    'Packet Length Mean', 'Fwd Packet Length Min', 'Average Packet Size', 'Protocol',
    'Fwd Packet Length Max', 'Inbound', 'Flow Bytes/s', 'Unnamed: 0', 'Source Port',
    'Max Packet Length', 'Fwd IAT Mean', 'Flow IAT Mean', 'Down/Up Ratio', 'Fwd Packets/s',
    'Flow Packets/s', 'Flow IAT Std', 'Fwd IAT Std', 'URG Flag Count', 'act_data_pkt_fwd',
    'Idle Max', 'Fwd IAT Max', 'Idle Mean', 'Flow IAT Max', 'Idle Min', 'Bwd Packet Length Min',
    'Flow Duration', 'Fwd IAT Total', 'Idle Std', 'Init_Win_bytes_forward', 'CWE Flag Count',
    'Bwd Packet Length Mean', 'Avg Bwd Segment Size', 'Subflow Fwd Bytes', 'Total Length of Fwd Packets',
    'Fwd PSH Flags', 'RST Flag Count', 'Packet Length Std', 'Destination Port',
    'Init_Win_bytes_backward', 'Fwd Packet Length Std', 'Bwd Packet Length Std',
    'Bwd Packet Length Max', 'Bwd IAT Total', 'Bwd IAT Max', 'Bwd IAT Min', 'Bwd IAT Std'
]

# ✅ Page Title
st.title("🚨 DDoS Attack Detection System")
st.markdown("Upload CICDDoS test samples or manually input features to detect attacks.")

# --- Sidebar Settings ---
st.sidebar.header("⚙️ Settings")
selected_model_name = st.sidebar.selectbox("Choose Model for Prediction", list(models.keys()))
selected_model = models[selected_model_name]
st.sidebar.success(f"Selected Model: {selected_model_name}")

# --- Upload CSV Option ---
st.subheader("📁 Upload Test Data CSV")


# uploaded_file = st.file_uploader("Upload a CSV file with selected features", type=["csv"])
# if uploaded_file is not None:
#     try:
#         df = pd.read_csv(uploaded_file)
#         st.write("📄 Uploaded Data Sample:")
#         st.dataframe(df.head())

#         # Clean the data if 'Unnamed: 0' exists
#         # if 'Unnamed: 0' in df.columns:
#         #     df = df.drop(columns=['Unnamed: 0'])

#         # Now, feature selection
#         selected_features = [feat for feat in feature_names if feat in df.columns]

#         if len(selected_features) == 0:
#             st.error("❌ Uploaded CSV does not have matching feature columns.")
#         else:
#             X = df[selected_features]

#             # Scale and predict
#             X_scaled = scaler.transform(X)
#             preds = selected_model.predict(X_scaled)
#             df["Prediction"] = ["DDoS 🚨" if p == 1 else "Benign ✅" for p in preds]

#             st.success("✅ Predictions complete!")
#             st.dataframe(df)
#     except Exception as e:
#         st.error("⚠️ Error during prediction on uploaded data.")
#         if e is not None:
#             st.info(f"Details: {e}")
uploaded_file = st.file_uploader("Upload a CSV file with selected features and true labels", type=["csv"])
if uploaded_file is not None:
    try:
        df = pd.read_csv(uploaded_file)
        st.write("📄 Uploaded Data Sample:")
        st.dataframe(df.head())

        # Assume 'Label' column has true values (0 or 1) for Benign / DDoS
        if 'Label' not in df.columns:
            st.error("❌ Uploaded CSV must contain a 'Label' column with true classes.")
        else:
            selected_features = feature_names

            # Check if all features are present
            missing_features = set(selected_features) - set(df.columns)
            if missing_features:
                st.error(f"❌ Uploaded CSV missing these feature columns: {missing_features}")
            else:
                X = df[selected_features]
                y_true = df['Label']

                # Scale and Predict
                X_scaled = scaler.transform(X)
                y_pred = selected_model.predict(X_scaled)

                # Show predictions
                df["Predicted Label"] = ["DDoS 🚨" if p == 1 else "Benign ✅" for p in y_pred]

                st.success("✅ Predictions complete!")
                st.dataframe(df)

                # --- Calculate Accuracy ---
                from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

                acc = accuracy_score(y_true, y_pred)
                st.subheader("📊 Model Evaluation on Uploaded Test Data")
                st.success(f"🎯 Accuracy: {acc * 100:.2f}%")

                # Optional: Show classification report
                with st.expander("See Detailed Classification Report"):
                    st.text(classification_report(y_true, y_pred))

                # Optional: Show confusion matrix
                with st.expander("See Confusion Matrix"):
                    cm = confusion_matrix(y_true, y_pred)
                    st.text(cm)

    except Exception as e:
        st.error("⚠️ Error during prediction or evaluation on uploaded data.")
        if e is not None:
            st.info(f"Details: {e}")



# --- Manual Feature Input Option ---
st.markdown("---")
st.subheader("📝 Manual Input")

with st.form("manual_form"):
    user_input = []
    for feat in feature_names:
        if feat != 'Unnamed: 0':  # Skip the unnecessary column
            val = st.number_input(f"{feat}:", key=feat)
            user_input.append(val)
    submitted = st.form_submit_button("Predict")

if submitted:
    if user_input:  # Ensure features are entered
        try:
            scaled_input = scaler.transform([user_input])
            result = selected_model.predict(scaled_input)[0]
            label = "DDoS 🚨" if result == 1 else "Benign ✅"
            st.success(f"🧠 Prediction Result using {selected_model_name}: **{label}**")
        except Exception as e:
            st.error("⚠️ Prediction failed. Please check the inputs and model compatibility.")
            st.info(f"Details: {e}")
    else:
        st.warning("⚠️ Please enter all feature values before predicting.")
