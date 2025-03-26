import streamlit as st
import requests

# Replace with your actual Flask API URL deployed on Render or another server
API_URL = "https://selflearningdiagnostic2.onrender.com"  

st.title("🩺 AI-Powered Self-Learning Medical Assistant")
st.subheader("Enter your symptoms to get a prediction")

# Input field for symptoms
symptoms_input = st.text_input("Enter symptoms (separated by spaces)", "")

# Initialize session state for storing prediction
if "predicted_disease" not in st.session_state:
    st.session_state.predicted_disease = None

# Button to get a prediction
if st.button("Predict Disease"):
    if symptoms_input.strip():
        try:
            # Send a POST request to /predict endpoint
            response = requests.post(f"{API_URL}/predict", json={"symptoms": symptoms_input})
            if response.status_code == 200:
                result = response.json()
                predicted_disease = result.get("predicted_disease", "Unknown")
                st.session_state.predicted_disease = predicted_disease  # Store prediction in session state
                st.success(f"Predicted Disease: *{predicted_disease}*")
            else:
                st.error(f"API Error: {response.status_code} - {response.text}")
        except Exception as e:
            st.error(f"Request failed: {e}")
    else:
        st.warning("Please enter symptoms before clicking Predict Disease.")

# If a prediction was made, ask for feedback
if st.session_state.predicted_disease:
    st.subheader("Is the prediction correct?")
    feedback_choice = st.radio("Please select:", ("Yes", "No"), index=None)

    if feedback_choice == "No":
        # Input field for the correct disease name
        correct_disease = st.text_input("Enter the correct disease name", "")

        # Button to submit feedback
        if st.button("Submit Feedback"):
            if correct_disease.strip():
                try:
                    # Send feedback along with the original symptoms
                    feedback_response = requests.post(
                        f"{API_URL}/update",
                        json={"symptoms": symptoms_input, "correct_disease": correct_disease}
                    )
                    if feedback_response.status_code == 200:
                        st.success("Thank you! Your feedback has been recorded.")
                        # Reset prediction state after feedback submission
                        st.session_state.predicted_disease = None
                    else:
                        st.error(f"Feedback submission failed: {feedback_response.text}")
                except Exception as e:
                    st.error(f"Feedback request failed: {e}")
            else:
                st.warning("Please enter the correct disease name before submitting feedback.")
