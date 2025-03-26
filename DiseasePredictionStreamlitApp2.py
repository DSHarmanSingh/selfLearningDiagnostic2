import streamlit as st
import requests

# Replace with your actual Flask API URL deployed on Render or your server
API_URL = "https://selflearningdiagnostic2.onrender.com" 

st.title("🩺 AI-Powered Self-Learning Medical Assistant")
st.subheader("Enter your symptoms to get a prediction")

# Input field for symptoms (user query)
symptoms_input = st.text_input("Enter symptoms (separated by spaces)", "")

# Button to get a prediction
if st.button("Predict Disease"):
    if symptoms_input.strip():
        # Send a POST request to /predict endpoint
        try:
            response = requests.post(f"{API_URL}/predict", json={"symptoms": symptoms_input})
            if response.status_code == 200:
                result = response.json()
                predicted_disease = result.get("predicted_disease", "Unknown")
                st.success(f"Predicted Disease: *{predicted_disease}*")
                
                # Feedback section (only if prediction is incorrect)
                st.subheader("Is the prediction correct?")
                feedback_choice = st.radio("Please select:", ("Yes", "No"))
                
                if feedback_choice == "No":
                    # Input field for the correct disease name
                    correct_disease = st.text_input("Enter the correct disease name", "")
                    if st.button("Submit Feedback"):
                        if correct_disease.strip():
                            # Send feedback along with the original symptoms
                            feedback_response = requests.post(
                                f"{API_URL}/update", 
                                json={"symptoms": symptoms_input, "correct_disease": correct_disease}
                            )
                            if feedback_response.status_code == 200:
                                st.success("Thank you! Your feedback has been recorded.")
                            else:
                                st.error(f"Feedback submission failed: {feedback_response.text}")
                        else:
                            st.warning("Please enter the correct disease name before submitting feedback.")
            else:
                st.error(f"API Error: {response.status_code} - {response.text}")
        except Exception as e:
            st.error(f"Request failed: {e}")
    else:
        st.warning("Please enter symptoms before clicking Predict Disease.")
