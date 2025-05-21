# 💡 Healthcare Cost Prediction App

This project is a machine learning-powered Streamlit app that predicts individual medical insurance costs based on features like age, gender, BMI, children, smoking status, and region.

 🚀 Demo

To run the app locally:

bash
streamlit run healthcare_app.py


🧠 About the Model
The app uses a Linear Regression model trained on the insurance.csv dataset. The model was evaluated using Mean Absolute Error (MAE) to ensure that predictions are reasonably accurate (MAE under $3500).

📊 Features
Upload and view the dataset
-Preprocessing with Label Encoding for categorical features
-Train/Test data splitting (80/20)
-Visualize predictions vs. actual costs
-Display Mean Absolute Error (MAE)
-Interactive Streamlit UI


📦 Requirements
Python 3.7+
Streamlit
Pandas
Scikit-learn
Matplotlib
