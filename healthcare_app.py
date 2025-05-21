import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import mean_absolute_error

# Set page title
st.title("Healthcare Cost Prediction (Regression Model)")


# Load data
df = pd.read_csv('insurance.csv')
st.subheader("Raw Data")
st.dataframe(df.head())

# Encode categorical columns
data = df.copy()
categorical_cols = ['sex', 'smoker', 'region']
le = LabelEncoder()
for col in categorical_cols:
    data[col] = le.fit_transform(data[col])

# Features and target
X = data.drop('charges', axis=1)
y = data['charges']

# Train-test split
train_X, test_X, train_y, test_y = train_test_split(X, y, test_size=0.2, random_state=42)

# Train model
model = LinearRegression()
model.fit(train_X, train_y)

# Make predictions
predictions = model.predict(test_X)
mae = mean_absolute_error(test_y, predictions)
st.write(f"**Mean Absolute Error:** ${mae:.2f}")

st.write(f"**Mean Absolute Error:** ${mae:.2f}")

# Plot Actual vs Predicted
st.subheader("Actual vs Predicted Charges")

fig, ax = plt.subplots(figsize=(8,6))
ax.scatter(test_y, predictions, alpha=0.5, color='teal')
ax.plot([test_y.min(), test_y.max()], [test_y.min(), test_y.max()], 'r--')
ax.set_xlabel('Actual Charges')
ax.set_ylabel('Predicted Charges')
ax.set_title('Actual vs Predicted Charges')
ax.grid(True)

st.pyplot(fig)


st.subheader("💡 Predict Charges for a New Person")

# User input fields
age = st.slider("Age", 18, 100, 30)
sex = st.selectbox("Sex", ['male', 'female'])
bmi = st.number_input("BMI", 10.0, 50.0, 25.0)
children = st.number_input("Number of Children", 0, 10, 0)
smoker = st.selectbox("Smoker", ['yes', 'no'])
region = st.selectbox("Region", ['southwest', 'southeast', 'northwest', 'northeast'])

# Encode input values (must match training encoding)
sex_encoded = le.fit(df['sex']).transform([sex])[0]
smoker_encoded = le.fit(df['smoker']).transform([smoker])[0]
region_encoded = le.fit(df['region']).transform([region])[0]

# Create input data for prediction
input_data = pd.DataFrame([[
    age, sex_encoded, bmi, children, smoker_encoded, region_encoded
]], columns=['age', 'sex', 'bmi', 'children', 'smoker', 'region'])

# Predict button
if st.button("Predict Charges"):
    prediction = model.predict(input_data)[0]
    st.success(f"Estimated Medical Charges: ${prediction:.2f}")


if st.button("❌ Exit App"):
    st.warning("App stopped.")
    st.stop()  # Halts further execution


st.markdown("## 📊 Dataset Overview")

# Show full dataset (optional)
if st.checkbox("Show raw data"):
    st.dataframe(df)

# Summary stats for numeric columns
st.subheader("📈 Summary Statistics")
st.dataframe(df.describe())

# Value counts for categorical columns
st.subheader("🔢 Category Distributions")

for col in ['sex', 'smoker', 'region']:
    st.write(f"**{col.capitalize()}**")
    st.bar_chart(df[col].value_counts())


st.subheader("💥 BMI vs Charges")
fig1, ax1 = plt.subplots()
scatter = ax1.scatter(df['bmi'], df['charges'], c=df['age'], cmap='cool', alpha=0.6)
ax1.set_xlabel("BMI")
ax1.set_ylabel("Charges")
ax1.set_title("BMI vs Charges Colored by Age")
st.pyplot(fig1)
 

st.subheader("🚬 Average Charges by Smoking Status")

smoker_group = df.groupby('smoker')['charges'].mean().sort_values()
st.bar_chart(smoker_group)


