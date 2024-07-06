import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
from openpyxl import Workbook, load_workbook
import streamlit as st

# Load the training data
df = pd.read_excel(r"C:\Users\USER\Downloads\Training.xlsx")

# Split the data into features and target
x = df.iloc[:, :-1]
y = df["prognosis"]

# Train-test split
xtrain, xtest, ytrain, ytest = train_test_split(x, y, test_size=0.2)

# Train the model
model = DecisionTreeClassifier()
z = model.fit(xtrain, ytrain)

# Predict and calculate accuracy
ypred = z.predict(xtest)
accuracy = accuracy_score(ypred, ytest)

# Streamlit UI
st.title("Disease Prediction System")
st.write(f"Model Accuracy: {accuracy * 100:.2f}%")

# Display the symptoms list
st.write("Please provide at least five symptoms from the list below to get an accurate disease prediction:")

symptoms = list(xtrain.columns)
selected_symptoms = st.multiselect("Select Symptoms", symptoms)

if st.button("Predict Disease"):
    if len(selected_symptoms) < 5:
        st.warning("Please select at least 5 symptoms.")
    else:
        # Prepare user symptoms input
        p = [1 if symptom in selected_symptoms else 0 for symptom in symptoms]
        
        # Predict disease
        user_symptom_df = pd.DataFrame([p], columns=symptoms)
        disease = model.predict(user_symptom_df)
        st.write(f"The predicted disease is: {disease[0]}")

if st.button("Show Symptoms List"):
    st.write(symptoms)
