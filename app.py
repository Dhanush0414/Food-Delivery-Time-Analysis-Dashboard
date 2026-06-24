import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

st.title("🚚 Food Delivery Time Prediction App")

df = pd.read_csv("data/food_delivery_Times.csv")

st.subheader("Dataset Preview")
st.write(df.head())

st.subheader("Dataset Info")
st.write(df.shape)

st.subheader("Missing Values")
st.write(df.isnull().sum())

st.subheader("Statistical Summary")
st.write(df.describe())

st.subheader("Delivery Time Distribution")

if "Delivery_Time_min" in df.columns:
    fig, ax = plt.subplots()
    sns.histplot(df["Delivery_Time_min"], kde=True, ax=ax)
    st.pyplot(fig)

st.subheader("Correlation Matrix")

correlation = df.corr(numeric_only=True)

st.write(correlation)

st.subheader("Correlation Heatmap")

fig, ax = plt.subplots(figsize=(8,6))

sns.heatmap(correlation,
            annot=True,
            cmap="coolwarm",
            ax=ax)

st.pyplot(fig)

st.subheader("Dataset Insights")

st.write("Total Orders:", df.shape[0])

st.write("Average Delivery Time:",
         round(df["Delivery_Time_min"].mean(), 2),
         "minutes")

st.write("Maximum Delivery Time:",
         df["Delivery_Time_min"].max(),
         "minutes")

st.write("Minimum Delivery Time:",
         df["Delivery_Time_min"].min(),
         "minutes")