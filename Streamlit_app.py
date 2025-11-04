# stream_samp.py
import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Step 1: Load processed dataset
st.title("🛍️ Women's Clothing E-Commerce Reviews Dashboard")

@st.cache_data
def load_data():
    return pd.read_csv("processed_feedback.csv")

df = load_data()

st.subheader("Sample of Processed Reviews")
st.write(df.head())

# Step 2: Sentiment distribution
st.subheader("Sentiment Distribution")
fig, ax = plt.subplots()
sns.countplot(data=df, x="sentiment_label", palette="coolwarm", ax=ax)
st.pyplot(fig)

# Step 3: Emotion distribution
if "emotion" in df.columns:
    st.subheader("Emotion Distribution")
    fig, ax = plt.subplots()
    sns.countplot(data=df, x="emotion", order=df["emotion"].value_counts().index, palette="Set2", ax=ax)
    plt.xticks(rotation=45)
    st.pyplot(fig)

# Step 4: Topic modeling
if "topic" in df.columns:
    st.subheader("Topic Distribution")
    fig, ax = plt.subplots(figsize=(10, 5))
    sns.countplot(data=df, x="topic", order=df["topic"].value_counts().index, palette="Paired", ax=ax)
    plt.xticks(rotation=90)
    st.pyplot(fig)

    # Show top topics
    st.write("### Top 10 Topics with Examples")
    for topic in df["topic"].value_counts().index[:10]:
        st.markdown(f"**Topic {topic}:**")
        sample_texts = df[df["topic"] == topic]["processed_text"].head(3).tolist()
        for txt in sample_texts:
            st.write(f"- {txt}")
