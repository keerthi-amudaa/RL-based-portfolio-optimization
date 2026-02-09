import os
import streamlit as st
import pandas as pd
import plotly.express as px
from dotenv import load_dotenv

from utils.data import ASSET_CATEGORIES
from utils.news import fetch_news
from utils.finbert import load_finbert, predict_sentiment

# -----------------------------
# LOAD ENV VARIABLES
# -----------------------------
load_dotenv()
news_api_key = os.getenv("NEWS_API_KEY")

st.set_page_config(layout="wide")
st.title("📰 Market News & AI Sentiment Analysis")

st.markdown(
"""
This page analyzes **real-time financial news** and applies **FinBERT**, a
finance-specific NLP model, to understand **market sentiment**.

📌 News sentiment often acts as a **leading indicator** for volatility.
"""
)

# -----------------------------
# VALIDATE API KEY
# -----------------------------
if not news_api_key:
    st.error(
        "❌ NewsAPI key not found. Please add `NEWS_API_KEY` to your `.env` file."
    )
    st.stop()

# -----------------------------
# ASSET SELECTION
# -----------------------------
category = st.selectbox("Asset Category", list(ASSET_CATEGORIES.keys()))
ticker = st.selectbox("Select Ticker", ASSET_CATEGORIES[category])

# -----------------------------
# LOAD FINBERT (CACHED)
# -----------------------------
@st.cache_resource
def load_model():
    return load_finbert()

tokenizer, model = load_model()

# -----------------------------
# FETCH NEWS
# -----------------------------
articles = fetch_news(ticker, news_api_key)

if not articles:
    st.warning("No recent news found for this ticker.")
    st.stop()

texts = [
    a["title"] + ". " + (a["description"] or "")
    for a in articles
]

sentiments, probs = predict_sentiment(texts, tokenizer, model)

# -----------------------------
# DISPLAY NEWS + SENTIMENT
# -----------------------------
news_df = pd.DataFrame({
    "Title": [a["title"] for a in articles],
    "Source": [a["source"]["name"] for a in articles],
    "Sentiment": sentiments,
    "Positive": probs[:, 2],
    "Neutral": probs[:, 1],
    "Negative": probs[:, 0],
    "URL": [a["url"] for a in articles]
})

st.markdown("## 🗞️ News Articles & AI Interpretation")

for _, row in news_df.iterrows():
    with st.expander(f"{row['Title']} ({row['Sentiment']})"):
        st.write(f"**Source:** {row['Source']}")
        st.write(f"🔗 [Read full article]({row['URL']})")
        st.progress(float(row["Positive"]))

        st.markdown(
            f"""
            **Sentiment Breakdown**
            - 🟢 Positive: {row['Positive']:.2f}
            - ⚪ Neutral: {row['Neutral']:.2f}
            - 🔴 Negative: {row['Negative']:.2f}
            """
        )

# -----------------------------
# AGGREGATED SENTIMENT
# -----------------------------
sentiment_counts = (
    news_df["Sentiment"]
    .value_counts()
    .reset_index()
)
sentiment_counts.columns = ["Sentiment", "Count"]

fig = px.bar(
    sentiment_counts,
    x="Sentiment",
    y="Count",
    title=f"📊 Overall News Sentiment for {ticker}",
    color="Sentiment"
)

st.plotly_chart(fig, use_container_width=True)

# -----------------------------
# EXPLANATION
# -----------------------------
st.info(
"""
### 📘 How to interpret this page

**FinBERT**
- A transformer trained specifically on financial text
- Understands earnings, guidance, regulation, and risk

**Insights you can draw**
- Mostly positive → bullish short-term sentiment
- Mixed → uncertainty or consolidation
- Negative → potential downside risk

⚠️ News sentiment should always be used **alongside technical and risk analysis**.
"""
)
