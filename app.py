import streamlit as st
import joblib

# Load model and vectorizer
model = joblib.load("sentiment_model.pkl")
tfidf = joblib.load("tfidf_vectorizer.pkl")

# Page configuration
st.set_page_config(
    page_title="Amazon Review Analyzer",
    page_icon="🛍️",
    layout="centered"
)

# Custom CSS
st.markdown("""
<style>
    .main-title {
        text-align: center;
        font-size: 40px;
        font-weight: bold;
    }

    .subtitle {
        text-align: center;
        font-size: 18px;
        margin-bottom: 30px;
    }
</style>
""", unsafe_allow_html=True)

# Header
st.markdown(
    '<div class="main-title">🛍️ Amazon Review Analyzer</div>',
    unsafe_allow_html=True
)

st.markdown(
    '<div class="subtitle">AI-powered sentiment analysis for customer reviews</div>',
    unsafe_allow_html=True
)

# Review input
review = st.text_area(
    "✍️ Enter your Amazon review",
    height=150,
    placeholder="Example: The product quality is excellent and I really loved it!"
)

# Prediction
if st.button("🔮 Analyze Sentiment", use_container_width=True):

    if not review.strip():
        st.warning("⚠️ Please enter a review first.")

    else:
        review_tfidf = tfidf.transform([review])
        prediction = model.predict(review_tfidf)[0]

        if prediction == 1:
            st.success("😊 Positive Review")
            st.write("The model predicts that this review has a **positive sentiment**.")
        else:
            st.error("😞 Negative Review")
            st.write("The model predicts that this review has a **negative sentiment**.")

# Footer
st.divider()

st.caption(
    "Machine Learning Project • TF-IDF + Logistic Regression"
)