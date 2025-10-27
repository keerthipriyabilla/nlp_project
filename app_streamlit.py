# app_streamlit.py
# ==============================================
# 🧠 Streamlit App for Sentiment Prediction using BiLSTM + Attention

import streamlit as st
import numpy as np
import tensorflow as tf
import joblib
import re
import nltk
from nltk.corpus import stopwords
from tensorflow.keras.preprocessing.sequence import pad_sequences

# --------------------------------
# 🔹 NLTK Setup
# --------------------------------
nltk.download('stopwords', quiet=True)
stop_words = set(stopwords.words('english'))

# --------------------------------
# 🔹 Load Model & Artifacts
# --------------------------------
class AttentionLayer(tf.keras.layers.Layer):
    def __init__(self, **kwargs):
        super(AttentionLayer, self).__init__(**kwargs)

    def build(self, input_shape):
        self.W = self.add_weight(shape=(input_shape[-1], input_shape[-1]),
                                 initializer="glorot_uniform", trainable=True)
        self.b = self.add_weight(shape=(input_shape[-1],),
                                 initializer="zeros", trainable=True)
        self.u = self.add_weight(shape=(input_shape[-1], 1),
                                 initializer="glorot_uniform", trainable=True)
        super(AttentionLayer, self).build(input_shape)

    def call(self, inputs):
        u_it = tf.tanh(tf.tensordot(inputs, self.W, axes=1) + self.b)
        a_it = tf.nn.softmax(tf.tensordot(u_it, self.u, axes=1), axis=1)
        return tf.reduce_sum(inputs * a_it, axis=1)

@st.cache_resource
def load_model_artifacts():
    model = tf.keras.models.load_model(
        "bilstm_attention_model.h5",
        custom_objects={"AttentionLayer": AttentionLayer},  # 👈 important
        compile=False
    )
    tokenizer = joblib.load("tokenizer.joblib")
    label_encoder = joblib.load("label_encoder.joblib")
    return model, tokenizer, label_encoder

model, tokenizer, label_encoder = load_model_artifacts()

# --------------------------------
# 🔹 Text Cleaning Function
# --------------------------------
def clean_text(text):
    text = str(text).lower()
    text = re.sub(r'http\S+', '', text)
    text = re.sub(r'[^a-z\s]', '', text)
    text = " ".join([w for w in text.split() if w not in stop_words])
    return text

# --------------------------------
# 🌈 Streamlit Page Config
# --------------------------------
st.set_page_config(
    page_title="Mental Health Sentiment Analyzer 🧠",
    page_icon="🧩",
    layout="centered",
    initial_sidebar_state="expanded"
)

# --------------------------------
# 🎨 Sidebar
# --------------------------------
with st.sidebar:
    st.title("🧠 Mental Health Analyzer")
    st.markdown(
        """
        This app predicts the **mental health sentiment** of user posts using a BiLSTM + Attention model trained on a custom dataset.
        """
    )
    st.info("Model: BiLSTM + Attention (GloVe 100d)")
    st.markdown("---")
    st.caption("Developed with ❤️ using Streamlit + TensorFlow")

# --------------------------------
# 🏠 Main Interface
# --------------------------------
st.title("💬 Mental Health Post Analyzer")
st.write("Enter your thoughts or a social media post to analyze its emotional tone.")

# Input area
user_input = st.text_area("📝 Enter your text below:", height=150, placeholder="Type something like: I feel so lost and tired lately...")

# Prediction Button
if st.button("🔍 Analyze Sentiment"):
    if user_input.strip() == "":
        st.warning("Please enter some text for analysis.")
    else:
        # Preprocess
        clean = clean_text(user_input)
        seq = tokenizer.texts_to_sequences([clean])
        padded = pad_sequences(seq, maxlen=100, padding='post', truncating='post')

        # Predict
        pred = model.predict(padded)
        pred_class = np.argmax(pred, axis=1)[0]
        sentiment = label_encoder.inverse_transform([pred_class])[0]
        confidence = np.max(pred) * 100

        # Display Result
        st.success(f"### 🧭 Predicted Sentiment: **{sentiment.upper()}**")
        st.progress(float(confidence / 100))
        st.write(f"**Confidence:** {confidence:.2f}%")

        # Optional: Detailed probability bar chart
        st.markdown("#### 📊 Prediction Confidence per Class")
        probs = pred.flatten()
        prob_dict = {label_encoder.classes_[i]: probs[i] for i in range(len(probs))}
        st.bar_chart(prob_dict)

# --------------------------------
# 🎯 Footer
# --------------------------------
st.markdown("---")
st.markdown(
    "<div style='text-align:center; color:grey;'>© 2025 Mental Health Sentiment Analyzer | Built with Streamlit</div>",
    unsafe_allow_html=True
)
