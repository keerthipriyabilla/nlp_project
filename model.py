# train_model.py
# ==============================================
# ✅ Import Dependencies
import pandas as pd
import numpy as np
import re
import os, joblib
import nltk
import tensorflow as tf
from nltk.corpus import stopwords
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from imblearn.over_sampling import RandomOverSampler
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.layers import Layer, Dense, Dropout, Embedding, Bidirectional, LSTM, Input
from tensorflow.keras.models import Model
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
import matplotlib.pyplot as plt
import seaborn as sns

# -------------------------------
# Step 1: Download NLTK Data
# -------------------------------
nltk.download('stopwords')
stop_words = set(stopwords.words('english'))

# -------------------------------
# Step 2: Load Dataset
# -------------------------------
df = pd.read_csv("Mental Health Dataset.csv")
print("✅ Dataset loaded successfully! Shape:", df.shape)

# -------------------------------
# Step 3: Clean Text
# -------------------------------
def clean_text(text):
    text = str(text).lower()
    text = re.sub(r'http\S+', '', text)
    text = re.sub(r'[^a-z\s]', '', text)
    text = " ".join([w for w in text.split() if w not in stop_words])
    return text

df['clean_text'] = df['posts'].apply(clean_text)

# -------------------------------
# Step 4: Encode Labels
# -------------------------------
le = LabelEncoder()
df['label'] = le.fit_transform(df['predicted'])
joblib.dump(le, "label_encoder.joblib")
print("✅ Label classes:", list(le.classes_))

# -------------------------------
# Step 5: Split Data
# -------------------------------
X_train, X_test, y_train, y_test = train_test_split(
    df['clean_text'], df['label'], test_size=0.1, random_state=42, stratify=df['label']
)

# -------------------------------
# Step 6: Tokenization & Padding
# -------------------------------
max_words = 10000
max_len = 100

tokenizer = Tokenizer(num_words=max_words, oov_token="<OOV>")
tokenizer.fit_on_texts(X_train)
joblib.dump(tokenizer, "tokenizer.joblib")

X_train_seq = tokenizer.texts_to_sequences(X_train)
X_test_seq = tokenizer.texts_to_sequences(X_test)
X_train_pad = pad_sequences(X_train_seq, maxlen=max_len, padding='post', truncating='post')
X_test_pad = pad_sequences(X_test_seq, maxlen=max_len, padding='post', truncating='post')

# -------------------------------
# Step 7: Handle Imbalance
# -------------------------------
ros = RandomOverSampler(random_state=42)
X_train_bal, y_train_bal = ros.fit_resample(X_train_pad, y_train)

# -------------------------------
# Step 8: Prepare Labels
# -------------------------------
num_classes = len(le.classes_)
y_train_cat = to_categorical(y_train_bal, num_classes)
y_test_cat = to_categorical(y_test, num_classes)

# -------------------------------
# Step 9: Load Local GloVe Embeddings
# -------------------------------
embedding_dim = 100
glove_dir = "glove.6B"
glove_file = os.path.join(glove_dir, "glove.6B.100d.txt")

if not os.path.exists(glove_file):
    raise FileNotFoundError(
        f"❌ GloVe file not found at {glove_file}. Please make sure it exists.\n"
        "Expected path: glove.6B/glove.6B.100d.txt"
    )

print("✅ Loading local GloVe embeddings...")

embedding_index = {}
with open(glove_file, encoding="utf8") as f:
    for line in f:
        values = line.split()
        word = values[0]
        coefs = np.asarray(values[1:], dtype="float32")
        embedding_index[word] = coefs

# ✅ Create Embedding Matrix
embedding_matrix = np.zeros((max_words, embedding_dim))
for word, i in tokenizer.word_index.items():
    if i < max_words:
        vec = embedding_index.get(word)
        if vec is not None:
            embedding_matrix[i] = vec

print("✅ Embedding matrix ready:", embedding_matrix.shape)

# -------------------------------
# Step 10: Define Attention Layer
# -------------------------------
class AttentionLayer(Layer):
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

# -------------------------------
# Step 11: Build Model
# -------------------------------
input_layer = Input(shape=(max_len,))
embedding_layer = Embedding(
    input_dim=max_words,
    output_dim=embedding_dim,
    weights=[embedding_matrix],
    trainable=False
)(input_layer)

x = Bidirectional(LSTM(128, return_sequences=True, dropout=0.3))(embedding_layer)
x = AttentionLayer()(x)
x = Dense(128, activation='relu')(x)
x = Dropout(0.4)(x)
output_layer = Dense(num_classes, activation='softmax')(x)

model = Model(inputs=input_layer, outputs=output_layer)
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
model.summary()

# -------------------------------
# Step 12: Train Model
# -------------------------------
early_stop = EarlyStopping(monitor='val_loss', patience=3, restore_best_weights=True)
lr_reduce = ReduceLROnPlateau(monitor='val_loss', factor=0.3, patience=2)

history = model.fit(
    X_train_bal, y_train_cat,
    epochs=10,
    batch_size=64,
    validation_split=0.1,
    callbacks=[early_stop, lr_reduce],
    verbose=1
)

# -------------------------------
# Step 13: Evaluate & Save
# -------------------------------
y_pred = np.argmax(model.predict(X_test_pad), axis=1)
print("\n✅ Test Accuracy:", round(accuracy_score(y_test, y_pred)*100, 2), "%")
print(classification_report(y_test, y_pred, target_names=le.classes_))

# ✅ Plot Training History
plt.figure(figsize=(10, 4))
plt.subplot(1, 2, 1)
plt.plot(history.history['accuracy'], label='Train Acc')
plt.plot(history.history['val_accuracy'], label='Val Acc')
plt.legend()
plt.title('Accuracy')

plt.subplot(1, 2, 2)
plt.plot(history.history['loss'], label='Train Loss')
plt.plot(history.history['val_loss'], label='Val Loss')
plt.legend()
plt.title('Loss')
plt.show()

# ✅ Confusion Matrix
cm = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=le.classes_,
            yticklabels=le.classes_)
plt.title("🧩 Confusion Matrix (Test Data)")
plt.xlabel("Predicted Labels")
plt.ylabel("True Labels")
plt.show()

# ✅ Save Model
model.save("bilstm_attention_model.h5")
print("\n✅ Model saved as 'bilstm_attention_model.h5'")

