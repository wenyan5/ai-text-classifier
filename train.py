import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
import pickle

data = {
    "text": [
        "Win money now",
        "Call me later",
        "Limited offer just for you",
        "Let's have lunch tomorrow",
        "Claim your free prize now",
        "Are we meeting today?"
    ],
    "label": [1, 0, 1, 0, 1, 0]  # 1=spam, 0=normal
}

df = pd.DataFrame(data)

# 文本向量化
vectorizer = TfidfVectorizer()
X = vectorizer.fit_transform(df["text"])
y = df["label"]

# 模型训练
model = LogisticRegression()
model.fit(X, y)

# 保存模型
pickle.dump(model, open("model.pkl", "wb"))
pickle.dump(vectorizer, open("vectorizer.pkl", "wb"))

print("Model trained and saved.")
