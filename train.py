import ssl
import certifi
import pickle
from sklearn.datasets import fetch_20newsgroups
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report


ssl._create_default_https_context = lambda: ssl.create_default_context(cafile=certifi.where())

# ----------------------
# 加载数据
# ----------------------
newsgroups = fetch_20newsgroups(subset='all')
X = newsgroups.data
y = newsgroups.target
labels = newsgroups.target_names

# ----------------------
# 划分训练/测试
# ----------------------
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# ----------------------
# 训练管道
# ----------------------
model = make_pipeline(TfidfVectorizer(stop_words='english'), LogisticRegression(max_iter=1000))
model.fit(X_train, y_train)

# ----------------------
# 评估
# ----------------------
print(classification_report(y_test, model.predict(X_test), target_names=labels))

# ----------------------
# 保存模型
# ----------------------
pickle.dump(model, open("text_classifier.pkl", "wb"))
print("Model trained and saved.")
