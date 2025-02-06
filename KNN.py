import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# 載入資料集
df = pd.read_csv('Friday-WorkingHours-Afternoon-DDos.pcap_ISCX.csv')

# 處理非數值型欄位並刪除不需要的欄位
df.columns = df.columns.str.strip()  # 移除所有欄位名稱的空格
df = df.drop(columns=['Flow ID', 'Source IP', 'Destination IP', 'Timestamp'])

# 將 ' Label' 欄位轉換為數值型
df['Label'] = df['Label'].apply(lambda x: 0 if x == 'BENIGN' else 1)

# 處理缺失值和無窮大值
df.replace([np.inf, -np.inf], np.nan, inplace=True)
df.fillna(0, inplace=True)

# 定義特徵和目標變量
X = df.drop(columns=['Label'])
y = df['Label']

# 分割訓練集和測試集
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 初始化 KNN 模型
knn_model = KNeighborsClassifier(n_neighbors=5)

# 訓練 KNN 模型
knn_model.fit(X_train, y_train)

# 在測試集上進行預測
y_pred = knn_model.predict(X_test)

# 評估模型
print("----- 測試資料集的評估 -----")
print("測試資料集準確度:", accuracy_score(y_test, y_pred))
print(classification_report(y_test, y_pred))

# 混淆矩陣繪製
cm = confusion_matrix(y_test, y_pred)
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=['BENIGN', 'DDoS'], yticklabels=['BENIGN', 'DDoS'])
plt.title('Confusion Matrix')
plt.xlabel('Predicted')
plt.ylabel('True')
plt.show()
