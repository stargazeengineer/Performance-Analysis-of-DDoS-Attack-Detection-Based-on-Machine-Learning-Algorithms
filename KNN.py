import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# 步驟 1: 載入資料集
df = pd.read_csv('C:/Users/Jason/OneDrive/桌面/專題/DDoS_Detect_Using_ML/Friday-WorkingHours-Afternoon-DDos.pcap_ISCX.csv')

# 步驟 2 & 3: 處理非數值型欄位並刪除不需要的欄位
df.columns = df.columns.str.strip()  # 移除所有欄位名稱的空格
# 轉換 Timestamp 欄位
df['Timestamp'] = pd.to_datetime(df['Timestamp'])
df['Hour'] = df['Timestamp'].dt.hour
df['Minute'] = df['Timestamp'].dt.minute
df['Second'] = df['Timestamp'].dt.second
df['Seconds_Since_Start'] = (df['Timestamp'] - df['Timestamp'].min()).dt.total_seconds()
df = df.drop(columns=['Timestamp'])

df = df.drop(columns=['Flow ID', 'Source IP', 'Destination IP', 'CWE Flag Count','Bwd Avg Bytes/Bulk','Bwd Avg Bulk Rate',
                      'Bwd Avg Packets/Bulk','Fwd Avg Bulk Rate','Bwd URG Flags','Fwd Avg Bytes/Bulk','Fwd Avg Packets/Bulk','Fwd URG Flags','Bwd PSH Flags',
                      'ECE Flag Count','RST Flag Count','Fwd IAT Mean','Fwd IAT Min', 'Fwd IAT Min', 'Source Port',
                      'Max Packet Length', 'Bwd IAT Total', 'Subflow Bwd Bytes', 'Flow Duration', 'Packet Length Std', 'Bwd Packets/s', 'URG Flag Count',
                      'Fwd Packet Length Min', 'Down/Up Ratio', 'Flow IAT Mean', 'Packet Length Mean', 'Bwd IAT Mean', 'Flow IAT Max', 'Fwd Packets/s',
                      'Fwd IAT Max', 'Packet Length Variance', 'Flow Bytes/s', 'min_seg_size_forward', 'Fwd Packet Length Std', 'Min Packet Length',
                      'Bwd IAT Min', 'Flow Packets/s', 'Bwd Packet Length Min', 'ACK Flag Count', 'Flow IAT Min', 'FIN Flag Count', 'PSH Flag Count',
                      'Idle Max', 'Idle Mean', 'Active Mean', 'Active Std', 'Bwd IAT Std', 'Idle Std', 'Active Min', 'Active Max', 'Protocol', 
                      'SYN Flag Count', 'Fwd PSH Flags' ])

# 步驟 4: 將 ' Label' 欄位轉換為數值型
df['Label'] = df['Label'].apply(lambda x: 0 if x == 'BENIGN' else 1)

# 步驟 5 & 6: 處理缺失值和無窮大值
df.replace([np.inf, -np.inf], np.nan, inplace=True)
df.fillna(0, inplace=True)

# 定義特徵和目標變量
X = df.drop(columns=['Label'])
y = df['Label']

# 步驟 7: 分割訓練集和測試集
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 步驟 8: 初始化 KNN 模型
knn_model = KNeighborsClassifier(n_neighbors=5)

# 步驟 9: 訓練 KNN 模型
knn_model.fit(X_train, y_train)

# 步驟 10: 在測試集上進行預測
y_pred = knn_model.predict(X_test)

# 步驟 11: 評估模型
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
