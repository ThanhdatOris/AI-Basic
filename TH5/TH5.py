import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
import joblib
import os
import time
from datetime import datetime
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.utils import to_categorical

def get_unique_filename(directory, base_filename):
    counter = 1
    filename = f"{base_filename}.pkl"
    while os.path.exists(os.path.join(directory, filename)):
        filename = f"{base_filename}_{counter}.pkl"
        counter += 1
    return os.path.join(directory, filename)

# Bắt đầu đo thời gian chạy chương trình
start_time = time.time()

# Đọc dữ liệu từ file CSV
data = pd.read_csv('TH5/dataset.csv')

# Giả sử cột cuối cùng là nhãn (label) và các cột còn lại là đặc trưng (features)
X = data.iloc[:, :-1]  # Đặc trưng
y = data.iloc[:, -1]   # Nhãn

# Chia dữ liệu thành tập huấn luyện (70%) và tập kiểm tra (30%)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Kiểm tra kích thước dữ liệu
print(f"X_train shape: {X_train.shape}")
print(f"X_test shape: {X_test.shape}")

# Chuyển đổi nhãn thành dạng one-hot encoding
y_train_mlp = to_categorical(y_train)
y_test_mlp = to_categorical(y_test)

# Tạo thư mục model nếu chưa tồn tại
os.makedirs('TH5/model', exist_ok=True)

# Huấn luyện và đánh giá mô hình MLP
mlp_start_time = time.time()
mlp = Sequential([
    Dense(128, activation='relu', input_shape=(X_train.shape[1],)),
    Dense(64, activation='relu'),
    Dense(len(y.unique()), activation='softmax')
])
mlp.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
mlp.fit(X_train, y_train_mlp, epochs=10, batch_size=32, validation_split=0.2)
mlp_end_time = time.time()

mlp_loss, mlp_accuracy = mlp.evaluate(X_test, y_test_mlp)
print(f'Độ chính xác của mô hình MLP: {mlp_accuracy * 100:.2f}%')
print(f'Thời gian xử lý mô hình MLP: {mlp_end_time - mlp_start_time:.2f} giây')

# Lưu mô hình MLP vào file
mlp_model_path = get_unique_filename('TH5/model', 'mlp_model')
mlp.save(mlp_model_path)
print(f'Mô hình MLP đã được lưu vào file {mlp_model_path}')

# Huấn luyện mô hình SVM
svm_start_time = time.time()
svm = SVC(kernel='linear')
svm.fit(X_train, y_train)
svm_end_time = time.time()

# Dự đoán và đánh giá mô hình SVM
y_pred_svm = svm.predict(X_test)
accuracy_svm = accuracy_score(y_test, y_pred_svm)
print(f'Độ chính xác của mô hình SVM: {accuracy_svm * 100:.2f}%')
print(f'Thời gian xử lý mô hình SVM: {svm_end_time - svm_start_time:.2f} giây')

# Lưu mô hình SVM vào file
svm_model_path = get_unique_filename('TH5/model', 'svm_model')
joblib.dump(svm, svm_model_path)
print(f'Mô hình SVM đã được lưu vào file {svm_model_path}')

# Đọc mô hình từ file
loaded_mlp = tf.keras.models.load_model(mlp_model_path)
loaded_svm = joblib.load(svm_model_path)

# Dự đoán và đánh giá mô hình MLP đã được lưu
y_pred_loaded_mlp = loaded_mlp.predict(X_test)
y_pred_loaded_mlp = y_pred_loaded_mlp.argmax(axis=1)

# Dự đoán và đánh giá mô hình SVM đã được lưu
y_pred_loaded_svm = loaded_svm.predict(X_test)

# Đánh giá mô hình đã được lưu trên confusion matrix
print('MLP:')
print(pd.crosstab(y_test, y_pred_loaded_mlp, rownames=['Actual'], colnames=['Predicted']))
print()
print('SVM:')
print(pd.crosstab(y_test, y_pred_loaded_svm, rownames=['Actual'], colnames=['Predicted']))

# Kết thúc đo thời gian chạy chương trình
end_time = time.time()
total_time = end_time - start_time
print(f'Thời gian chạy chương trình: {total_time:.2f} giây')

# Hiển thị dấu thời gian sau khi kết thúc chương trình
print(f'Chương trình kết thúc vào: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}')