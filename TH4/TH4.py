import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
import joblib
import os
import time
from datetime import datetime
import matplotlib.pyplot as plt

def get_unique_filename(directory, base_filename):
    # Tạo tên file mới mà không ghi đè file cũ.
    counter = 1
    filename = f"{base_filename}.pkl"
    while os.path.exists(os.path.join(directory, filename)):
        filename = f"{base_filename}_{counter}.pkl"
        counter += 1
    return os.path.join(directory, filename)

# Bắt đầu đo thời gian chạy chương trình
start_time = time.time()

# Đọc dữ liệu từ file CSV
data = pd.read_csv('TH4/dataset.csv')

# Giả sử cột cuối cùng là nhãn (label) và các cột còn lại là đặc trưng (features)
X = data.iloc[:, :-1]  # Đặc trưng
y = data.iloc[:, -1]   # Nhãn

# Chia dữ liệu thành tập huấn luyện (70%) và tập kiểm tra (30%)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Break line
print()

# Tạo thư mục model nếu chưa tồn tại
os.makedirs('TH4/model', exist_ok=True)

# Huấn luyện và đánh giá mô hình KNN với các giá trị k từ 1 đến 10
knn_accuracies = []
knn_times = []

for k in range(1, 11):
    knn_start_time = time.time()
    knn = KNeighborsClassifier(n_neighbors=k)
    knn.fit(X_train, y_train)
    knn_end_time = time.time()

    y_pred_knn = knn.predict(X_test)
    accuracy_knn = accuracy_score(y_test, y_pred_knn)
    knn_accuracies.append(accuracy_knn)
    knn_times.append(knn_end_time - knn_start_time)

    print(f'Độ chính xác của mô hình KNN với k={k}: {accuracy_knn * 100:.2f}%')
    print(f'Thời gian xử lý mô hình KNN với k={k}: {knn_end_time - knn_start_time:.2f} giây')
    print()

# Break line
print()

# Lưu mô hình KNN với k=10 vào file
knn_model_path = get_unique_filename('TH4/model', 'knn_model')
joblib.dump(knn, knn_model_path)
print(f'Mô hình KNN với k = 10 đã được lưu vào file {knn_model_path}')

# Break line
print()

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
svm_model_path = get_unique_filename('TH4/model', 'svm_model')
joblib.dump(svm, svm_model_path)
print(f'Mô hình SVM đã được lưu vào file {svm_model_path}')

# Break line
print()

# Đọc mô hình từ file
loaded_knn = joblib.load(knn_model_path)
loaded_svm = joblib.load(svm_model_path)

# Dự đoán và đánh giá mô hình KNN đã được lưu
y_pred_loaded_knn = loaded_knn.predict(X_test)

# Dự đoán và đánh giá mô hình SVM đã được lưu
y_pred_loaded_svm = loaded_svm.predict(X_test)

# Đánh giá mô hình đã được lưu trên confusion matrix
print('KNN:')
print(pd.crosstab(y_test, y_pred_loaded_knn, rownames=['Actual'], colnames=['Predicted']))
print()
print('SVM:')
print(pd.crosstab(y_test, y_pred_loaded_svm, rownames=['Actual'], colnames=['Predicted']))

# Kết thúc đo thời gian chạy chương trình
end_time = time.time()
total_time = end_time - start_time
print(f'Thời gian chạy chương trình: {total_time:.2f} giây')

# Hiển thị dấu thời gian sau khi kết thúc chương trình
print(f'Chương trình kết thúc vào: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}')

# Vẽ biểu đồ độ chính xác và thời gian xử lý của KNN
plt.figure(figsize=(12, 6))

# Biểu đồ độ chính xác
plt.subplot(1, 2, 1)
plt.plot(range(1, 11), knn_accuracies, marker='o')
plt.title('Độ chính xác của KNN theo giá trị k')
plt.xlabel('k')
plt.ylabel('Độ chính xác')
plt.grid(True)

# Biểu đồ thời gian xử lý
plt.subplot(1, 2, 2)
plt.plot(range(1, 11), knn_times, marker='o', color='r')
plt.title('Thời gian xử lý của KNN theo giá trị k')
plt.xlabel('k')
plt.ylabel('Thời gian (giây)')
plt.grid(True)

# Hiển thị biểu đồ
plt.tight_layout()
plt.show()