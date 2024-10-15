import tkinter as tk
from tkinter import ttk, messagebox
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, confusion_matrix
import joblib
import os
import time
from datetime import datetime
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.utils import to_categorical

class MLApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Machine Learning GUI")
        
        self.create_widgets()
        
    def create_widgets(self):
        self.menu = tk.Menu(self.root)
        self.root.config(menu=self.menu)
        
        self.model_menu = tk.Menu(self.menu, tearoff=0)
        self.menu.add_cascade(label="Models", menu=self.model_menu)
        self.model_menu.add_command(label="KNN", command=self.run_knn)
        self.model_menu.add_command(label="SVM", command=self.run_svm)
        self.model_menu.add_command(label="CNN", command=self.run_cnn)
        
        self.output_text = tk.Text(self.root, wrap=tk.WORD, height=20, width=80)
        self.output_text.pack(pady=10)
        
    def run_knn(self):
        self.output_text.delete(1.0, tk.END)
        self.output_text.insert(tk.END, "Running KNN...\n")
        
        data = pd.read_csv('TH5/dataset.csv')
        X = data.iloc[:, :-1]
        y = data.iloc[:, -1]
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
        
        knn = KNeighborsClassifier(n_neighbors=10)
        knn.fit(X_train, y_train)
        y_pred_knn = knn.predict(X_test)
        accuracy_knn = accuracy_score(y_test, y_pred_knn)
        
        self.output_text.insert(tk.END, f'Accuracy of KNN: {accuracy_knn * 100:.2f}%\n')
        self.plot_confusion_matrix(y_test, y_pred_knn, "KNN Confusion Matrix")
        
    def run_svm(self):
        self.output_text.delete(1.0, tk.END)
        self.output_text.insert(tk.END, "Running SVM...\n")
        
        data = pd.read_csv('TH5/dataset.csv')
        X = data.iloc[:, :-1]
        y = data.iloc[:, -1]
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
        
        svm = SVC(kernel='linear')
        svm.fit(X_train, y_train)
        y_pred_svm = svm.predict(X_test)
        accuracy_svm = accuracy_score(y_test, y_pred_svm)
        
        self.output_text.insert(tk.END, f'Accuracy of SVM: {accuracy_svm * 100:.2f}%\n')
        self.plot_confusion_matrix(y_test, y_pred_svm, "SVM Confusion Matrix")
        
    def run_cnn(self):
        self.output_text.delete(1.0, tk.END)
        self.output_text.insert(tk.END, "Running CNN...\n")
        
        data = pd.read_csv('TH5/dataset.csv')
        X = data.iloc[:, :-1]
        y = data.iloc[:, -1]
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
        
        y_train_mlp = to_categorical(y_train)
        y_test_mlp = to_categorical(y_test)
        
        mlp = Sequential([
            Dense(128, activation='relu', input_shape=(X_train.shape[1],)),
            Dense(64, activation='relu'),
            Dense(len(y.unique()), activation='softmax')
        ])
        mlp.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
        mlp.fit(X_train, y_train_mlp, epochs=10, batch_size=32, validation_split=0.2)
        
        mlp_loss, mlp_accuracy = mlp.evaluate(X_test, y_test_mlp)
        self.output_text.insert(tk.END, f'Accuracy of CNN: {mlp_accuracy * 100:.2f}%\n')
        
        y_pred_mlp = mlp.predict(X_test)
        y_pred_mlp = y_pred_mlp.argmax(axis=1)
        self.plot_confusion_matrix(y_test, y_pred_mlp, "CNN Confusion Matrix")
        
    def plot_confusion_matrix(self, y_true, y_pred, title):
        cm = confusion_matrix(y_true, y_pred)
        fig, ax = plt.subplots()
        cax = ax.matshow(cm, cmap=plt.cm.Blues)
        fig.colorbar(cax)
        plt.title(title)
        plt.xlabel('Predicted')
        plt.ylabel('Actual')
        plt.show()

if __name__ == "__main__":
    root = tk.Tk()
    app = MLApp(root)
    root.mainloop()