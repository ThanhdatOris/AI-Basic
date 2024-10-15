import tkinter as tk
from tkinter import ttk
from tkinter import messagebox
import random
from PIL import Image, ImageTk

class ColorApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Color Suggestion GUI")
        self.root.geometry("600x400")
        self.root.configure(bg="#f0f0f0")
        
        self.history = []
        
        self.create_widgets()
        
    def create_widgets(self):
        self.style = ttk.Style()
        self.style.configure("TButton", padding=6, relief="flat", background="#ccc")
        
        self.generate_button = ttk.Button(self.root, text="Generate Random Color", command=self.generate_color)
        self.generate_button.pack(pady=20)
        
        self.color_label = tk.Label(self.root, text="Color Code: #FFFFFF", bg="#f0f0f0", font=("Arial", 12))
        self.color_label.pack(pady=5)
        
        self.gradient_label = tk.Label(self.root, text="Gradient Code: linear-gradient(to right, #FFFFFF, #FFFFFF)", bg="#f0f0f0", font=("Arial", 12))
        self.gradient_label.pack(pady=5)
        
        self.canvas = tk.Canvas(self.root, width=300, height=100, bg="#FFFFFF", bd=0, highlightthickness=0)
        self.canvas.pack(pady=20)
        
        self.history_label = tk.Label(self.root, text="History", bg="#f0f0f0", font=("Arial", 12))
        self.history_label.pack(pady=5)
        
        self.history_listbox = tk.Listbox(self.root, width=50, height=10)
        self.history_listbox.pack(pady=10)
        
    def generate_color(self):
        color = "#{:06x}".format(random.randint(0, 0xFFFFFF))
        gradient = f"linear-gradient(to right, {color}, {self.lighten_color(color)})"
        
        self.color_label.config(text=f"Color Code: {color}")
        self.gradient_label.config(text=f"Gradient Code: {gradient}")
        
        self.update_canvas(color)
        self.save_history(color, gradient)
        
    def lighten_color(self, color, factor=0.5):
        r = int(color[1:3], 16)
        g = int(color[3:5], 16)
        b = int(color[5:7], 16)
        
        r = int(r + (255 - r) * factor)
        g = int(g + (255 - g) * factor)
        b = int(b + (255 - b) * factor)
        
        return "#{:02x}{:02x}{:02x}".format(r, g, b)
    
    def update_canvas(self, color):
        self.canvas.delete("all")
        self.canvas.create_rectangle(0, 0, 150, 100, fill=color, outline=color)
        self.canvas.create_rectangle(150, 0, 300, 100, fill=self.lighten_color(color), outline=self.lighten_color(color))
        
    def save_history(self, color, gradient):
        self.history.append((color, gradient))
        self.history_listbox.insert(tk.END, f"{color} | {gradient}")
        
if __name__ == "__main__":
    root = tk.Tk()
    app = ColorApp(root)
    root.mainloop()