import tkinter as tk
from tkinter import scrolledtext
import sys
import io

def run_code():
    code = code_area.get('1.0', tk.END)
    old_stdout = sys.stdout
    old_stderr = sys.stderr
    sys.stdout = io.StringIO()
    sys.stderr = io.StringIO()
    
    try:
        exec(code)
        output = sys.stdout.getvalue()
        error = sys.stderr.getvalue()
    except Exception as e:
        output = ''
        error = str(e)
    
    sys.stdout = old_stdout
    sys.stderr = old_stderr
    
    output_area.config(state=tk.NORMAL)
    output_area.delete('1.0', tk.END)
    output_area.insert(tk.END, output + error)
    output_area.config(state=tk.DISABLED)

# Create main window
root = tk.Tk()
root.title("Python Learning GUI")

# Create and place widgets
instruction_label = tk.Label(root, text="Write your Python code below and click 'Run Code' to execute:")
instruction_label.pack(pady=5)

code_area = scrolledtext.ScrolledText(root, width=80, height=20)
code_area.pack(pady=5)

run_button = tk.Button(root, text="Run Code", command=run_code)
run_button.pack(pady=5)

output_label = tk.Label(root, text="Output:")
output_label.pack(pady=5)

output_area = scrolledtext.ScrolledText(root, width=80, height=10, state=tk.DISABLED)
output_area.pack(pady=5)

# Run the main loop
root.mainloop()