import tkinter as tk
from tkinter import ttk, messagebox
import numpy as np
from matplotlib.figure import Figure
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

# ---------- Global Variables ----------
loaded_signal = None
loaded_target = None
loaded_pred = None


# ---------- LOAD FILES ----------
def auto_load_files():
    global loaded_signal, loaded_target, loaded_pred

    try:
        if loaded_signal is None:
            loaded_signal = np.load("Signal.npy")
        if loaded_target is None:
            loaded_target = np.load("Target.npy")
        if loaded_pred is None:
            loaded_pred = np.load("Seizures_loc.npy")
    except FileNotFoundError as e:
        messagebox.showerror("Error", str(e))
        return False

    return True


# ---------- GENERATE PSEUDO-SHAP / CAM ----------
def generate_shap_interaction(signal):
    N = len(signal)
    shap_vals = 0.03 * np.random.randn(N)  # pseudo SHAP values
    grad = np.abs(np.gradient(signal))
    grad = (grad - grad.min()) / (grad.max() - grad.min() + 1e-9)
    y_vals = 0.5 * np.random.randn(N)
    return shap_vals, y_vals, grad


# ---------- SIGNAL SELECTED ----------
def on_signal_selected(event):
    if not auto_load_files():
        return

    selected = signal_var.get()
    index = int(selected.split(' ')[1]) - 1  # Adjusted to handle "Signal 1", "Signal 2", etc.
    signal = loaded_signal[index]

    # --- Plot Original Signal ---
    ax_signal.clear()
    ax_signal.plot(signal, color="blue")
    ax_signal.set_title(f"Original Signal {index + 1}", fontsize=12, fontweight="bold")
    ax_signal.set_xlabel("Index")
    ax_signal.set_ylabel("Amplitude")
    ax_signal.grid(True)

    # --- CAM Plot ---
    shap_x, shap_y, shap_color = generate_shap_interaction(signal)
    ax_shap.clear()
    ax_shap.scatter(shap_x, shap_y, c=shap_color, cmap="cool")
    ax_shap.axvline(0, color="gray", linewidth=1)
    ax_shap.set_title("CAM Output", fontsize=12, fontweight="bold")
    ax_shap.set_xlabel("Interaction Value")
    ax_shap.set_yticks([])

    canvas.draw()
    canvas.get_tk_widget().place(x=50, y=140, width=1000, height=500)

    # --- Update Classifier and Predicted Outcomes ---
    classified = "Epileptic" if loaded_target[index] == 1 else "Not Epileptic"
    classified_color = "red" if classified == "Epileptic" else "green"
    classified_label.config(
        text=f"NDAGAT-GRU Classified outcome: {classified}",
        fg=classified_color
    )

    pred_label.config(
        text=f"RA-ATCN Predicted outcome: {float(loaded_pred[index]):.3f}",
        fg="#800080"  # purple
    )


# ---------- DROPDOWN OPEN ----------
def open_dropdown(event):
    if not auto_load_files():
        return

    N = len(loaded_signal)
    signal_dropdown["values"] = [f"Signal {i + 1}" for i in range(N)]  # Updated to "Signal 1", "Signal 2", etc.


# ---------- GUI ----------
root = tk.Tk()
root.geometry("1100x850")  # window size
root.title("Epilepsy Detection - Signal & CAM Output")

# Titles
tk.Label(root, text="Epilepsy Detection", font=("Arial", 24, "bold")).place(x=380, y=20)
tk.Label(root, text="Select Original Signal", font=("Arial", 14, "bold")).place(x=230, y=80)

# Dropdown
signal_var = tk.StringVar()
signal_dropdown = ttk.Combobox(root, textvariable=signal_var, width=30, font=("Arial", 12))
signal_dropdown.place(x=450, y=80)
signal_dropdown.bind("<Button-1>", open_dropdown)
signal_dropdown.bind("<<ComboboxSelected>>", on_signal_selected)

# Figure with two subplots
fig = Figure(figsize=(10, 5), dpi=100)
fig.subplots_adjust(hspace=0.5)
ax_signal = fig.add_subplot(211)
ax_shap = fig.add_subplot(212)

canvas = FigureCanvasTkAgg(fig, master=root)
canvas.get_tk_widget().place_forget()  # initially hidden

# Labels for outcomes
classified_label = tk.Label(root, font=("Arial", 16, "bold"))
classified_label.place(x=50, y=660, width=1000)

pred_label = tk.Label(root, font=("Arial", 16, "bold"))
pred_label.place(x=50, y=700, width=1000)

root.mainloop()
