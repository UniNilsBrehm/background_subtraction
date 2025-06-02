import cv2
import random
import numpy as np
from tqdm import tqdm
import tkinter as tk
from tkinter import filedialog, ttk
import subprocess

# --- Contrast enhancement function ---
def enhance_contrast(gray_frame):
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    return clahe.apply(gray_frame)

# --- File selection ---
root = tk.Tk()
root.withdraw()
input_path = filedialog.askopenfilename(title="Select input video", filetypes=[("MP4 files", "*.mp4")])
if not input_path:
    print("No video selected.")
    exit()

cap = cv2.VideoCapture(input_path)
frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = cap.get(cv2.CAP_PROP_FPS)

# --- Frame preview and selection ---
selected_frames = []
current_frame = 0

def show_frame(frame_index):
    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_index)
    ret, frame = cap.read()
    if not ret:
        return
    display = cv2.resize(frame.copy(), (640, 480))
    text = f"Frame: {frame_index}"
    if frame_index in selected_frames:
        text += "Selected"
    cv2.putText(display, text, (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
    cv2.imshow("Frame Preview", display)

def on_slider_move(val):
    global current_frame
    current_frame = int(float(val))
    show_frame(current_frame)

def add_frame():
    if current_frame not in selected_frames:
        selected_frames.append(current_frame)
        label_selected.config(text="Selected: " + ', '.join(map(str, selected_frames)))

def on_key(event):
    global current_frame
    if event.keysym == 'Right':
        current_frame = min(current_frame + 1, frame_count - 1)
    elif event.keysym == 'Left':
        current_frame = max(current_frame - 1, 0)
    elif event.keysym == 'Return':
        add_frame()
    elif event.keysym == 'Escape':
        preview_window.quit()
    slider.set(current_frame)
    show_frame(current_frame)

# --- GUI window ---
preview_window = tk.Tk()
preview_window.title("Select Background Frames")
preview_window.bind('<Key>', on_key)

slider = ttk.Scale(preview_window, from_=0, to=frame_count - 1, orient=tk.HORIZONTAL, length=400, command=on_slider_move)
slider.pack(pady=10)

btn_add = tk.Button(preview_window, text="Add Frame to Background Set", command=add_frame)
btn_add.pack(pady=5)

label_selected = tk.Label(preview_window, text="Selected: None")
label_selected.pack(pady=5)

btn_done = tk.Button(preview_window, text="Done", command=preview_window.quit)
btn_done.pack(pady=10)

# Start preview with first frame
show_frame(current_frame)
preview_window.mainloop()
cv2.destroyAllWindows()

if not selected_frames:
    print("No frames selected.")
    cap.release()
    exit()

print(f"Selected background frames: {selected_frames}")

# --- Build background model ---
bg_frames = []
for idx in selected_frames:
    cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
    ret, frame = cap.read()
    if not ret:
        continue
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    gray = enhance_contrast(gray)
    bg_frames.append(gray.astype(np.float32))

if not bg_frames:
    print("Failed to load background frames.")
    cap.release()
    exit()

background = np.median(bg_frames, axis=0).astype(np.uint8)

# --- FFmpeg Streaming Setup ---
output_path = input_path.rsplit('.', 1)[0] + "_cleaned_compressed.mp4"
ffmpeg_path = r"C:\Users\nb1079\Desktop\ffmpeg\bin\ffmpeg.exe"

ffmpeg_cmd = [
    ffmpeg_path,
    '-y',
    '-f', 'rawvideo',
    '-vcodec', 'rawvideo',
    '-pix_fmt', 'gray',
    '-s', f'{width}x{height}',
    '-r', str(fps),
    '-i', '-',  # stdin
    '-an',
    '-vcodec', 'libx264',
    '-crf', '23',
    '-pix_fmt', 'yuv420p',
    output_path
]

try:
    print("Launching FFmpeg for direct compression...")
    # proc = subprocess.Popen(ffmpeg_cmd, stdin=subprocess.PIPE)
    proc = subprocess.Popen(
        ffmpeg_cmd,
        stdin=subprocess.PIPE,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
except FileNotFoundError:
    print("FFmpeg path is incorrect. Please update ffmpeg_path.")
    cap.release()
    exit()

# --- Threshold selection GUI with slider and video preview ---
selected_threshold = 0
preview_idx = 0

def update_preview():
    global preview_idx, gray_preview, diff_preview
    cap.set(cv2.CAP_PROP_POS_FRAMES, preview_idx)
    ret, frame = cap.read()
    if not ret:
        return

    gray_preview = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    gray_preview = enhance_contrast(gray_preview)
    diff_preview = cv2.absdiff(gray_preview, background)

    _, mask_preview = cv2.threshold(diff_preview, threshold_slider.get(), 255, cv2.THRESH_BINARY)
    mask_preview = cv2.morphologyEx(mask_preview, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))

    combined = np.hstack([
        cv2.resize(gray_preview, (320, 240)),
        cv2.resize(mask_preview, (320, 240)),
        cv2.resize(diff_preview, (320, 240)),
        cv2.resize(background, (320, 240)),
    ])
    display = cv2.cvtColor(combined, cv2.COLOR_GRAY2BGR)

    cv2.putText(display, "Original", (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,255,255), 1)
    cv2.putText(display, "Thresholded", (330, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,255,255), 1)
    cv2.putText(display, "Difference", (660, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,255,255), 1)
    cv2.putText(display, "Background Model", (990, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,255,255), 1)

    cv2.namedWindow("Threshold Preview", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("Threshold Preview", 1280, 480)
    cv2.imshow("Threshold Preview", display)

def on_threshold_slider(val):
    update_preview()

def on_frame_slider(val):
    global preview_idx
    preview_idx = int(float(val))
    update_preview()

def on_key_threshold(event):
    global preview_idx
    if event.keysym == 'Right':
        preview_idx = min(preview_idx + 1, frame_count - 1)
    elif event.keysym == 'Left':
        preview_idx = max(preview_idx - 1, 0)
    elif event.keysym == 'Return':
        confirm_threshold()
    elif event.keysym == 'Escape':
        threshold_window.quit()
    frame_slider.set(preview_idx)
    update_preview()

def confirm_threshold():
    global selected_threshold
    selected_threshold = threshold_slider.get()
    threshold_window.quit()

def random_frame():
    global preview_idx
    preview_idx = random.randint(0, frame_count - 1)
    frame_slider.set(preview_idx)
    update_preview()

# --- Tkinter window for threshold adjustment ---
threshold_window = tk.Tk()
threshold_window.title("Adjust Threshold")
threshold_window.bind('<Key>', on_key_threshold)

# Frame slider
tk.Label(threshold_window, text="Frame:").pack()
frame_slider = ttk.Scale(threshold_window, from_=0, to=frame_count-1, orient=tk.HORIZONTAL, length=400, command=on_frame_slider)
frame_slider.pack(pady=5)

# Random frame button
btn_random = tk.Button(threshold_window, text="Random Frame", command=random_frame)
btn_random.pack(pady=5)

# Threshold slider
tk.Label(threshold_window, text="Threshold:").pack()
threshold_slider = tk.Scale(threshold_window, from_=0, to=255, orient=tk.HORIZONTAL, length=400, command=on_threshold_slider)
threshold_slider.pack(pady=5)

# Confirm button
btn_confirm = tk.Button(threshold_window, text="Confirm Threshold", command=confirm_threshold)
btn_confirm.pack(pady=10)

# Initialize sliders and preview (after all widgets are created)
frame_slider.set(preview_idx)
threshold_slider.set(selected_threshold)
update_preview()

# Run GUI
threshold_window.mainloop()
cv2.destroyWindow("Threshold Preview")

# --- Process and stream frames ---
cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
for _ in tqdm(range(frame_count), desc="Processing video"):
    ret, frame = cap.read()
    if not ret:
        break
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    gray = enhance_contrast(gray)

    diff = cv2.absdiff(gray, background)
    # th = np.percentile(diff, 90)
    # _, mask = cv2.threshold(diff, th, 255, cv2.THRESH_BINARY)
    _, mask = cv2.threshold(diff, selected_threshold, 255, cv2.THRESH_BINARY)

    # automatic thresholding
    # _, mask = cv2.threshold(diff, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))

    proc.stdin.write(mask.tobytes())

# Finalize
cap.release()
proc.stdin.close()
proc.wait()

print(f"Compressed video saved as: {output_path}")
