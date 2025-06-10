import cv2
import random
import numpy as np
from tqdm import tqdm
import tkinter as tk
from tkinter import filedialog, ttk
import subprocess

# ######################################################################################################################
# ======================================================================================================================
# SETTINGS

# Make sure to set the correct path to your ffmpeg.exe (windows)
# Example: ffmpeg_path = r"C:\FFmpegTool\bin/ffmpeg.exe"
ffmpeg_path = r"C:\FFmpegTool\bin\ffmpeg.exe"  # User needs to verify this path


# ======================================================================================================================
# ######################################################################################################################

# --- Contrast enhancement function ---
def enhance_contrast(gray_frame):
    """
    Applies CLAHE (Contrast Limited Adaptive Histogram Equalization) to a grayscale image
    to enhance its contrast.
    """
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    return clahe.apply(gray_frame)


# --- Function to draw polygon ROI ---
def draw_polygon_roi(frame, window_name="Draw ROI"):
    """
    Allows the user to draw a polygon ROI on a given frame using mouse clicks.
    Returns the array of points defining the polygon.
    """
    points = []
    drawing = False
    # Use a temporary copy for drawing feedback to avoid modifying the original frame directly
    temp_display_frame = frame.copy()

    def mouse_callback(event, x, y, flags, param):
        nonlocal drawing, points, temp_display_frame

        if event == cv2.EVENT_LBUTTONDOWN:
            drawing = True
            points.append((x, y))
            # Draw point
            cv2.circle(temp_display_frame, (x, y), 3, (0, 255, 0), -1)  # Green circle
            if len(points) > 1:
                cv2.line(temp_display_frame, points[-2], points[-1], (0, 255, 0), 2)  # Green line
            cv2.imshow(window_name, temp_display_frame)

        elif event == cv2.EVENT_LBUTTONUP:
            drawing = False

    cv2.namedWindow(window_name)
    cv2.setMouseCallback(window_name, mouse_callback)
    cv2.imshow(window_name, temp_display_frame)  # Show initial frame

    # Print instructions to console, as they are now also in the info window
    print(f"\n--- Instructions for '{window_name}' ---")
    print("  - Click to add points to outline the tail. Connect the points to form a polygon.")
    print("  - Press 'c' to CONFIRM the ROI (requires at least 3 points).")
    print("  - Press 'd' to DELETE the last point.")
    print("  - Press 'r' to RESET (clear all points and start over).")
    print("  - Press 'Escape' to CANCEL (no ROI will be selected for this frame).")

    while True:
        key = cv2.waitKey(1) & 0xFF
        if key == ord('c'):  # Confirm ROI
            if len(points) >= 3:  # A polygon needs at least 3 points
                break
            else:
                # Use a custom message box instead of alert()
                # For simplicity here, will print to console. A full GUI app would use a tk.messagebox.
                print("GUI Message: Please add at least 3 points to form a polygon before confirming.")
        elif key == ord('d'):  # Delete last point
            if points:
                points.pop()
                # Redraw to show changes on a fresh copy of the frame to erase previous drawings
                temp_display_frame = frame.copy()
                for i in range(len(points)):
                    cv2.circle(temp_display_frame, points[i], 3, (0, 255, 0), -1)
                    if i > 0:
                        cv2.line(temp_display_frame, points[i - 1], points[i], (0, 255, 0), 2)
                cv2.imshow(window_name, temp_display_frame)
        elif key == ord('r'):  # Reset all points
            points = []
            temp_display_frame = frame.copy()
            cv2.imshow(window_name, temp_display_frame)
        elif key == 27:  # Escape key
            points = []  # Clear points if cancelled
            break

    cv2.destroyWindow(window_name)  # Close the ROI drawing window
    return np.array(points, dtype=np.int32) if points else None


# --- Function to display keyboard shortcuts in a separate window ---
def show_keyboard_shortcuts_info():
    """
    Creates and displays a Tkinter Toplevel window with keyboard shortcut information.
    """
    info_window = tk.Toplevel()
    info_window.title("Keyboard Shortcuts Info")
    info_window.geometry("450x400")  # Set a fixed size for the info window
    info_window.resizable(False, False)  # Make the window non-resizable

    # Frame for better organization and padding
    main_frame = ttk.Frame(info_window, padding="15")
    main_frame.pack(fill=tk.BOTH, expand=True)

    ttk.Label(main_frame, text="Keyboard Shortcuts Guide", font=("Arial", 14, "bold")).pack(pady=10)

    # Shortcuts for Frame Selection Phase
    ttk.Label(main_frame, text="--- Frame Selection Phase ---", font=("Arial", 11, "underline")).pack(anchor=tk.W,
                                                                                                      pady=(10, 5))
    ttk.Label(main_frame, text="  •  Left/Right Arrow: Navigate frames", font=("Arial", 10)).pack(anchor=tk.W)
    ttk.Label(main_frame, text="  •  Enter: Confirm frame & proceed to ROI drawing", font=("Arial", 10)).pack(
        anchor=tk.W)
    ttk.Label(main_frame, text="  •  Escape: Exit the application", font=("Arial", 10)).pack(anchor=tk.W)

    # Shortcuts for ROI Drawing Phase (within OpenCV window)
    ttk.Label(main_frame, text="\n--- ROI Drawing Phase (OpenCV Window) ---", font=("Arial", 11, "underline")).pack(
        anchor=tk.W, pady=(10, 5))
    ttk.Label(main_frame, text="  •  Mouse Click: Add point to polygon", font=("Arial", 10)).pack(anchor=tk.W)
    ttk.Label(main_frame, text="  •  'c': Confirm ROI (requires at least 3 points)", font=("Arial", 10)).pack(
        anchor=tk.W)
    ttk.Label(main_frame, text="  •  'd': Delete last point", font=("Arial", 10)).pack(anchor=tk.W)
    ttk.Label(main_frame, text="  •  'r': Reset (clear all points)", font=("Arial", 10)).pack(anchor=tk.W)
    ttk.Label(main_frame, text="  •  Escape: Cancel ROI selection for current frame", font=("Arial", 10)).pack(
        anchor=tk.W)

    # Shortcuts for Threshold Adjustment Phase
    ttk.Label(main_frame, text="\n--- Threshold Adjustment Phase ---", font=("Arial", 11, "underline")).pack(
        anchor=tk.W, pady=(10, 5))
    ttk.Label(main_frame, text="  •  Left/Right Arrow: Navigate preview frames", font=("Arial", 10)).pack(anchor=tk.W)
    ttk.Label(main_frame, text="  •  Enter: Confirm threshold & start processing", font=("Arial", 10)).pack(anchor=tk.W)
    ttk.Label(main_frame, text="  •  Escape: Exit the application", font=("Arial", 10)).pack(anchor=tk.W)

    # "Got it" button to close the info window
    ttk.Button(main_frame, text="Got It!", command=info_window.destroy).pack(pady=20)


# --- File selection using Tkinter ---
root = tk.Tk()
root.withdraw()  # Hide the main Tkinter window

# Display keyboard shortcuts info window first
show_keyboard_shortcuts_info()

input_path = filedialog.askopenfilename(title="Select input video", filetypes=[("MP4 files", "*.mp4")])
if not input_path:
    print("No video selected. Exiting.")
    exit()

# Open the video file
cap = cv2.VideoCapture(input_path)
if not cap.isOpened():
    print(f"Error: Could not open video file {input_path}. Exiting.")
    exit()

frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = cap.get(cv2.CAP_PROP_FPS)

if frame_count == 0:
    print("Error: Video contains no frames. Exiting.")
    cap.release()
    exit()

# --- Variables for sequential frame and ROI selection ---
frame_1_idx = -1
frame_2_idx = -1
roi_points_1 = None
roi_points_2 = None
current_selection_step = 1  # State: 1 for frame 1, 2 for frame 2
current_frame_preview_idx = 0  # Current frame shown on the slider for preview


def show_frame_for_selection(frame_index):
    """
    Reads a frame from the video, resizes it for preview, adds text overlay, and displays it.
    Returns the original full-size frame.
    """
    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_index)
    ret, frame = cap.read()
    if not ret:
        print(f"Warning: Could not read frame {frame_index}.")
        return None  # Return None if frame cannot be read

    # Create a display copy (resized) for the Tkinter preview window
    display_frame = cv2.resize(frame.copy(), (640, 480))
    text = f"Frame: {frame_index}"
    if current_selection_step == 1:
        text += " (Select Frame 1: Tail in Center)"
    elif current_selection_step == 2:
        text += " (Select Frame 2: Tail Off-Center)"

    cv2.putText(display_frame, text, (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
    cv2.imshow("Frame Preview", display_frame)
    return frame  # Return the original frame for potential ROI drawing


def on_slider_move_selection(val):
    """Callback for the frame selection slider."""
    global current_frame_preview_idx
    current_frame_preview_idx = int(float(val))
    show_frame_for_selection(current_frame_preview_idx)


def confirm_frame_and_draw_roi():
    """
    Confirms the current frame selection and initiates the ROI drawing process.
    Manages the sequential flow for Frame 1 and Frame 2 selection.
    """
    global frame_1_idx, frame_2_idx, roi_points_1, roi_points_2, current_selection_step

    # Read the current original frame for ROI drawing before resizing
    cap.set(cv2.CAP_PROP_POS_FRAMES, current_frame_preview_idx)
    ret, original_frame_for_roi = cap.read()
    if not ret:
        print("Failed to read frame for ROI drawing. Please try another frame.")
        return

    cv2.destroyAllWindows()  # Close the current preview window before opening ROI drawing window

    if current_selection_step == 1:
        frame_1_idx = current_frame_preview_idx
        label_selected.config(text=f"Frame 1 selected: {frame_1_idx}. Now draw ROI for Frame 1.")

        # Call the ROI drawing function
        roi_points_1 = draw_polygon_roi(original_frame_for_roi, "Draw ROI for Frame 1 (Tail Center)")

        if roi_points_1 is None or len(roi_points_1) < 3:  # Check if ROI was cancelled or invalid
            print("ROI for Frame 1 was cancelled or insufficient points were provided. Exiting.")
            preview_window.quit()
            return

        print(f"ROI for Frame 1 confirmed with {len(roi_points_1)} points.")
        current_selection_step = 2  # Move to the next selection step
        label_status.config(text="Select Frame 2: Tail Off-Center")
        btn_confirm_selection.config(text="Confirm Frame 2 & Draw ROI")
        # Re-show the preview for the next step, ensuring the cv2 window reappears
        show_frame_for_selection(current_frame_preview_idx)

    elif current_selection_step == 2:
        frame_2_idx = current_frame_preview_idx
        label_selected.config(text=f"Frame 2 selected: {frame_2_idx}. Now draw ROI for Frame 2.")

        # Call the ROI drawing function
        roi_points_2 = draw_polygon_roi(original_frame_for_roi, "Draw ROI for Frame 2 (Tail Off-Center)")

        if roi_points_2 is None or len(roi_points_2) < 3:  # Check if ROI was cancelled or invalid
            print("ROI for Frame 2 was cancelled or insufficient points were provided. Exiting.")
            preview_window.quit()
            return

        print(f"ROI for Frame 2 confirmed with {len(roi_points_2)} points.")
        # Both frames and ROIs are selected, close the Tkinter window
        preview_window.quit()


def on_key_selection(event):
    """Keyboard callback for the frame selection window."""
    global current_frame_preview_idx
    if event.keysym == 'Right':
        current_frame_preview_idx = min(current_frame_preview_idx + 1, frame_count - 1)
    elif event.keysym == 'Left':
        current_frame_preview_idx = max(current_frame_preview_idx - 1, 0)
    elif event.keysym == 'Return':  # Press Enter to confirm frame and proceed to ROI
        confirm_frame_and_draw_roi()
    elif event.keysym == 'Escape':  # Press Escape to quit the selection process
        preview_window.quit()
    slider.set(current_frame_preview_idx)
    show_frame_for_selection(current_frame_preview_idx)


# --- GUI window for sequential frame and ROI selection ---
preview_window = tk.Tk()
preview_window.title("Select Frames & Outline Tail for Background Subtraction")
preview_window.bind('<Key>', on_key_selection)

label_status = tk.Label(preview_window, text="Select Frame 1: Tail in Center", font=("Arial", 12, "bold"))
label_status.pack(pady=5)

slider = ttk.Scale(preview_window, from_=0, to=frame_count - 1, orient=tk.HORIZONTAL, length=500,
                   command=on_slider_move_selection)
slider.pack(pady=10)

btn_confirm_selection = tk.Button(preview_window, text="Confirm Frame 1 & Draw ROI", command=confirm_frame_and_draw_roi,
                                  font=("Arial", 10), bg="#4CAF50", fg="white")
btn_confirm_selection.pack(pady=5)

label_selected = tk.Label(preview_window, text="No frames selected yet.", font=("Arial", 10))
label_selected.pack(pady=5)

# Start preview with the first frame
show_frame_for_selection(current_frame_preview_idx)
preview_window.mainloop()  # This blocks until preview_window.quit() is called
cv2.destroyAllWindows()  # Ensure all OpenCV windows are closed after Tkinter window closes

# --- Check if selection was successful ---
if frame_1_idx == -1 or frame_2_idx == -1 or roi_points_1 is None or roi_points_2 is None:
    print("Frame or ROI selection was incomplete or cancelled. Exiting.")
    cap.release()
    exit()

print(f"Successfully selected Frame 1 (idx: {frame_1_idx}) with {len(roi_points_1)} ROI points.")
print(f"Successfully selected Frame 2 (idx: {frame_2_idx}) with {len(roi_points_2)} ROI points.")

# --- Build custom background model by removing tail and combining ---
print("\nGenerating tail-less background model using selected frames and ROIs...")


def remove_tail(frame_idx, roi_pts):
    """
    Loads a specific frame, converts to grayscale, enhances contrast,
    creates a mask from the ROI, and then uses inpainting to remove the tail.
    """
    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
    ret, frame = cap.read()
    if not ret:
        print(f"Error reading frame {frame_idx} for tail removal.")
        return None

    gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    enhanced_gray = enhance_contrast(gray_frame)

    # Create an empty mask of the same size as the grayscale frame
    mask = np.zeros(enhanced_gray.shape, dtype=np.uint8)

    # Fill the polygon area in the mask with white (255)
    # The ROI points must be an array of arrays (e.g., [roi_pts]) for fillPoly
    cv2.fillPoly(mask, [roi_pts], 255)

    # Inpaint the masked area (remove the tail)
    # INPAINT_TELEA is often good for natural images, INPAINT_NS for more structured areas
    # inpaintRadius: The radius of a circular neighborhood of each point in the inpaintMask
    # that is considered by the algorithm. A larger radius will consider more surrounding
    # pixels for filling, potentially leading to smoother but less precise results.
    tail_removed_frame = cv2.inpaint(enhanced_gray, mask, inpaintRadius=5, flags=cv2.INPAINT_TELEA)

    return tail_removed_frame


# Process Frame 1 and remove the tail
frame_1_tail_removed = remove_tail(frame_1_idx, roi_points_1)
# Process Frame 2 and remove the tail
frame_2_tail_removed = remove_tail(frame_2_idx, roi_points_2)

if frame_1_tail_removed is None or frame_2_tail_removed is None:
    print("Failed to generate one or both tail-less frames. Exiting.")
    cap.release()
    exit()

# Combine the two tail-less frames to form the final background model
# Using median provides robustness against remaining slight variations in the 'background' areas.
background = np.median([frame_1_tail_removed, frame_2_tail_removed], axis=0).astype(np.uint8)
print("Tail-less background model successfully generated.")

# Display the generated background for user confirmation
cv2.imshow("Generated Tail-less Background Model (Press any key to continue)", background)
cv2.waitKey(0)  # Wait indefinitely until a key is pressed
cv2.destroyWindow("Generated Tail-less Background Model (Press any key to continue)")

# --- FFmpeg Streaming Setup ---
# Construct the output path for the processed video
output_path = input_path.rsplit('.', 1)[0] + "_cleaned_compressed.mp4"

# FFmpeg command to convert raw grayscale frames to H.264 MP4
ffmpeg_cmd = [
    ffmpeg_path,
    '-y',  # Overwrite output file without asking
    '-f', 'rawvideo',  # Input format is raw video
    '-vcodec', 'rawvideo',  # Input video codec is raw
    '-pix_fmt', 'gray',  # Input pixel format is grayscale
    '-s', f'{width}x{height}',  # Input frame size
    '-r', str(fps),  # Input frame rate
    '-i', '-',  # Read input from stdin
    '-an',  # No audio
    '-vcodec', 'libx264',  # Output video codec is H.264
    '-crf', '23',  # Constant Rate Factor (0-51, lower means higher quality/larger file)
    '-pix_fmt', 'yuv420p',  # Output pixel format (required for H.264 compatibility)
    output_path  # Output file path
]

try:
    print("\nLaunching FFmpeg for direct video compression...")
    # Use subprocess.Popen to run FFmpeg, redirecting stdin for frame input
    # stdout and stderr are redirected to DEVNULL to prevent cluttering the console
    proc = subprocess.Popen(
        ffmpeg_cmd,
        stdin=subprocess.PIPE,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
except FileNotFoundError:
    print(f"Error: FFmpeg not found at '{ffmpeg_path}'. Please verify and update the 'ffmpeg_path' variable.")
    cap.release()
    exit()
except Exception as e:
    print(f"An error occurred while launching FFmpeg: {e}")
    cap.release()
    exit()

# --- Threshold selection GUI with slider and video preview ---
# This section now uses the newly generated 'background' for difference calculation.
selected_threshold = 0  # Default threshold value
preview_idx = 0  # Current frame index for threshold preview


def update_preview():
    """
    Updates the four-panel preview window for threshold adjustment:
    Original, Thresholded, Difference, and Background Model.
    """
    global preview_idx, gray_preview, diff_preview
    cap.set(cv2.CAP_PROP_POS_FRAMES, preview_idx)
    ret, frame = cap.read()
    if not ret:
        print(f"Warning: Could not read frame {preview_idx} for threshold preview.")
        return

    # Convert frame to grayscale and enhance contrast for the current preview frame
    gray_preview = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    gray_preview = enhance_contrast(gray_preview)

    # Calculate the absolute difference between the current frame and the custom background
    diff_preview = cv2.absdiff(gray_preview, background)  # Using the new 'background'

    # Apply binary thresholding based on the slider value
    _, mask_preview = cv2.threshold(diff_preview, threshold_slider.get(), 255, cv2.THRESH_BINARY)
    # Apply morphological opening to remove small noise (e.g., small white dots)
    mask_preview = cv2.morphologyEx(mask_preview, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))

    # Resize all four images to a common dimension for horizontal stacking
    display_gray_preview = cv2.resize(gray_preview, (320, 240))
    display_mask_preview = cv2.resize(mask_preview, (320, 240))
    display_diff_preview = cv2.resize(diff_preview, (320, 240))
    display_background = cv2.resize(background, (320, 240))  # The generated background

    # Stack the images horizontally
    combined = np.hstack([
        display_gray_preview,
        display_mask_preview,
        display_diff_preview,
        display_background,
    ])
    # Convert back to BGR for text overlay (putText expects 3 channels)
    display = cv2.cvtColor(combined, cv2.COLOR_GRAY2BGR)

    # Add labels to each panel
    cv2.putText(display, "Original (Enhanced)", (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
    cv2.putText(display, "Thresholded Output", (330, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
    cv2.putText(display, "Difference from BG", (660, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
    cv2.putText(display, "Background Model", (990, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)

    cv2.namedWindow("Threshold Preview", cv2.WINDOW_NORMAL)  # Allow window resizing
    cv2.resizeWindow("Threshold Preview", 1280, 480)  # Set initial window size
    cv2.imshow("Threshold Preview", display)


def on_threshold_slider(val):
    """Callback for the threshold slider."""
    update_preview()


def on_frame_slider(val):
    """Callback for the frame preview slider in threshold window."""
    global preview_idx
    preview_idx = int(float(val))
    update_preview()


def on_key_threshold(event):
    """Keyboard callback for the threshold adjustment window."""
    global preview_idx
    if event.keysym == 'Right':  # Move to next frame
        preview_idx = min(preview_idx + 1, frame_count - 1)
    elif event.keysym == 'Left':  # Move to previous frame
        preview_idx = max(preview_idx - 1, 0)
    elif event.keysym == 'Return':  # Press Enter to confirm threshold
        confirm_threshold()
    elif event.keysym == 'Escape':  # Press Escape to quit threshold selection
        threshold_window.quit()
    frame_slider.set(preview_idx)
    update_preview()


def confirm_threshold():
    """Confirms the selected threshold and closes the window."""
    global selected_threshold
    selected_threshold = threshold_slider.get()
    threshold_window.quit()


def random_frame():
    """Jumps to a random frame for threshold preview."""
    global preview_idx
    preview_idx = random.randint(0, frame_count - 1)
    frame_slider.set(preview_idx)
    update_preview()


# --- Tkinter window for threshold adjustment ---
threshold_window = tk.Tk()
threshold_window.title("Adjust Threshold for Background Subtraction")
threshold_window.bind('<Key>', on_key_threshold)

tk.Label(threshold_window, text="Preview Frame:", font=("Arial", 10, "bold")).pack(pady=5)
frame_slider = ttk.Scale(threshold_window, from_=0, to=frame_count - 1, orient=tk.HORIZONTAL, length=400,
                         command=on_frame_slider)
frame_slider.pack(pady=5)

btn_random = tk.Button(threshold_window, text="Random Frame", command=random_frame, font=("Arial", 9), bg="#607D8B",
                       fg="white")
btn_random.pack(pady=5)

tk.Label(threshold_window, text="Adjust Threshold:", font=("Arial", 10, "bold")).pack(pady=5)
threshold_slider = tk.Scale(threshold_window, from_=0, to=255, orient=tk.HORIZONTAL, length=400,
                            command=on_threshold_slider,
                            label="Threshold Value", showvalue=True, tickinterval=50)
threshold_slider.pack(pady=5)

btn_confirm = tk.Button(threshold_window, text="Confirm Threshold", command=confirm_threshold, font=("Arial", 10),
                        bg="#009688", fg="white")
btn_confirm.pack(pady=10)

# Initialize sliders and preview (after all widgets are created and packed)
frame_slider.set(preview_idx)
# Set an initial reasonable threshold, e.g., 50
threshold_slider.set(50)
update_preview()  # Display the initial preview

# Run the Tkinter GUI for threshold adjustment
threshold_window.mainloop()
cv2.destroyWindow("Threshold Preview")  # Close the OpenCV preview window after Tkinter window closes

# --- Process and stream frames ---
# Rewind video capture to the beginning
cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
print(f"\nStarting video processing with selected threshold: {selected_threshold}...")

# Iterate through each frame, apply background subtraction, and stream to FFmpeg
for _ in tqdm(range(frame_count), desc="Processing video"):
    ret, frame = cap.read()
    if not ret:
        print(f"Warning: Failed to read frame during main processing at index {_}.")
        break  # Exit loop if frame cannot be read

    # Convert frame to grayscale and enhance contrast
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    gray = enhance_contrast(gray)

    # Calculate absolute difference from the custom background model
    diff = cv2.absdiff(gray, background)

    # Apply binary thresholding using the user-selected threshold
    _, mask = cv2.threshold(diff, selected_threshold, 255, cv2.THRESH_BINARY)

    # Apply morphological opening to clean up the mask (remove small noise)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))

    # Write the resulting grayscale mask directly to FFmpeg's stdin
    try:
        proc.stdin.write(mask.tobytes())
    except BrokenPipeError:
        print("FFmpeg pipe broke. Likely FFmpeg exited prematurely. Check FFmpeg output or command.")
        break
    except Exception as e:
        print(f"Error writing to FFmpeg stdin: {e}")
        break

# --- Finalize ---
cap.release()  # Release the video capture object
proc.stdin.close()  # Close stdin to FFmpeg, signaling end of input
proc.wait()  # Wait for FFmpeg process to finish

print(f"\nVideo processing complete. Compressed video saved as: {output_path}")

