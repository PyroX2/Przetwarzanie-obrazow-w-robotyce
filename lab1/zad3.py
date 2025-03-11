import cv2
import dearpygui.dearpygui as dpg
import numpy as np


slider_values = {"clip_limit": 5, "tile_grid_size": 8}
hist_eq_enabled = False
clahe_enabled = False

# read a image using imread
original_frame = cv2.imread('lab1/example_images/low_contrast.jpeg')
original_frame = cv2.resize(original_frame, (0, 0), fx=0.3, fy=0.3)
original_frame = cv2.cvtColor(original_frame, cv2.COLOR_BGR2RGB)
frame_shape = original_frame.shape

dpg.create_context()

def equalize_hist(frame):
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    frame[..., 2] = cv2.equalizeHist(frame[..., 2])
    return cv2.cvtColor(frame, cv2.COLOR_HSV2BGR)

def clahe(frame):
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    clahe = cv2.createCLAHE(clipLimit=slider_values["clip_limit"], tileGridSize=(
        slider_values["tile_grid_size"], slider_values["tile_grid_size"]))
    frame[..., 2] = clahe.apply(frame[..., 2])
    return cv2.cvtColor(frame, cv2.COLOR_HSV2BGR)


# Callback for enabling and disabling masking
def button_callback(sender, app_data, user_data):
    global hist_eq_enabled
    hist_eq_enabled = not hist_eq_enabled
    if hist_eq_enabled:
        dpg.set_item_label(sender, "Disable equalization")
    else:
        dpg.set_item_label(sender, "Enable equalization")
    change_image()

# Callback for enabling and disabling masking
def clahe_button_callback(sender, app_data, user_data):
    global clahe_enabled
    clahe_enabled = not clahe_enabled
    if clahe_enabled:
        dpg.set_item_label(sender, "Disable CLAHE")
    else:
        dpg.set_item_label(sender, "Enable CLAHE")
    change_image()


# Update slider values
def update_value(sender, app_data):
    """Updates the variable only when the slider is released."""
    slider_values[sender] = app_data
    change_image()  # Update image


def change_image():
    processed_image = original_frame.copy()

    if hist_eq_enabled:
        if clahe_enabled:
            processed_image = clahe(processed_image)
        else:
            processed_image = equalize_hist(processed_image)

    processed_image = np.asarray(
        processed_image, dtype=np.float32) / 255.0  # Normalize to 0-1
    dpg.set_value("processed_image", processed_image)


with dpg.texture_registry(show=False):
    frame_normalized = np.asarray(
        original_frame, dtype=np.float32) / 255.0  # Normalize to 0-1
    dpg.add_raw_texture(
        width=frame_shape[1],
        height=frame_shape[0],
        default_value=frame_normalized,
        tag="original_image",
        format=dpg.mvFormat_Float_rgb
    )
    dpg.add_raw_texture(
        width=frame_shape[1],
        height=frame_shape[0],
        default_value=frame_normalized,
        tag="processed_image",
        format=dpg.mvFormat_Float_rgb
    )
# Slider GUI
with dpg.window(label="Slider GUI", width=500, height=600, tag="Primary Window"):
    for key in slider_values.keys():
        dpg.add_slider_int(label=f"{key}", default_value=0.0, min_value=1, max_value=20,
                           tag=key, callback=update_value, width=500, height=500)
    
    eq_button = dpg.add_button(
        tag="enable_masking", label="Enable equalization")
    dpg.set_item_callback(eq_button, button_callback)

    clahe_button = dpg.add_button(
        tag="clahe_equalization", label="CLAHE equalization")
    dpg.set_item_callback(clahe_button, clahe_button_callback)

    dpg.add_spacing(count=3)

# Image GUI
with dpg.window(label="Image Viewer", width=2*frame_shape[1], height=frame_shape[0]+50, pos=[800, 0]):
    with dpg.group(horizontal=True):
        dpg.add_image("original_image")
        dpg.add_image("processed_image")

# Setup viewport
dpg.create_viewport(title='Lab1', width=800 +
                    frame_shape[1], height=frame_shape[0]+50)
dpg.setup_dearpygui()
dpg.set_global_font_scale(3)
dpg.set_primary_window("Primary Window", True)
dpg.show_viewport()
dpg.start_dearpygui()

dpg.destroy_context()
