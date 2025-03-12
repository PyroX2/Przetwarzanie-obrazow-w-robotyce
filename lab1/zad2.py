import cv2
import dearpygui.dearpygui as dpg
import numpy as np

dpg.create_context()

# Create sliders
slider_values = {"hue": 0.0, "hue_range": 1.0}
masking_enabled = False

# Load image
frame = cv2.imread("lab1/example_images/example_image.png")
frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
frame = cv2.resize(frame, (0, 0), fx=0.5, fy=0.5)
frame_shape = frame.shape


# Process HSV image
def hsv_processing(frame):
    color_value = slider_values["hue"] / 359 * 179
    hue_range = slider_values["hue_range"] / 359 * 179

    max_value = color_value+hue_range
    
    # If range is over 179 get the mask to 179 and from 0
    if max_value > 179:
        mask1 = cv2.inRange(frame, (color_value, 0, 0),
                            (179, 255, 255))
        mask2 = cv2.inRange(frame, (0, 0, 0),
                            (max_value-179, 255, 255))
        mask = cv2.bitwise_or(mask1, mask2)
    else:
        mask = cv2.inRange(frame, (color_value, 0, 0),
                       (max_value, 255, 255))
    frame = cv2.bitwise_and(frame, frame, mask=mask)
    return frame


# Callback for enabling and disabling masking
def button_callback(sender, app_data, user_data):
    global masking_enabled
    masking_enabled = not masking_enabled
    if masking_enabled:
        dpg.set_item_label(sender, "Disable masking")
    else:
        dpg.set_item_label(sender, "Enable masking")
    change_image()


# Update slider values
def update_value(sender, app_data):
    """Updates the variable only when the slider is released."""
    slider_values[sender] = app_data
    change_image()  # Update image


# Change the image
def change_image():
    processed_image = frame.copy()
    processed_image = cv2.cvtColor(processed_image, cv2.COLOR_RGB2HSV)

    if masking_enabled:
        processed_image = hsv_processing(processed_image)

    processed_image = cv2.cvtColor(processed_image, cv2.COLOR_HSV2RGB)
    processed_image = np.asarray(
        processed_image, dtype=np.float32) / 255.0  # Normalize to 0-1
    dpg.set_value("image", processed_image)


# Add image texture
with dpg.texture_registry(show=False):
    frame_normalized = np.asarray(
        frame, dtype=np.float32) / 255.0  # Normalize to 0-1
    dpg.add_raw_texture(
        width=frame_shape[1],
        height=frame_shape[0],
        default_value=frame_normalized,
        tag="image",
        format=dpg.mvFormat_Float_rgb
    )
    
# Slider GUI
with dpg.window(label="Slider GUI", width=500, height=600, tag="Primary Window"):
    
    dpg.add_slider_int(label="Hue", default_value=0.0, min_value=0, max_value=359,
                        tag="hue", callback=update_value, width=500, height=500)
    dpg.add_slider_int(label="Hue Range", default_value=1.0, min_value=1, max_value=359,
                        tag="hue_range", callback=update_value, width=500, height=500)
    masking_button = dpg.add_button(
        tag="enable_masking", label="Enable masking")
    dpg.set_item_callback(masking_button, button_callback)
    dpg.add_spacing(count=3)

# Image GUI
with dpg.window(label="Image Viewer", width=frame_shape[1], height=frame_shape[0]+50, pos=[800, 0]):
    dpg.add_image("image")

# Setup viewport
dpg.create_viewport(title='Lab1', width=800 +
                    frame_shape[1], height=frame_shape[0]+50)
dpg.setup_dearpygui()
dpg.set_global_font_scale(3)
dpg.set_primary_window("Primary Window", True)
dpg.show_viewport()
dpg.start_dearpygui()

dpg.destroy_context()
