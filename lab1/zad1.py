import dearpygui.dearpygui as dpg
import cv2
import numpy as np


# Add color spaces
items = ["RGB", "BGR", "HSV", "YUV"]
color_spaces = {
    "RGB": None,
    "BGR": cv2.COLOR_RGB2BGR,
    "HSV": cv2.COLOR_RGB2HSV,
    "YUV": cv2.COLOR_RGB2YUV
}

dpg.create_context()


# Load original image
original_frame = cv2.imread("lab1/example_images/example_image.png")
original_frame = cv2.cvtColor(original_frame, cv2.COLOR_BGR2RGB)
original_frame = cv2.resize(original_frame, (0, 0), fx=0.5, fy=0.5)
frame_shape = original_frame.shape


# Create callback for combo box with color spaces
def combo_callback(sender, app_data):
    print(f"Selected item: {app_data}, Index: {dpg.get_value(sender)}")
    if dpg.get_value(sender) == "RGB":
        show_image(original_frame)
    else:
        change_image(color_spaces[dpg.get_value(sender)])
    
    
# Change the image
def change_image(color_space):
    processed_image = original_frame.copy()
    processed_image = cv2.cvtColor(processed_image, color_space)
    show_image(processed_image)


# Show image
def show_image(frame):
    frame = np.asarray(
        frame, dtype=np.float32) / 255.0  # Normalize to 0-1
    dpg.set_value("image", frame)

# Add image texture
with dpg.texture_registry(show=False):
    frame_normalized = np.asarray(
        original_frame, dtype=np.float32) / 255.0  # Normalize to 0-1
    dpg.add_raw_texture(
        width=frame_shape[1],
        height=frame_shape[0],
        default_value=frame_normalized,
        tag="image",
        format=dpg.mvFormat_Float_rgb
    )

# Create combobox window
with dpg.window(label="Color Spaces", width=800, height=frame_shape[0]+50):
    dpg.add_combo(
        items,
        label="Select Item",
        callback=combo_callback,
        default_value=items[0]
    )

# Create image viewer window
with dpg.window(label="Image Viewer", width=frame_shape[1], height=frame_shape[0]+50, pos=[800, 0]):
    dpg.add_image("image")
    
dpg.create_viewport(title='Lab1', width=800+frame_shape[1], height=frame_shape[0]+50)

dpg.setup_dearpygui()
dpg.set_global_font_scale(3)
dpg.show_viewport()
dpg.start_dearpygui()
dpg.destroy_context()