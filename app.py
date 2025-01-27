import cv2
import numpy as np
import streamlit as st
from PIL import Image
import os

# Function to calculate the distance to the object based on known reference size
def calculate_distance_to_object(reference_size_real_world, reference_size_pixels, focal_length_pixels):
    # Calculate the distance to the reference object
    distance_to_reference_object = (reference_size_real_world * focal_length_pixels) / reference_size_pixels
    return distance_to_reference_object

# Function to detect tree and estimate its real-world size using camera parameters
def detect_tree_with_opencv(image_path, focal_length_pixels, reference_size_real_world):
    # Check if image path exists
    if not os.path.exists(image_path):
        return None, "Image path is invalid or does not exist.", 0, 0, 0

    # Load the image using OpenCV
    image = cv2.imread(image_path)
    if image is None:
        return None, "Failed to load image.", 0, 0, 0

    # Convert the image to HSV (Hue, Saturation, Value) color space
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

    # Define color range for green (trees)
    lower_green = np.array([25, 40, 40])
    upper_green = np.array([90, 255, 255])

    # Create a mask that captures the green regions
    mask = cv2.inRange(hsv, lower_green, upper_green)

    # Clean up the mask using morphological operations (erosion and dilation)
    kernel_size = 5
    kernel = np.ones((kernel_size, kernel_size), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)

    # Apply the mask to the original image to extract tree-like regions
    result = cv2.bitwise_and(image, image, mask=mask)

    # Find contours (outlines) of the detected green areas
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Filter contours based on area and shape (ignore small or irregular contours)
    min_area = 1000
    min_perimeter = 500
    large_contours = [
        cnt for cnt in contours if cv2.contourArea(cnt) > min_area and cv2.arcLength(cnt, True) > min_perimeter
    ]

    # If no valid contour is found, return None
    if not large_contours:
        return None, "No significant tree detected.", 0, 0, 0

    # Find the bounding box for the largest tree (for measurement purposes)
    x, y, w, h = cv2.boundingRect(large_contours[0])

    # Draw the largest contour (tree) on the original image
    cv2.drawContours(image, large_contours, -1, (255, 0, 0), 2)

    # Draw vertical and horizontal lines around the detected tree
    cv2.line(image, (x + w // 2, y), (x + w // 2, y + h), (0, 255, 0), 2)  # Vertical line
    cv2.line(image, (x, y + h // 2), (x + w, y + h // 2), (0, 255, 0), 2)  # Horizontal line

    # Convert the result back to RGB for displaying in Streamlit
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    # Approximate tree size (height, width) based on bounding box
    tree_height_pixels = h
    tree_width_pixels = w
    tree_crown_width_pixels = w

    # Assume a known size of reference object in the image
    reference_object_size_pixels = 100  # e.g., 100 pixels
    reference_size_real_world = 1.0  # e.g., 1 meter

    # Calculate the distance to the reference object (in meters)
    distance_to_reference_object = calculate_distance_to_object(
        reference_size_real_world, reference_object_size_pixels, focal_length_pixels
    )

    # Calculate the real-world size of the tree using the reference object's distance
    tree_height_meters = (tree_height_pixels * distance_to_reference_object) / focal_length_pixels
    tree_width_meters = (tree_width_pixels * distance_to_reference_object) / focal_length_pixels
    tree_crown_width_meters = (tree_crown_width_pixels * distance_to_reference_object) / focal_length_pixels

    return image_rgb, "Tree detected", tree_height_meters, tree_width_meters, tree_crown_width_meters


# Streamlit app interface
def main():
    st.title("Tree Detection and Measurement with Camera Parameters")
    st.write("Capture or upload an image to detect trees and measure their size.")

    option = st.radio("Choose input method", ("Capture Image", "Upload Image"))

    # Camera parameters (focal length in pixels)
    focal_length_pixels = st.number_input("Focal Length (in pixels)", min_value=1, value=800, step=1)

    if focal_length_pixels <= 0:
        st.error("Focal length must be a positive value.")
        return

    if option == "Capture Image":
        st.subheader("Capture Image from Camera")
        image = st.camera_input("Take a picture of the tree")

        if image is not None:
            image_pil = Image.open(image).convert("RGB")
            image_path = "captured_image.jpg"
            image_pil.save(image_path)

            result_image, message, height, width, crown_size = detect_tree_with_opencv(
                image_path, focal_length_pixels, 1.0
            )

            if result_image is not None:
                st.image(result_image, caption=message, use_column_width=True)
                if message == "Tree detected":
                    st.write(f"Tree Height (approx): {height:.2f} meters")
                    st.write(f"Tree Width (approx): {width:.2f} meters")
                    st.write(f"Tree Crown Size (approx): {crown_size:.2f} meters")
            else:
                st.error(message)

    elif option == "Upload Image":
        uploaded_file = st.file_uploader("Choose an image...", type="jpg")

        if uploaded_file is not None:
            image_pil = Image.open(uploaded_file).convert("RGB")
            image_path = "uploaded_image.jpg"
            image_pil.save(image_path)

            result_image, message, height, width, crown_size = detect_tree_with_opencv(
                image_path, focal_length_pixels, 1.0
            )

            if result_image is not None:
                st.image(result_image, caption=message, use_column_width=True)
                if message == "Tree detected":
                    st.write(f"Tree Height (approx): {height:.2f} meters")
                    st.write(f"Tree Width (approx): {width:.2f} meters")
                    st.write(f"Tree Crown Size (approx): {crown_size:.2f} meters")
            else:
                st.error(message)


if __name__ == "__main__":
    main()
