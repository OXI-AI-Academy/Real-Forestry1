import cv2
import numpy as np
import streamlit as st
from PIL import Image
import os
import geocoder  # Install using: pip install geocoder

# Function to get current geo-location
def get_location():
    try:
        g = geocoder.ip('me')  # Get current location based on IP
        if g.latlng:
            return g.latlng[0], g.latlng[1]  # Return (latitude, longitude)
        else:
            return None, None
    except Exception as e:
        return None, None

# Function to calculate the distance to the object
def calculate_distance_to_object(reference_size_real_world, reference_size_pixels, focal_length_pixels):
    return (reference_size_real_world * focal_length_pixels) / reference_size_pixels

# Function to detect tree and measure its size
def detect_tree_with_opencv(image_path, focal_length_pixels, reference_size_real_world):
    if not os.path.exists(image_path):
        return None, "Image path is invalid.", 0, 0, 0, 0

    image = cv2.imread(image_path)
    if image is None:
        return None, "Failed to load image.", 0, 0, 0, 0

    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    
    # Define green range (adjust for better detection)
    lower_green = np.array([25, 40, 40])
    upper_green = np.array([90, 255, 255])
    
    mask = cv2.inRange(hsv, lower_green, upper_green)
    kernel = np.ones((5, 5), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)

    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    min_area = 1000
    min_perimeter = 500
    large_contours = [cnt for cnt in contours if cv2.contourArea(cnt) > min_area and cv2.arcLength(cnt, True) > min_perimeter]

    if not large_contours:
        return None, "No significant tree detected.", 0, 0, 0, 0

    # Find the bounding box (tree detection)
    x, y, w, h = cv2.boundingRect(large_contours[0])
    
    # Approximate Trunk Height (bottom to top measurement)
    trunk_height_pixels = h // 2  # Assuming trunk is lower half of the bounding box

    cv2.drawContours(image, large_contours, -1, (255, 0, 0), 2)
    cv2.line(image, (x + w // 2, y), (x + w // 2, y + h), (0, 255, 0), 2)
    cv2.line(image, (x, y + h // 2), (x + w, y + h // 2), (0, 255, 0), 2)

    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    reference_object_size_pixels = 100  
    reference_size_real_world = 1.0  

    distance_to_reference_object = (reference_size_real_world * focal_length_pixels) / reference_object_size_pixels

    tree_height_meters = (h * distance_to_reference_object) / focal_length_pixels
    tree_width_meters = (w * distance_to_reference_object) / focal_length_pixels
    tree_crown_width_meters = (w * distance_to_reference_object) / focal_length_pixels
    trunk_height_meters = (trunk_height_pixels * distance_to_reference_object) / focal_length_pixels  # Trunk height calculation

    return image_rgb, "Tree detected", tree_height_meters, tree_width_meters, tree_crown_width_meters, trunk_height_meters

# Streamlit app interface
def main():
    st.title("🌳 Tree Detection & Measurement App")
    st.write("Capture or upload an image to detect trees and measure their size.")

    option = st.radio("Choose input method", ("Capture Image", "Upload Image"))

    # Camera parameters (focal length in pixels)
    focal_length_pixels = st.number_input("📷 Focal Length (in pixels)", min_value=1, value=800, step=1)

    if focal_length_pixels <= 0:
        st.error("Focal length must be a positive value.")
        return

    if option == "Capture Image":
        st.subheader("📸 Capture Image from Camera")
        image = st.camera_input("Take a picture of the tree")

        if image is not None:
            image_pil = Image.open(image).convert("RGB")
            image_path = "captured_image.jpg"
            image_pil.save(image_path)

            result_image, message, height, width, crown_size, trunk_height = detect_tree_with_opencv(
                image_path, focal_length_pixels, 1.0
            )

            if result_image is not None:
                st.image(result_image, caption=message, use_column_width=True)
                if message == "Tree detected":
                    st.success("✅ Tree Measurement Results:")
                    st.write(f"🌲 **Tree Height:** {height:.2f} meters")
                    st.write(f"🌳 **Tree Width:** {width:.2f} meters")
                    st.write(f"🌿 **Tree Crown Size:** {crown_size:.2f} meters")
                    st.write(f"🪵 **Tree Trunk Height:** {trunk_height:.2f} meters")

                    # Get Geo-location
                    lat, lon = get_location()
                    if lat and lon:
                        st.write(f"📍 **Tree Location:** {lat}, {lon}")
                    else:
                        st.write("📍 **Tree Location:** Unable to retrieve location.")

            else:
                st.error(message)

    elif option == "Upload Image":
        uploaded_file = st.file_uploader("📤 Choose an image...", type=["jpg", "jpeg", "png"])

        if uploaded_file is not None:
            image_pil = Image.open(uploaded_file).convert("RGB")
            image_path = "uploaded_image.jpg"
            image_pil.save(image_path)

            result_image, message, height, width, crown_size, trunk_height = detect_tree_with_opencv(
                image_path, focal_length_pixels, 1.0
            )

            if result_image is not None:
                st.image(result_image, caption=message, use_column_width=True)
                if message == "Tree detected":
                    st.success("✅ Tree Measurement Results:")
                    st.write(f"🌲 **Tree Height:** {height:.2f} meters")
                    st.write(f"🌳 **Tree Width:** {width:.2f} meters")
                    st.write(f"🌿 **Tree Crown Size:** {crown_size:.2f} meters")
                    st.write(f"🪵 **Tree Trunk Height:** {trunk_height:.2f} meters")

                    # Get Geo-location
                    lat, lon = get_location()
                    if lat and lon:
                        st.write(f"📍 **Tree Location:** {lat}, {lon}")
                    else:
                        st.write("📍 **Tree Location:** Unable to retrieve location.")

            else:
                st.error(message)

if __name__ == "__main__":
    main()
