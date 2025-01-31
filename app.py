import cv2
import numpy as np
import streamlit as st
from PIL import Image
import os
import geocoder
from PIL.ExifTags import TAGS

# Function to get current geo-location
def get_location():
    try:
        g = geocoder.ip('me')
        if g.latlng:
            return g.latlng[0], g.latlng[1]
        else:
            return None, None
    except Exception:
        return None, None

# Function to extract focal length from image metadata
def get_focal_length(image_path):
    try:
        img = Image.open(image_path)
        exif_data = img._getexif()
        if exif_data:
            for tag, value in exif_data.items():
                tag_name = TAGS.get(tag, tag)
                if tag_name == "FocalLength":
                    return float(value)  # Extract focal length
        return None  # No focal length found
    except Exception:
        return None

# Function to calculate distance to the object
def calculate_distance(reference_size_real_world, reference_size_pixels, focal_length_pixels):
    return (reference_size_real_world * focal_length_pixels) / reference_size_pixels

# Function to detect tree and measure its size
def detect_tree_with_opencv(image_path, reference_size_real_world):
    if not os.path.exists(image_path):
        return None, "Image path is invalid.", 0, 0, 0, 0

    # Extract focal length from image metadata
    focal_length_pixels = get_focal_length(image_path)
    
    if focal_length_pixels is None:
        focal_length_pixels = 800  # Default focal length if not available

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

    x, y, w, h = cv2.boundingRect(large_contours[0])
    trunk_height_pixels = h // 2  

    cv2.drawContours(image, large_contours, -1, (255, 0, 0), 2)
    cv2.line(image, (x + w // 2, y), (x + w // 2, y + h), (0, 255, 0), 2)
    cv2.line(image, (x, y + h // 2), (x + w, y + h // 2), (0, 255, 0), 2)

    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    reference_object_size_pixels = 100  
    distance_to_reference_object = (reference_size_real_world * focal_length_pixels) / reference_object_size_pixels

    tree_height_meters = (h * distance_to_reference_object) / focal_length_pixels
    tree_width_meters = (w * distance_to_reference_object) / focal_length_pixels
    tree_crown_width_meters = (w * distance_to_reference_object) / focal_length_pixels
    trunk_height_meters = (trunk_height_pixels * distance_to_reference_object) / focal_length_pixels  

    return image_rgb, "Tree detected", tree_height_meters, tree_width_meters, tree_crown_width_meters, trunk_height_meters

# Streamlit app interface
def main():
    st.title("🌳 Tree Detection & Measurement App")
    st.write("Capture or upload an image to detect trees and measure their size.")

    option = st.radio("Choose input method", ("Capture Image", "Upload Image"))

    if option == "Capture Image":
        st.subheader("📸 Capture Image from Camera")
        image = st.camera_input("Take a picture of the tree")

        if image is not None:
            image_pil = Image.open(image).convert("RGB")
            image_path = "captured_image.jpg"
            image_pil.save(image_path)

            result_image, message, height, width, crown_size, trunk_height = detect_tree_with_opencv(image_path, 1.0)

            if result_image is not None:
                st.image(result_image, caption=message, use_column_width=True)
                if message == "Tree detected":
                    st.success("✅ Tree Measurement Results:")
                    st.write(f"🌲 **Tree Height:** {height:.2f} meters")
                    st.write(f"🌳 **Tree Width:** {width:.2f} meters")
                    st.write(f"🌿 **Tree Crown Size:** {crown_size:.2f} meters")
                    st.write(f"🪵 **Tree Trunk Height:** {trunk_height:.2f} meters")

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

            result_image, message, height, width, crown_size, trunk_height = detect_tree_with_opencv(image_path, 1.0)

            if result_image is not None:
                st.image(result_image, caption=message, use_column_width=True)
                if message == "Tree detected":
                    st.success("✅ Tree Measurement Results:")
                    st.write(f"🌲 **Tree Height:** {height:.2f} meters")
                    st.write(f"🌳 **Tree Width:** {width:.2f} meters")
                    st.write(f"🌿 **Tree Crown Size:** {crown_size:.2f} meters")
                    st.write(f"🪵 **Tree Trunk Height:** {trunk_height:.2f} meters")

                    lat, lon = get_location()
                    if lat and lon:
                        st.write(f"📍 **Tree Location:** {lat}, {lon}")
                    else:
                        st.write("📍 **Tree Location:** Unable to retrieve location.")

            else:
                st.error(message)

if __name__ == "__main__":
    main()
