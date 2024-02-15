import cv2
import cv2.aruco as aruco
import numpy as np
import matplotlib.pyplot as plt

# Define parameters
num_markers = 20
marker_size = 800  # Increased marker size in pixels
markers_per_row = 3  # Number of markers per row
markers_per_col = 7  # Number of markers per column
margin = 50  # Margin between markers
page_width_mm = 250  # Page width in mm
page_height_mm = 700  # Further increased page height in mm to accommodate larger markers
dpi = 300  # Dots per inch

# Calculate page size in pixels
page_width_px = int(page_width_mm * dpi / 25.4)
page_height_px = int(page_height_mm * dpi / 25.4)

# Create blank white page
page = np.ones((page_height_px, page_width_px, 3), dtype=np.uint8) * 255

# Create ArUco dictionary
dictionary = aruco.Dictionary_get(aruco.DICT_4X4_50)

# Generate markers and arrange them on the page
for i in range(num_markers):
    row = i // markers_per_row
    col = i % markers_per_row

    # Generate marker image
    marker_image = np.zeros((marker_size, marker_size), dtype=np.uint8)
    marker_image = aruco.drawMarker(dictionary, i, marker_size, marker_image, 1)

    # Convert single-channel to three-channel image
    marker_image_rgb = cv2.cvtColor(marker_image, cv2.COLOR_GRAY2RGB)

    # Calculate position to place marker on page
    x = col * (marker_size + margin) + margin
    y = row * (marker_size + margin) + margin

    # Paste marker onto page
    page[y:y+marker_size, x:x+marker_size] = marker_image_rgb

# Create figure and axis using matplotlib
fig, ax = plt.subplots(figsize=(page_width_mm / 25.4, page_height_mm / 25.4), dpi=dpi)

# Display the page without axes
ax.imshow(page)
ax.axis('off')

# Save the page as PDF
plt.savefig("markers_page.pdf", format='pdf', bbox_inches='tight', pad_inches=0)

print("Markers generated and saved as markers_page.pdf")
