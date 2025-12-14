import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import os

# Set global font family to serif
plt.rcParams['font.family'] = 'serif'

# Define file paths
base_dir = '/root/lanyun-tmp/data_code'
chengdu_path = os.path.join(base_dir, 'chengdu_grid_map.png')
haikou_path = os.path.join(base_dir, 'haikou_grid_map.png')

# Load images
try:
    img_chengdu = mpimg.imread(chengdu_path)
    img_haikou = mpimg.imread(haikou_path)
except FileNotFoundError as e:
    print(f"Error loading images: {e}")
    exit(1)

# Get dimensions (Height, Width, Channels)
h1, w1, _ = img_chengdu.shape
h2, w2, _ = img_haikou.shape

# Calculate aspect ratios (Width / Height)
ar1 = w1 / h1
ar2 = w2 / h2

# Create figure
# We want to maintain consistent scale (zoom level) between the two images.
# Since both images have the same pixel width (3510), we should display them
# with the same physical width on the figure.
# The heights will then naturally vary based on their pixel heights.
# Chengdu (h=3308) should appear taller than Haikou (h=2920).

fig_width = 14
# Calculate required height based on the taller image (Chengdu) to fit well
# Aspect Ratio of Chengdu is ~1.06 (W/H). So H = W / 1.06.
# If each subplot width is roughly W_sub = fig_width / 2,
# Height should be enough to accommodate the taller image.
fig, axes = plt.subplots(1, 2, figsize=(fig_width, 8), 
                         gridspec_kw={'width_ratios': [1, 1]})

# Plot Chengdu map
axes[0].imshow(img_chengdu)
axes[0].axis('off')  # Hide axes
axes[0].set_title('(a) Chengdu Urban Grid Map', fontsize=20, fontweight='bold', pad=15)

# Plot Haikou map
axes[1].imshow(img_haikou)
axes[1].axis('off')  # Hide axes
axes[1].set_title('(b) Haikou Urban Grid Map', fontsize=20, fontweight='bold', pad=15)

# Adjust spacing
# Use tight_layout with padding to handle varying heights elegantly
plt.tight_layout(pad=3.0)

# Save the merged figure
output_path = os.path.join(base_dir, 'merged_grid_maps.png')
plt.savefig(output_path, bbox_inches='tight', dpi=300)

print(f"Merged image saved to {output_path}")
