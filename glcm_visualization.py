import numpy as np
import cv2
import matplotlib.pyplot as plt
from skimage.feature import graycomatrix, graycoprops

def show_image_and_glcm(image_path):
    """
    Visualize an image and its GLCM components
    """
    # Read and convert image
    image = cv2.imread(image_path)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    # Create a smaller patch for better visualization
    height, width = gray.shape
    center_y, center_x = height // 2, width // 2
    size = 8  # Small patch size for demonstration
    patch = gray[center_y:center_y+size, center_x:center_x+size]
    
    # Calculate GLCM for the patch
    distances = [1]  # Use distance 1 for clearer visualization
    angles = [0]     # Use only horizontal direction for simplicity
    glcm = graycomatrix(patch, distances=distances, angles=angles,
                       symmetric=True, normed=True)
    glcm = glcm[:, :, 0, 0]  # Get the 2D GLCM matrix
    
    # Create figure with subplots
    plt.style.use('default')
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    fig.suptitle('GLCM Visualization', fontsize=16)
    
    # Original Image
    axes[0, 0].imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    axes[0, 0].set_title('Original Image')
    axes[0, 0].axis('off')
    
    # Grayscale Image
    axes[0, 1].imshow(gray, cmap='gray')
    axes[0, 1].set_title('Grayscale Image')
    axes[0, 1].axis('off')
    
    # Selected Patch
    axes[0, 2].imshow(patch, cmap='gray')
    axes[0, 2].set_title(f'{size}x{size} Patch')
    axes[0, 2].axis('off')
    
    # Patch Values
    axes[1, 0].imshow(patch, cmap='gray')
    axes[1, 0].set_title('Patch Pixel Values')
    # Add pixel values as text
    for i in range(size):
        for j in range(size):
            axes[1, 0].text(j, i, str(patch[i, j]), 
                          ha='center', va='center', color='red')
    axes[1, 0].grid(True)
    
    # GLCM Matrix
    im = axes[1, 1].imshow(glcm, cmap='viridis')
    axes[1, 1].set_title('GLCM Matrix')
    plt.colorbar(im, ax=axes[1, 1])
    
    # GLCM Features
    properties = ['contrast', 'correlation', 'energy', 'homogeneity']
    features = {}
    for prop in properties:
        features[prop] = graycoprops(graycomatrix(gray, distances=[3], 
                                                angles=[0, np.pi/4, np.pi/2, 3*np.pi/4],
                                                symmetric=True, normed=True), prop)[0]
    
    feature_text = '\n'.join([f'{prop}: {features[prop][0]:.4f}' for prop in properties])
    axes[1, 2].text(0.1, 0.5, feature_text, fontsize=10)
    axes[1, 2].set_title('GLCM Features (0° direction)')
    axes[1, 2].axis('off')
    
    plt.tight_layout()
    plt.show()  # Show the plot

    # Print detailed explanation
    print("\nGLCM Process Explanation:")
    print("1. Original image is converted to grayscale")
    print("2. A small patch is selected for visualization")
    print("3. GLCM counts pixel pair occurrences:")
    print("   - For each pixel, look at its neighbor")
    print("   - Record intensity value pairs in the matrix")
    print("4. GLCM matrix shows relationship between pixel intensities:")
    print("   - Rows: reference pixel intensity")
    print("   - Columns: neighbor pixel intensity")
    print("   - Values: frequency of occurrence")
    print("\nFeature Meanings:")
    print("- Contrast: Measures intensity differences between pixels")
    print("- Correlation: Measures linear dependency between pixels")
    print("- Energy: Measures uniformity (sum of squared elements)")
    print("- Homogeneity: Measures closeness of elements to diagonal")

if __name__ == "__main__":
    image_path = "/home/duyle/Downloads/Rice_photos/Huongthom/Huong_thom-1/DSC6533_idx18.png"
    show_image_and_glcm(image_path)
