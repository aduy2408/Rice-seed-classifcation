import numpy as np
import matplotlib.pyplot as plt

# Create a simple 4x4 image with 3 gray levels (0, 1, 2)
image = np.array([
    [0, 0, 1, 1],
    [0, 0, 1, 1],
    [0, 2, 2, 2],
    [0, 2, 2, 2]
], dtype=np.uint8)

def create_glcm(img, symmetric=True):
    """Create GLCM manually"""
    levels = 3
    glcm = np.zeros((levels, levels), dtype=int)
    rows, cols = img.shape
    
    for i in range(rows):
        for j in range(cols-1):
            reference = img[i, j]
            neighbor = img[i, j+1]
            glcm[reference, neighbor] += 1
            if symmetric:
                glcm[neighbor, reference] += 1
    
    return glcm / glcm.sum()

def calculate_contrast(P):
    """Calculate contrast: Σ(i-j)²P(i,j)
    Measures intensity contrast between pixel and neighbor"""
    contrast = 0
    rows, cols = P.shape
    for i in range(rows):
        for j in range(cols):
            contrast += ((i-j)**2) * P[i,j]
    return contrast

def calculate_energy(P):
    """Calculate energy: Σ(P(i,j)²)
    Measures uniformity of texture"""
    return np.sum(P**2)

def calculate_homogeneity(P):
    """Calculate homogeneity: Σ P(i,j)/(1+(i-j)²)
    Measures closeness of element distribution to diagonal"""
    homogeneity = 0
    rows, cols = P.shape
    for i in range(rows):
        for j in range(cols):
            homogeneity += P[i,j] / (1 + (i-j)**2)
    return homogeneity

def calculate_correlation(P):
    """Calculate correlation: Σ((i-μi)(j-μj)P(i,j))/(σi σj)
    Measures linear dependency of gray levels"""
    rows, cols = P.shape
    i_indices = np.arange(rows).reshape(-1, 1)
    j_indices = np.arange(cols).reshape(1, -1)
    
    # Calculate means and standard deviations
    μi = np.sum(i_indices * P)
    μj = np.sum(j_indices * P)
    σi = np.sqrt(np.sum(((i_indices - μi)**2) * P))
    σj = np.sqrt(np.sum(((j_indices - μj)**2) * P))
    
    if σi == 0 or σj == 0:
        return 0
    
    correlation = 0
    for i in range(rows):
        for j in range(cols):
            correlation += ((i-μi)*(j-μj)*P[i,j])/(σi*σj)
    return correlation

# Calculate GLCM
glcm = create_glcm(image, symmetric=True)

# Create visualization
plt.figure(figsize=(20, 10))

# 1. Show original image
plt.subplot(231)
plt.imshow(image, cmap='gray')
plt.title('Original Image')
for i in range(4):
    for j in range(4):
        plt.text(j, i, str(image[i,j]), 
                ha='center', va='center', color='red')

# 2. Show GLCM
plt.subplot(232)
plt.imshow(glcm, cmap='viridis')
plt.title('GLCM (Probability Matrix)')
plt.colorbar()
for i in range(3):
    for j in range(3):
        plt.text(j, i, f'{glcm[i,j]:.2f}', 
                ha='center', va='center', color='white' if glcm[i,j] > 0.1 else 'black')

# 3. Contrast Calculation
plt.subplot(233)
contrast_matrix = np.zeros_like(glcm)
for i in range(3):
    for j in range(3):
        contrast_matrix[i,j] = ((i-j)**2) * glcm[i,j]
plt.imshow(contrast_matrix, cmap='viridis')
plt.title(f'Contrast Components\nSum = {calculate_contrast(glcm):.2f}')
plt.colorbar()
for i in range(3):
    for j in range(3):
        plt.text(j, i, f'({i-j})²*{glcm[i,j]:.2f}\n={contrast_matrix[i,j]:.2f}', 
                ha='center', va='center', color='white' if contrast_matrix[i,j] > 0.1 else 'black')

# 4. Energy Calculation
plt.subplot(234)
energy_matrix = glcm**2
plt.imshow(energy_matrix, cmap='viridis')
plt.title(f'Energy Components\nSum = {calculate_energy(glcm):.2f}')
plt.colorbar()
for i in range(3):
    for j in range(3):
        plt.text(j, i, f'({glcm[i,j]:.2f})²\n={energy_matrix[i,j]:.2f}', 
                ha='center', va='center', color='white' if energy_matrix[i,j] > 0.1 else 'black')

# 5. Homogeneity Calculation
plt.subplot(235)
homogeneity_matrix = np.zeros_like(glcm)
for i in range(3):
    for j in range(3):
        homogeneity_matrix[i,j] = glcm[i,j] / (1 + (i-j)**2)
plt.imshow(homogeneity_matrix, cmap='viridis')
plt.title(f'Homogeneity Components\nSum = {calculate_homogeneity(glcm):.2f}')
plt.colorbar()
for i in range(3):
    for j in range(3):
        plt.text(j, i, f'{glcm[i,j]:.2f}/(1+{(i-j)**2})\n={homogeneity_matrix[i,j]:.2f}', 
                ha='center', va='center', color='white' if homogeneity_matrix[i,j] > 0.1 else 'black')

# 6. Correlation Calculation
plt.subplot(236)
correlation_matrix = np.zeros_like(glcm)
rows, cols = glcm.shape
i_indices = np.arange(rows).reshape(-1, 1)
j_indices = np.arange(cols).reshape(1, -1)
μi = np.sum(i_indices * glcm)
μj = np.sum(j_indices * glcm)
σi = np.sqrt(np.sum(((i_indices - μi)**2) * glcm))
σj = np.sqrt(np.sum(((j_indices - μj)**2) * glcm))
for i in range(rows):
    for j in range(cols):
        correlation_matrix[i,j] = ((i-μi)*(j-μj)*glcm[i,j])/(σi*σj)
plt.imshow(correlation_matrix, cmap='viridis')
plt.title(f'Correlation Components\nSum = {calculate_correlation(glcm):.2f}')
plt.colorbar()
for i in range(3):
    for j in range(3):
        plt.text(j, i, f'({i-μi:.2f})*({j-μj:.2f})*{glcm[i,j]:.2f}\n={correlation_matrix[i,j]:.2f}', 
                ha='center', va='center', color='white' if correlation_matrix[i,j] > 0.1 else 'black')

plt.tight_layout()
plt.show()

# Print detailed explanation of each feature
print("\nGLCM Features Explanation:")
print("\n1. Contrast:", calculate_contrast(glcm))
print("   - Measures intensity difference between a pixel and its neighbor")
print("   - High when there are big differences in intensity")
print("   - Formula: Σ(i-j)²P(i,j)")
print("   - Weighted by square of difference")

print("\n2. Energy:", calculate_energy(glcm))
print("   - Measures uniformity of texture")
print("   - High when image is very uniform (few gray level transitions)")
print("   - Formula: ΣP(i,j)²")
print("   - Also known as Angular Second Moment")

print("\n3. Homogeneity:", calculate_homogeneity(glcm))
print("   - Measures closeness of element distribution to GLCM diagonal")
print("   - High when most pairs are along diagonal (similar values)")
print("   - Formula: ΣP(i,j)/(1+(i-j)²)")
print("   - Weighted inversely by difference squared")

print("\n4. Correlation:", calculate_correlation(glcm))
print("   - Measures linear dependency of gray levels")
print("   - High when image has a definite pattern")
print("   - Formula: Σ((i-μi)(j-μj)P(i,j))/(σi σj)")
print("   - Uses mean and standard deviation of row/column sums")
