"""
Simple test script to visualize hex files
Run this to quickly check your image exports
"""

import numpy as np
import matplotlib.pyplot as plt
import os
import sys

# ============================================================
# CONFIGURATION
# ============================================================

DATAWIDTH = 16
FRAC = 13
SCALE = 1 << FRAC

def from_fixed_16(fixed_val):
    """Convert fixed-point back to float"""
    if fixed_val > 32767:
        fixed_val = fixed_val - 65536
    return float(fixed_val) / SCALE

def read_hex_file(filename):
    """Read a hex file and return array of values"""
    try:
        values = []
        with open(filename, 'r') as f:
            for line in f:
                hex_val = int(line.strip(), 16)
                values.append(hex_val)
        return np.array(values, dtype=np.uint16)
    except FileNotFoundError:
        print(f"❌ ERROR: File not found: {filename}")
        return None
    except Exception as e:
        print(f"❌ ERROR reading {filename}: {e}")
        return None

def test_single_image(image_path):
    """Test loading and displaying a single image"""
    print(f"\nTesting: {image_path}")
    print("-" * 60)
    
    hex_values = read_hex_file(image_path)
    if hex_values is None:
        return False
    
    print(f"✓ Successfully read {len(hex_values)} values")
    
    if len(hex_values) != 784:
        print(f"❌ ERROR: Expected 784 values (28x28), got {len(hex_values)}")
        return False
    
    # Convert to float
    pixels = [from_fixed_16(val) for val in hex_values]
    image = np.array(pixels).reshape(28, 28)
    
    print(f"✓ Reshaped to 28x28")
    print(f"  Min value: {image.min():.6f}")
    print(f"  Max value: {image.max():.6f}")
    print(f"  Mean value: {image.mean():.6f}")
    
    # Display
    plt.figure(figsize=(6, 6))
    plt.imshow(image, cmap='gray')
    plt.colorbar(label='Pixel Value')
    plt.title(f'{os.path.basename(image_path)}')
    plt.tight_layout()
    plt.show()
    
    return True

def test_directory(directory='test_images'):
    """Test all images in a directory"""
    print(f"\nScanning directory: {directory}")
    print("=" * 60)
    
    if not os.path.exists(directory):
        print(f"❌ Directory not found: {directory}")
        print(f"   Current working directory: {os.getcwd()}")
        print(f"   Files in current directory:")
        for f in os.listdir('.'):
            print(f"     - {f}")
        return False
    
    # Find image files
    image_files = sorted([f for f in os.listdir(directory) 
                         if f.startswith('image_') and f.endswith('.hex')])
    
    if not image_files:
        print(f"❌ No image_*.hex files found in {directory}")
        print(f"   Files found:")
        for f in os.listdir(directory):
            print(f"     - {f}")
        return False
    
    print(f"✓ Found {len(image_files)} image files")
    
    # Create grid
    num_images = len(image_files)
    cols = min(5, num_images)
    rows = (num_images + cols - 1) // cols
    
    fig, axes = plt.subplots(rows, cols, figsize=(3*cols, 3*rows))
    if num_images == 1:
        axes = np.array([axes])
    else:
        axes = axes.flatten()
    
    success_count = 0
    
    for idx, img_file in enumerate(image_files):
        img_path = os.path.join(directory, img_file)
        
        # Read image
        hex_values = read_hex_file(img_path)
        if hex_values is None or len(hex_values) != 784:
            axes[idx].text(0.5, 0.5, 'ERROR', ha='center', va='center')
            axes[idx].set_title(f'#{idx} FAILED', color='red')
            axes[idx].axis('off')
            continue
        
        # Convert and reshape
        pixels = [from_fixed_16(val) for val in hex_values]
        image = np.array(pixels).reshape(28, 28)
        
        # Try to read label
        label_file = img_file.replace('image_', 'label_').replace('.hex', '.txt')
        label_path = os.path.join(directory, label_file)
        label = None
        if os.path.exists(label_path):
            try:
                with open(label_path, 'r') as f:
                    content = f.read().strip()
                    if content.isdigit():
                        label = int(content)
            except:
                pass
        
        # Display
        axes[idx].imshow(image, cmap='gray')
        axes[idx].axis('off')
        
        title = f'#{idx}'
        if label is not None:
            title += f' Label:{label}'
        axes[idx].set_title(title)
        
        success_count += 1
    
    # Hide unused subplots
    for idx in range(num_images, len(axes)):
        axes[idx].axis('off')
    
    plt.suptitle(f'Images from {directory}', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.show()
    
    print(f"\n✓ Successfully visualized {success_count}/{num_images} images")
    return success_count == num_images

def test_weights(weight_path='weights_L1_N0.hex'):
    """Test loading a weight file"""
    print(f"\nTesting weights: {weight_path}")
    print("-" * 60)
    
    hex_values = read_hex_file(weight_path)
    if hex_values is None:
        return False
    
    print(f"✓ Successfully read {len(hex_values)} weight values")
    
    # Convert to float
    weights = [from_fixed_16(val) for val in hex_values]
    weights = np.array(weights)
    
    print(f"  Min: {weights.min():.6f}")
    print(f"  Max: {weights.max():.6f}")
    print(f"  Mean: {weights.mean():.6f}")
    print(f"  Std: {weights.std():.6f}")
    
    # Visualize
    if len(weights) == 784:
        # Layer 1 - can reshape to 28x28
        weights_img = weights.reshape(28, 28)
        plt.figure(figsize=(6, 6))
        plt.imshow(weights_img, cmap='RdBu', 
                  vmin=-np.abs(weights_img).max(), 
                  vmax=np.abs(weights_img).max())
        plt.colorbar(label='Weight Value')
        plt.title(f'{os.path.basename(weight_path)}\n(reshaped to 28x28)')
    else:
        # Layer 2 or other - bar chart
        plt.figure(figsize=(10, 4))
        plt.bar(range(len(weights)), weights)
        plt.xlabel('Weight Index')
        plt.ylabel('Weight Value')
        plt.title(f'{os.path.basename(weight_path)}')
        plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()
    
    return True

def main():
    """Main test function"""
    print("=" * 60)
    print("HEX FILE VISUALIZATION TESTER")
    print("=" * 60)
    print(f"Current directory: {os.getcwd()}")
    print()
    
    # Check command line arguments
    if len(sys.argv) > 1:
        test_path = sys.argv[1]
        
        if os.path.isdir(test_path):
            print(f"Testing directory: {test_path}")
            test_directory(test_path)
        elif os.path.isfile(test_path):
            print(f"Testing single file: {test_path}")
            if 'image' in test_path:
                test_single_image(test_path)
            elif 'weight' in test_path:
                test_weights(test_path)
            else:
                print("❌ Unknown file type (expected 'image_*.hex' or 'weights_*.hex')")
        else:
            print(f"❌ Path not found: {test_path}")
    else:
        # Default: test common locations
        print("No path specified, testing default locations...\n")
        
        # Test 1: test_images directory
        if os.path.exists('test_images'):
            test_directory('test_images')
        else:
            print("❌ test_images/ directory not found")
        
        # Test 2: single image in current directory
        if os.path.exists('input.hex'):
            print("\n" + "=" * 60)
            test_single_image('input.hex')
        
        # Test 3: weights
        if os.path.exists('weights_L1_N0.hex'):
            print("\n" + "=" * 60)
            test_weights('weights_L1_N0.hex')
        
        print("\n" + "=" * 60)
        print("USAGE:")
        print("=" * 60)
        print("  python test_visualization.py                    # Test default locations")
        print("  python test_visualization.py test_images/       # Test specific directory")
        print("  python test_visualization.py image_0.hex        # Test single image")
        print("  python test_visualization.py weights_L1_N0.hex  # Test weight file")

if __name__ == "__main__":
    main()