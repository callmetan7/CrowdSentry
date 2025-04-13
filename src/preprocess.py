import os
import numpy as np
import scipy.io
import scipy.ndimage
import cv2
from tqdm import tqdm


def generate_density_map(image_shape, points, sigma=15):
    """Generate density map for given points."""
    density_map = np.zeros(image_shape, dtype=np.float32)
    for point in points:
        x, y = min(int(point[0]), image_shape[1] - 1), min(int(point[1]), image_shape[0] - 1)
        density_map[y, x] += 1
    density_map = scipy.ndimage.gaussian_filter(density_map, sigma=sigma)
    return density_map


def preprocess_data(part):
    data_path = f"data/ShanghaiTech/part_{part}/"
    save_path = f"data/processed/part_{part}/"
    os.makedirs(save_path, exist_ok=True)

    for phase in ['train', 'test']:
        images_dir = os.path.join(data_path, f"{phase}_data/images")
        # Updated path to match 'ground-truth' folder
        gt_dir = os.path.join(data_path, f"{phase}_data/ground-truth")
        save_images_dir = os.path.join(save_path, phase, "images")
        save_density_dir = os.path.join(save_path, phase, "density_maps")

        os.makedirs(save_images_dir, exist_ok=True)
        os.makedirs(save_density_dir, exist_ok=True)

        for img_name in tqdm(os.listdir(images_dir)):
            img_path = os.path.normpath(os.path.join(images_dir, img_name))
            mat_path = os.path.normpath(os.path.join(gt_dir, img_name.replace('IMG', 'GT_IMG').replace('.jpg', '.mat')))

            # Debugging: Print paths
            print(f"Image Path: {img_path}")
            print(f"MAT Path: {mat_path}")

            # Skip missing .mat files
            if not os.path.exists(mat_path):
                print(f"Missing .mat file for: {img_name}")
                continue

            # Load image and ground truth
            image = cv2.imread(img_path)
            mat = scipy.io.loadmat(mat_path)
            points = mat["image_info"][0, 0][0, 0][0]  # Coordinates of people

            # Generate density map
            density_map = generate_density_map(image.shape[:2], points)

            # Save preprocessed data
            cv2.imwrite(os.path.join(save_images_dir, img_name), image)
            np.save(os.path.join(save_density_dir, img_name.replace('.jpg', '.npy')), density_map)


# Example Usage: preprocess_data('A') or preprocess_data('B')
