"""
Training pipeline:
- Reads candidates.csv
- Extracts 50x50 sub-images from .mhd around nodules
- Normalizes HU to [-1000, 400] and saves grayscale JPGs
- Converts to RGB arrays for DenseNet
- Augments positives 5x with realistic CT noise
- Trains DenseNet121 with callbacks
- Saves model, training history, and test split for later evaluation
"""

import os, glob
import numpy as np
import pandas as pd
from PIL import Image
import SimpleITK as sitk
from tqdm import tqdm
import tensorflow as tf

from sklearn.model_selection import train_test_split
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint

from cnn_model import CNNModel

# Set random seeds for reproducibility
np.random.seed(42)
tf.random.set_seed(42)

# Data I/O and CT utilities

class CTScan:
    """
    Reads a .mhd volume and provides HU-normalized 2D sub-image around given (x,y,z)
    """

    def __init__(self, filename=None, coords=None):
        self.filename = filename
        self.coords = coords
        self.ds = None
        self.image = None
        self._read_mhd_image()

    def _read_mhd_image(self):
        path = glob.glob(os.path.join('src/data/raw/**', self.filename + '.mhd'), recursive=True)[0]
        self.ds = sitk.ReadImage(path)
        self.image = sitk.GetArrayFromImage(self.ds)

    def _get_voxel_coords(self):
        # World (mm) -> voxel index conversion
        origin = self.ds.GetOrigin()
        spacing = self.ds.GetSpacing()
        voxel = [abs(self.coords[j] - origin[j]) / spacing[j] for j in range(3)]
        return tuple(map(int, voxel))

    def get_subimage(self, width=50):
        # Extract 2D slice (z, y, x) as width x width around the nodule center
        x, y, z = self._get_voxel_coords()
        half = width // 2
        return self.image[z, y-half:y+half, x-half:x+half]

    @staticmethod
    def normalize_hu(npzarray, minHU=-1000., maxHU=400.):
        # Map HU to [0,1]
        npzarray = (npzarray - minHU) / (maxHU - minHU)
        return np.clip(npzarray, 0, 1)

    def save_image(self, out_path, width=50):
        # Save grayscale sub-image as 8-bit JPG
        sub = self.get_subimage(width)
        sub = self.normalize_hu(sub)
        sub = (sub * 255).astype(np.uint8)
        Image.fromarray(sub).convert('L').save(out_path)


def save_dataset(df, out_dir):
    """
    Materialize dataset as images on disk for speed/repro
    """
    os.makedirs(out_dir, exist_ok=True)
    for idx, row in tqdm(df.iterrows(), total=len(df), desc=f"Saving {out_dir}"):
        scan = CTScan(row['seriesuid'], [row['coordX'], row['coordY'], row['coordZ']])
        scan.save_image(os.path.join(out_dir, f'image_{idx}.jpg'))


def load_images(df, img_dir):
    """
    Read saved grayscale JPGs, convert to RGB (3-ch) and scale to [0,1]
    """
    images = []
    for idx in tqdm(df.index, desc=f"Loading {img_dir}"):
        p = os.path.join(img_dir, f'image_{idx}.jpg')
        img = Image.open(p).convert('RGB')        # repeat channels
        images.append(np.array(img) / 255.0)
    return np.array(images)


def main():
    # Load and balance metadata
    candidates = pd.read_csv('src/data/candidates.csv')

    positives = candidates[candidates['class'] == 1]
    negatives = candidates[candidates['class'] == 0]

    # 2:1 negative:positive to keep more negatives but not overwhelm
    neg_sample = negatives.sample(n=len(positives) * 2, random_state=42)
    balanced_df = pd.concat([positives, neg_sample]).sample(frac=1, random_state=42)

    # Train/Val/Test split (70/15/15 stratified)
    X = balanced_df.iloc[:, :-1]
    y = balanced_df.iloc[:, -1]
    X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.3, stratify=y, random_state=42)
    X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, stratify=y_temp, random_state=42)

    # Build on-disk image sets
    save_dataset(X_train, 'src/data/train/')
    save_dataset(X_val,   'src/data/val/')
    save_dataset(X_test,  'src/data/test/')

    # Load into memory as RGB
    X_train_images = load_images(X_train, 'src/data/train/')
    X_val_images   = load_images(X_val,   'src/data/val/')
    X_test_images  = load_images(X_test,  'src/data/test/')

    y_train = y_train.values.reshape(-1, 1)
    y_val   = y_val.values.reshape(-1, 1)
    y_test  = y_test.values.reshape(-1, 1)

    # Sanity prints (shapes)
    print(f"Training images: {X_train_images.shape}, labels: {y_train.shape}")
    print(f"Validation images: {X_val_images.shape}, labels: {y_val.shape}")
    print(f"Test images: {X_test_images.shape}, labels: {y_test.shape}")

    # Augment positives 5x
    med_aug = ImageDataGenerator(
        rotation_range=15,
        width_shift_range=0.1,
        height_shift_range=0.1,
        shear_range=0.1,
        zoom_range=0.1,
        horizontal_flip=True,
        fill_mode='reflect',
        preprocessing_function=lambda x: x + np.random.normal(0, 0.03, x.shape)  # CT noise
    )

    X_pos = X_train_images[y_train.flatten() == 1]
    augmented = []
    for _ in range(5):               # 5x augmentation
        for img in X_pos:
            augmented.append(med_aug.random_transform(img))

    X_train_aug = np.concatenate([X_train_images, np.array(augmented)], axis=0)
    y_train_aug = np.concatenate([y_train, np.ones((len(augmented), 1))], axis=0)

    print(f"After augmentation: {X_train_aug.shape[0]} training samples")

    # Build and train model
    cnn = CNNModel(input_shape=(50, 50, 3), freeze_upto=100, l2=0.01, dropout=0.5)
    model = cnn.build()

    callbacks = [
        EarlyStopping(monitor='val_auc', patience=15, mode='max', restore_best_weights=True),
        ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=5, min_lr=1e-6),
        ModelCheckpoint('best_densenet.keras', save_best_only=True)
    ]

    history = model.fit(
        X_train_aug, y_train_aug,
        validation_data=(X_val_images, y_val),
        epochs=50,
        batch_size=32,
        callbacks=callbacks,
        verbose=1
    )

    # Persist artifacts for downstream steps
    np.save('training_history.npy', history.history)
    np.savez('test_data.npz', X_test=X_test_images, y_test=y_test)

    print("Training complete. Saved:")
    print("   - best_densenet.keras")
    print("   - training_history.npy")
    print("   - test_data.npz")

if __name__ == "__main__":
    main()