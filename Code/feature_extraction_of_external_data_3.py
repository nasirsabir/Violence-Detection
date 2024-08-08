# Enter the external data file path in line 13

import os
import sys
import h5py
import cv2
import numpy as np
from random import shuffle
from tensorflow.keras.applications import VGG16
from tensorflow.keras.models import Model

# Constants
DATA_DIR = "/Users/nasir/Downloads/Archive 2/ExternalData" # Verisetinden ayıracağınız videoları tek klasörde topladıktan sonra buraya dosya yolunu giriniz
IMG_SIZE = 224
IMG_DIMS = (IMG_SIZE, IMG_SIZE)
IMAGES_PER_VIDEO = 20

def print_progress(current, total):
    percentage_complete = current / total
    message = f"\r- Progress: {percentage_complete:.1%}"
    sys.stdout.write(message)
    sys.stdout.flush()

def extract_frames(video_path):
    frames = []
    video_capture = cv2.VideoCapture(video_path)
    success, frame = video_capture.read()
    count = 0

    while count < IMAGES_PER_VIDEO and success:
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        resized_frame = cv2.resize(rgb_frame, IMG_DIMS, interpolation=cv2.INTER_CUBIC)
        frames.append(resized_frame)
        success, frame = video_capture.read()
        count += 1

    frames_array = np.array(frames)
    normalized_frames = (frames_array / 255.).astype(np.float16)
    return normalized_frames

def get_video_labels(data_dir):
    violent_videos = []
    nonviolent_videos = []

    for root_dir, _, files in os.walk(data_dir):
        for file in files:
            if file.startswith('fi'):
                violent_videos.append((file, [1, 0]))
            elif file.startswith('no'):
                nonviolent_videos.append((file, [0, 1]))

    return violent_videos, nonviolent_videos

def process_and_store_data(videos, data_dir, model, output_file):
    all_transfer_values = []
    all_labels = []
    all_filenames = []

    for index, (video_name, label) in enumerate(videos):
        video_path = os.path.join(data_dir, video_name)
        frames_batch = extract_frames(video_path)
        transfer_values = model.predict(frames_batch)
        repeated_labels = np.tile(label, (IMAGES_PER_VIDEO, 1))
        repeated_filenames = [video_name] * IMAGES_PER_VIDEO
        
        all_transfer_values.append(transfer_values)
        all_labels.append(repeated_labels)
        all_filenames.extend(repeated_filenames)
        
        print_progress(index + 1, len(videos))

    all_transfer_values = np.concatenate(all_transfer_values, axis=0)
    all_labels = np.concatenate(all_labels, axis=0)

    with h5py.File(output_file, 'w') as h5_file:
        h5_file.create_dataset('data', data=all_transfer_values)
        h5_file.create_dataset('labels', data=all_labels)
        h5_file.create_dataset('filenames', data=np.array(all_filenames, dtype='S'))

def main():
    violent_videos, nonviolent_videos = get_video_labels(DATA_DIR)

    all_videos = violent_videos + nonviolent_videos
    shuffle(all_videos)

    # Verify the distribution
    total_violent_count = len(violent_videos)
    total_nonviolent_count = len(nonviolent_videos)

    print(f"Total dataset: {total_violent_count} violent, {total_nonviolent_count} non-violent")

    # Output a small sample of filenames and labels to verify
    print("\nSample from the dataset:")
    for video_name, label in all_videos[:5]:
        print(f"{video_name}: {label}")

    vgg16_model = VGG16(include_top=True, weights='imagenet')
    feature_extractor = Model(inputs=vgg16_model.input, outputs=vgg16_model.get_layer('fc2').output)

    process_and_store_data(all_videos, DATA_DIR, feature_extractor, 'externaldata.h5')

if __name__ == "__main__":
    main()
