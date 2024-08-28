import os
import glob
import shutil
from tqdm import tqdm
import random
from collections import defaultdict
import tarfile
from config import DS_TYPE, DS_FORMAT
from api_utils import create_dataset, upload_dataset, update_dataset
import requests


def make_dir_get_classes(DATA_DIR):
    IMAGES_TRAIN_DIR = os.path.join(os.path.dirname(DATA_DIR), 'split', 'images_train')
    IMAGES_VAL_DIR = os.path.join(os.path.dirname(DATA_DIR), 'split', 'images_val')
    IMAGES_TEST_DIR = os.path.join(os.path.dirname(DATA_DIR), 'split', 'images_test')

    # Get class names from the directory structure
    class_names = [d for d in os.listdir(DATA_DIR) if os.path.isdir(os.path.join(DATA_DIR, d))]
    # Create directories if they don't exist
    for dir_path in [IMAGES_TRAIN_DIR, IMAGES_VAL_DIR, IMAGES_TEST_DIR]:
        print("Path:", dir_path)
        if not os.path.exists(dir_path):
            os.makedirs(dir_path)

    # Create classes.txt file
    with open(os.path.join(os.path.dirname(IMAGES_TRAIN_DIR), 'classes.txt'), 'w') as f:
        for class_name in class_names:
            f.write(f"{class_name}\n")

    print("Class names:", class_names)
    return class_names, IMAGES_TRAIN_DIR, IMAGES_VAL_DIR, IMAGES_TEST_DIR


def get_source_distribution(src_dir, class_names):
    class_distribution = {}
    total_images = 0
    for class_name in class_names:
        assert os.path.exists(os.path.join(src_dir, class_name))
        images = glob.glob(os.path.join(src_dir, class_name, '*.*'))
        class_distribution[class_name] = len(images)
        total_images += len(images)
    return class_distribution, total_images

# Get and print source distribution
def print_source_distribution(DATA_DIR, class_names):
    source_distribution, source_total = get_source_distribution(DATA_DIR, class_names)
    print("\nOriginal Source Dataset Distribution:")
    print(f"{'Class':<15} {'Count':<10} {'Percentage':<10}")
    print("-" * 35)
    for class_name, count in source_distribution.items():
        percentage = count / source_total * 100
        print(f"{class_name:<15} {count:<10} {percentage:.2f}%")
    print(f"{'Total':<15} {source_total:<10} 100.00%")


# Function to split and copy images
def split_and_copy(class_names, src_dir, train_dir, val_dir, test_dir, train_ratio=0.7, val_ratio=0.15):
    class_distribution = defaultdict(lambda: defaultdict(int))
    total_images = 0

    for class_name in class_names:
        images = glob.glob(os.path.join(src_dir, class_name, '*.*'))
        total_images += len(images)
        for dir_path in [train_dir, val_dir, test_dir]:
            os.makedirs(os.path.join(dir_path, class_name), exist_ok=True)

    for class_name in tqdm(class_names, desc="Splitting dataset"):
        images = glob.glob(os.path.join(src_dir, class_name, '*.*'))
        random.shuffle(images)

        for img in images:
            rand_val = random.random()
            if rand_val < train_ratio:
                dest_dir = train_dir
                split_name = 'train'
            elif rand_val < train_ratio + val_ratio:
                dest_dir = val_dir
                split_name = 'val'
            else:
                dest_dir = test_dir
                split_name = 'test'

            shutil.copy2(img, os.path.join(dest_dir, class_name))
            class_distribution[class_name][split_name] += 1

    return class_distribution, total_images


def prepare_dataset(class_names, source_dir, train_dir, val_dir, test_dir):
    # Split the dataset
    class_distribution, total_images = split_and_copy(class_names, source_dir, train_dir, val_dir, test_dir)

    print("\nClass Distribution:")
    print(f"{'Class':<15} {'Train':<20} {'Val':<20} {'Test':<20} {'Total':<10}")
    print("-" * 85)

    empty_classes = []

    for class_name in class_names:
        train_count = class_distribution[class_name]['train']
        val_count = class_distribution[class_name]['val']
        test_count = class_distribution[class_name]['test']
        total_count = train_count + val_count + test_count

        if total_count == 0:
            empty_classes.append(class_name)
            print(f"{class_name:<15} No images found")
        else:
            print(f"{class_name:<15} "
                  f"{train_count:<10} ({train_count / total_count:.1%}) "
                  f"{val_count:<10} ({val_count / total_count:.1%}) "
                  f"{test_count:<10} ({test_count / total_count:.1%}) "
                  f"{total_count:<10}")

    # Count images in each split
    train_count = sum(class_distribution[c]['train'] for c in class_names)
    val_count = sum(class_distribution[c]['val'] for c in class_names)
    test_count = sum(class_distribution[c]['test'] for c in class_names)

    print(f"\nTotal images: Train: {train_count} ({train_count / total_images:.1%}), "
          f"Validation: {val_count} ({val_count / total_images:.1%}), "
          f"Test: {test_count} ({test_count / total_images:.1%})")

    if empty_classes:
        print("\nWarning: The following classes have no images:")
        for class_name in empty_classes:
            print(f"- {class_name}")

    print('\nDataset preparation completed.')

def create_tar_gz(source_dir, output_file, files_to_include):
    with tarfile.open(output_file, "w:gz") as tar:
        for file in files_to_include:
            tar.add(os.path.join(source_dir, file), arcname=file)

def process_tar(main_data_dir):
    # Define other necessary paths
    split_dir = os.path.join(main_data_dir, "split")
    split_tar_dir = os.path.join(main_data_dir, "split_tar")

    # Create the split_tar directory if it doesn't exist
    os.makedirs(split_tar_dir, exist_ok=True)

    # Create tar.gz files
    create_tar_gz(split_dir, os.path.join(split_tar_dir, "classification_train.tar.gz"), ["images_train", "classes.txt"])
    create_tar_gz(split_dir, os.path.join(split_tar_dir, "classification_val.tar.gz"), ["images_val", "classes.txt"])
    create_tar_gz(split_dir, os.path.join(split_tar_dir, "classification_test.tar.gz"), ["images_test", "classes.txt"])

    # Define dataset paths
    train_dataset_path = os.path.join(split_tar_dir, "classification_train.tar.gz")
    eval_dataset_path = os.path.join(split_tar_dir, "classification_val.tar.gz")
    test_dataset_path = os.path.join(split_tar_dir, "classification_test.tar.gz")

    print(f"Train dataset path: {train_dataset_path}")
    print(f"Evaluation dataset path: {eval_dataset_path}")
    print(f"Test dataset path: {test_dataset_path}")

    return train_dataset_path, eval_dataset_path, test_dataset_path


def create_and_upload_datasets(base_url, headers, train_dataset_path, eval_dataset_path, test_dataset_path, model_name = "TODO"):

    # Create and upload train dataset
    train_dataset_id = create_dataset(base_url, headers)
    # print(train_dataset_id, train_dataset_path)
    update_dataset(base_url, headers, train_dataset_id, "Siemens Train Dataset", "My train dataset")
    upload_dataset(base_url, headers, train_dataset_id, train_dataset_path)

    # Create and upload eval dataset
    eval_dataset_id = create_dataset(base_url, headers)
    update_dataset(base_url, headers, eval_dataset_id, "Siemens Eval dataset", "Eval dataset")
    upload_dataset(base_url, headers, eval_dataset_id, eval_dataset_path)

    # Create and upload test dataset
    test_dataset_id = create_dataset(base_url, headers)
    upload_dataset(base_url, headers, test_dataset_id, test_dataset_path)

    # # List all datasets
    # endpoint = f"{base_url}/datasets"
    # response = requests.get(endpoint, headers=headers)
    # assert response.status_code in (200, 201)
    # print("id\t\t\t\t\t type\t\t\t format\t\t name")
    # for rsp in response.json():
    #     print(rsp["id"], "\t", rsp["type"], "\t", rsp["format"], "\t\t", rsp["name"])

    return train_dataset_id, eval_dataset_id, test_dataset_id
