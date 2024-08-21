import os
import glob
import shutil
import random
from tqdm import tqdm
import tarfile
from collections import defaultdict


def split_tar_file(input_tar_path, output_dir, max_split_size=0.2 * 1024 * 1024 * 1024):
    os.makedirs(output_dir, exist_ok=True)

    with tarfile.open(input_tar_path, 'r') as original_tar:
        members = original_tar.getmembers()
        current_split_size = 0
        current_split_number = 0
        current_split_name = os.path.join(output_dir, f'smaller_file_{current_split_number}.tar')

        with tarfile.open(current_split_name, 'w') as split_tar:
            for member in members:
                if current_split_size + member.size <= max_split_size:
                    split_tar.addfile(member, original_tar.extractfile(member))
                    current_split_size += member.size
                else:
                    split_tar.close()
                    current_split_number += 1
                    current_split_name = os.path.join(output_dir, f'smaller_file_{current_split_number}.tar')
                    current_split_size = 0
                    split_tar = tarfile.open(current_split_name, 'w')
                    split_tar.addfile(member, original_tar.extractfile(member))
                    current_split_size += member.size


def get_source_distribution(src_dir):
    class_distribution = {}
    total_images = 0
    for class_name in os.listdir(src_dir):
        if os.path.isdir(os.path.join(src_dir, class_name)):
            images = glob.glob(os.path.join(src_dir, class_name, '*.*'))
            class_distribution[class_name] = len(images)
            total_images += len(images)
    return class_distribution, total_images


def split_and_copy(src_dir, train_dir, val_dir, test_dir, train_ratio=0.7, val_ratio=0.15):
    class_distribution = defaultdict(lambda: defaultdict(int))
    total_images = 0

    class_names = [d for d in os.listdir(src_dir) if os.path.isdir(os.path.join(src_dir, d))]
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


def prepare_dataset(src_dir, train_dir, val_dir, test_dir):
    class_names = [d for d in os.listdir(src_dir) if os.path.isdir(os.path.join(src_dir, d))]

    for dir_path in [train_dir, val_dir, test_dir]:
        os.makedirs(dir_path, exist_ok=True)

    with open(os.path.join(os.path.dirname(train_dir), 'classes.txt'), 'w') as f:
        for class_name in class_names:
            f.write(f"{class_name}\n")

    source_distribution, source_total = get_source_distribution(src_dir)
    print("\nOriginal Source Dataset Distribution:")
    print(f"{'Class':<15} {'Count':<10} {'Percentage':<10}")
    print("-" * 35)
    for class_name, count in source_distribution.items():
        percentage = count / source_total * 100
        print(f"{class_name:<15} {count:<10} {percentage:.2f}%")
    print(f"{'Total':<15} {source_total:<10} 100.00%")

    class_distribution, total_images = split_and_copy(src_dir, train_dir, val_dir, test_dir)

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