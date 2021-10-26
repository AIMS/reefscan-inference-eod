from PIL import Image
import math
import glob
import csv
import os
from time import time


def make_csv(dir_name, patch_folder):
    tic = time()

    photos = find_photos(dir_name)
    grid = grid_for_images(photos, len(dir_name), patch_folder)
    result = write_csv(grid, dir_name)
    print('Completed in {}'.format(convert_time(time() - tic)))
    return result


def write_csv(grid, dir_name):
    keys = grid[0].keys()
    csv_name = dir_name + '/photos.csv'
    with open(csv_name, 'w', newline='') as output_file:
        dict_writer = csv.DictWriter(output_file, keys, quoting=csv.QUOTE_NONNUMERIC)
        dict_writer.writeheader()
        dict_writer.writerows(grid)
    return csv_name


def find_photos(dir_name):
    pattern = dir_name + "/**/*.jpg"
    return glob.glob(pattern, recursive=True)


def grid_for_images(photos, len_base_folder, patch_dir):
    result = []
    i=0
    l = len(photos)
    if os.path.isdir(patch_dir):
        for filename in os.listdir(patch_dir):
            file_path = os.path.join(patch_dir, filename)
            os.unlink(file_path)

    for photo in photos:
        if i % 100 == 0:
            print(f"grid for photo {i} of {l}")
        i += 1
        image_grid = grid_for_image(photo, len_base_folder, patch_dir)
        result.extend(image_grid)
    return result


def grid_for_image(image_name, len_base_folder, patch_dir, cut_devisor=8):
    with Image.open(image_name) as img:
        width, height = img.size

        patch_size = height // cut_devisor
        patches_across = width // patch_size
        patches_down = cut_devisor

        hor_margin = (width - (patches_across * patch_size)) // 2

        result = []

        this_patch_dir = f"{patch_dir}/{image_name[len_base_folder:]}.patches"
        os.makedirs(this_patch_dir)

        for i in range(patches_across):
            for j in range(patches_down):
                patch_file = f"{this_patch_dir}/x{i}y{j}.jpg"
                x = math.floor((i+0.5) * patch_size) + hor_margin
                y = math.floor((j+0.5) * patch_size)
                result.append({"image_path": image_name, "patch_file": patch_file, "pointx": x, "pointy": y})

                patch = load_image_and_crop_inf(img, 256, 256, x, y)

                patch.save(patch_file)

    return result


def load_image_and_crop_inf(img, crop_width, crop_height, point_x, point_y, cut_divisor=8):
    width, height = img.size
    cut_width = int(height/cut_divisor)
    cut_height = int(height/cut_divisor)

    img = cut_patch(img, cut_width, cut_height, point_x, point_y)
    img = img.resize((crop_width, crop_height), Image.NEAREST)

    return img


def cut_patch(image, patch_width, patch_height, x, y):

    width, height = image.size
    dimensions = get_rect_dimensions_pixels(width, height, patch_width, patch_height, x, y)
    new_image = image.crop(dimensions)
    return new_image


def get_rect_dimensions_pixels(width, height, patchwidth, patchheight, pointx, pointy):
    return [int((pointx)-(patchwidth/2)), int((pointy)-(patchheight/2)),
            int((pointx)+(patchwidth/2)), int((pointy)+(patchheight/2))]


def convert_time(seconds):
    mins, sec = divmod(seconds, 60)
    hour, mins = divmod(mins, 60)
    if hour > 0:
        return "{:.0f} hour, {:.0f} minutes".format(hour, mins)
    elif mins > 0:
        return "{:.0f} minutes".format(mins)
    else:
        return "{:.0f} seconds".format(sec)


make_csv("C:/aims/reef-scan/images/20210727_233059_Seq02", "C:/aims/reef-scan/patches/20210727_233059_Seq02")

