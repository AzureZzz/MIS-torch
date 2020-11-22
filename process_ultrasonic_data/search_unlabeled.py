import argparse
import os
from shutil import copyfile

import re


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--path", default='D:/DeskTop/all', help="dir path")
    args = parser.parse_args()

    unlabeled_dir = os.path.join(args.path, 'unlabeled')
    if not os.path.exists(unlabeled_dir):
        os.mkdir(unlabeled_dir)
    file_name_list = os.listdir(args.path)
    json_name_list = [x for x in file_name_list if re.match('.+\.json$', x)]
    image_name_list = [x for x in file_name_list if re.match('.+\.bmp|.+\.png|.+\.jpg$', x)]
    image_name_list = [x for x in image_name_list if x[1] != '1']
    print(f'json number:{len(json_name_list)}, image number: {len(image_name_list)}')
    unlabeled_list = []
    for image in image_name_list:
        if image[1] != '1' and image.split('.')[0]+'.json' not in json_name_list:
            unlabeled_list.append(image)
    for name in unlabeled_list:
        src = os.path.join(args.path, name)
        copyfile(src, os.path.join(unlabeled_dir, name))
        os.remove(src)


if __name__ == '__main__':
    main()