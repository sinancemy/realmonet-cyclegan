import os
from PIL import Image


def filter_and_crop(src_dir, dst_dir, target_w, target_h):
    """
    For every .jpg file in the source directory, crops the image (center-preserving) to the given dimensions if the
    dimensions of the .jpg file are greater than the target dimensions, and saves it to the destination directory.

    :param src_dir: Path to source directory containing .jpg files only.
    :param dst_dir: Path to destination directory where filtered and cropped images will be written.
    :param target_w: Width of the outputted images.
    :param target_h: Height of the outputted images.
    :return:
    """
    n = 0
    for img_name in os.listdir(src_dir):
        src = "%s/%s" % (src_dir, img_name)
        dst = "%s/%s.jpg" % (dst_dir, n)
        img = Image.open(src)
        w, h = img.size
        if w >= target_w and h >= target_h:
            img_cropped = img.crop((w / 2 - target_w / 2, h / 2 - target_h / 2,
                                    w / 2 + target_w / 2, h / 2 + target_h / 2))
            img_cropped.save(dst)
            n += 1
