import os
from PIL import Image


def process_directory(src_dir, dst_dir, target_resolution):
    """
    For every .jpg file in the source directory equal to or larger than the target resolution, downscales the image as
    much as possible without going below the crop resolution, crops the image (center-preserving) to the given
    dimensions, and saves it to the destination directory.

    :param src_dir: Path to source directory containing .jpg files only.
    :param dst_dir: Path to destination directory where downscaled and cropped images will be written to.
    :param target_resolution: (w, h) where w = "Width of the outputted images", h = "Height of the outputted images".
    :return: n: Number of images written to the destination directory.
    """
    w_, h_ = target_resolution
    n = 0
    for img_name in os.listdir(src_dir):
        src = "%s/%s" % (src_dir, img_name)
        dst = "%s/%s.jpg" % (dst_dir, n)
        img = Image.open(src)
        w, h = img.size
        if w >= w_ and h >= h_:
            f = w / w_ if w < h else h / h_
            w, h = (int(w/f), int(h/f))
            img_downscaled = img.resize((w, h))
            img_cropped = img_downscaled.crop((w / 2 - w_ / 2, h / 2 - h_ / 2,
                                               w / 2 + w_ / 2, h / 2 + h_ / 2))
            img_cropped.save(dst)
            n += 1
    return n
