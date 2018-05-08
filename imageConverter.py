# AUTHOR: Unknown
# ADAPTED BY: @toderesa97

from PIL import Image
from PIL.Image import ANTIALIAS
import os

from os import path


def resizeImage(infile, output_dir="", folderName="",  size=(140, 80), verbose=True,
                appended_watermark="_resized",extension="JPEG", grayScale=True):

    outfile = os.path.splitext(infile)
    ext = outfile[len(outfile) - 1]
    filename = outfile[len(outfile) - 2] + appended_watermark + ext
    filename = filename.split("/")
    filename = filename[len(filename) - 1]
    max_width, max_height = size
    if infile != outfile:
        try:
            im = Image.open(infile)
            src_width, src_height = im.size
            src_ratio = float(src_width) / float(src_height)
            dst_width, dst_height = max_width, max_height
            dst_ratio = float(dst_width) / float(dst_height)

            if dst_ratio < src_ratio:
                crop_height = src_height
                crop_width = crop_height * dst_ratio
                x_offset = float(src_width - crop_width) / 2
                y_offset = 0
            else:
                crop_width = src_width
                crop_height = crop_width / dst_ratio
                x_offset = 0
                y_offset = float(src_height - crop_height) / 3
            img = im.crop((x_offset, y_offset, x_offset + int(crop_width), y_offset + int(crop_height)))
            img = img.resize((dst_width, dst_height), ANTIALIAS)
            if grayScale:
                img = img.convert("L")
            if verbose:
                print("SAVING -> " + output_dir + "/" + filename)
            img.save(output_dir + "/" + folderName +"/"+ filename, extension)
        except IOError:
            if verbose:
                print("FATAL ERROR WHILST OPENING IMAGE OR SAVING")


# This class takes from a list of images where the dimensions are not the same
# and convert all images to an standard size (140x80) and 1 channel (8-bit color)
# Converted images are placed within IMGR/{type}. Just create the directories if they did not exist
# Code explanation
#
#   for file in os.listdir(<actual_data_set_of_non_standard_images>):
#       resizeImage(<image>, <destination_folder>)
#
#

def convertSet(fromDirectory, toDirectory, folderName, size=(140, 80), verbose=True,
               appended_watermark="_resized", extension="JPEG", grayScale=True):
    if not os.path.exists(toDirectory):
        directory_to_create = path.dirname(path.abspath(__file__), ) + "/" + toDirectory;
        if verbose:
            print("Creating directory "+ directory_to_create)
        os.makedirs(directory_to_create)
    if not os.path.exists(toDirectory + "/" + folderName):
        folderName_to_create = path.dirname(path.abspath(__file__), ) + "/" + toDirectory + "/" + folderName;
        if verbose:
            print("Creating folder "+ folderName_to_create)
        os.makedirs(folderName_to_create)
    for file in os.listdir(fromDirectory):
        resizeImage(fromDirectory + "/" + file, toDirectory, folderName, size, verbose, appended_watermark, extension, grayScale)


# CAVEAT : <from directory> must obviously exist. The other one is created if it does not exist
size = (140, 80)
waterMark = "_resized"
ext = "JPEG"
convertSet("DATASET/245.windmill", "IMGR", "windmill", size, False, waterMark, ext, False)
convertSet("DATASET/246.wine-bottle", "IMGR", "wine", size, False, waterMark, ext, False)
convertSet("DATASET/256.toad", "IMGR", "toad", size, False, waterMark, ext, False)
