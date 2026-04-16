# read and resize the image from BioSR dataset

import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from read_mrc import read_mrc


def normalize_image(image):
    image_norm = (image-np.min(image))/(np.max(image)-np.min(image))
    image_norm = np.uint8(image_norm*255)
    return image_norm


# ccps_data = read_mrc('G:/BioSR_data/CCPs/GT_all.mrc')[1]
# er_data = read_mrc('G:/BioSR_data/ER/GT_all.mrc')[1]
# f_actin_data = read_mrc('G:/BioSR_data/F-actin/GT_all_a.mrc')[1]
microtubules_data = read_mrc('G:/BioSR_data/Microtubules/GT_all.mrc')[1]

range_x = 1004
range_y = 1004
dx = 0.0313
dx_new = dx
[range_x_new, range_y_new] = np.uint([range_x*dx/dx_new, range_y*dx/dx_new])

# n_ccps = ccps_data.shape[2]
# for i0 in range(n_ccps):
#     ccps_image = np.resize(ccps_data[:, :, i0], [range_x_new, range_y_new])
#     ccps_sub_image1 = normalize_image(ccps_image[:512, :512])
#     ccps_sub_image2 = normalize_image(ccps_image[:512, (range_y_new-512):])
#     ccps_sub_image3 = normalize_image(ccps_image[(range_x_new-512):, :512])
#     ccps_sub_image4 = normalize_image(ccps_image[(range_x_new-512):, (range_y_new - 512):])
#
#     image1 = Image.fromarray(ccps_sub_image1)
#     image1 = image1.convert('L')
#     image1.save('Training_data/BioSRdata/%s%d.png' % ('ccps', i0*4+0))
#     image2 = Image.fromarray(ccps_sub_image2)
#     image2 = image2.convert('L')
#     image2.save('Training_data/BioSRdata/%s%d.png' % ('ccps', i0 * 4 + 1))
#     image3 = Image.fromarray(ccps_sub_image3)
#     image3 = image3.convert('L')
#     image3.save('Training_data/BioSRdata/%s%d.png' % ('ccps', i0 * 4 + 2))
#     image4 = Image.fromarray(ccps_sub_image4)
#     image4 = image4.convert('L')
#     image4.save('Training_data/BioSRdata/%s%d.png' % ('ccps', i0 * 4 + 3))
#
# n_f_actin = f_actin_data.shape[2]
# for i1 in range(n_f_actin):
#     f_actin_image = np.resize(f_actin_data[:, :, i1], [range_x_new, range_y_new])
#     f_actin_sub_image1 = normalize_image(f_actin_image[:512, :512])
#     f_actin_sub_image2 = normalize_image(f_actin_image[:512, (range_y_new-512):])
#     f_actin_sub_image3 = normalize_image(f_actin_image[(range_x_new-512):, :512])
#     f_actin_sub_image4 = normalize_image(f_actin_image[(range_x_new-512):, (range_y_new - 512):])
#
#     image1 = Image.fromarray(f_actin_sub_image1)
#     image1 = image1.convert('L')
#     image1.save('Training_data/BioSRdata/%s%d.png' % ('f_actin', i1 * 4 + 0))
#     image2 = Image.fromarray(f_actin_sub_image2)
#     image2 = image2.convert('L')
#     image2.save('Training_data/BioSRdata/%s%d.png' % ('f_actin', i1 * 4 + 1))
#     image3 = Image.fromarray(f_actin_sub_image3)
#     image3 = image3.convert('L')
#     image3.save('Training_data/BioSRdata/%s%d.png' % ('f_actin', i1 * 4 + 2))
#     image4 = Image.fromarray(f_actin_sub_image4)
#     image4 = image4.convert('L')
#     image4.save('Training_data/BioSRdata/%s%d.png' % ('f_actin', i1 * 4 + 3))
#
# n_er = er_data.shape[2]
# for i2 in range(n_er):
#     er_image = np.resize(er_data[:, :, i2], [range_x_new, range_y_new])
#     er_sub_image1 = normalize_image(er_image[:512, :512])
#     er_sub_image2 = normalize_image(er_image[:512, (range_y_new-512):])
#     er_sub_image3 = normalize_image(er_image[(range_x_new-512):, :512])
#     er_sub_image4 = normalize_image(er_image[(range_x_new-512):, (range_y_new - 512):])
#
#     image1 = Image.fromarray(er_sub_image1)
#     image1 = image1.convert('L')
#     image1.save('Training_data/BioSRdata/%s%d.png' % ('er', i2 * 4 + 0))
#     image2 = Image.fromarray(er_sub_image2)
#     image2 = image2.convert('L')
#     image2.save('Training_data/BioSRdata/%s%d.png' % ('er', i2 * 4 + 1))
#     image3 = Image.fromarray(er_sub_image3)
#     image3 = image3.convert('L')
#     image3.save('Training_data/BioSRdata/%s%d.png' % ('er', i2 * 4 + 2))
#     image4 = Image.fromarray(er_sub_image4)
#     image4 = image4.convert('L')
#     image4.save('Training_data/BioSRdata/%s%d.png' % ('er', i2 * 4 + 3))

n_microtubules = microtubules_data.shape[2]
for i3 in range(n_microtubules):
    microtubules_image = np.resize(microtubules_data[:, :, i3], [range_x_new, range_y_new])
    microtubules_sub_image1 = normalize_image(microtubules_image[256:256+512, :512])
    microtubules_sub_image2 = normalize_image(microtubules_image[256:256+512, (range_y_new-512):])
    microtubules_sub_image3 = normalize_image(microtubules_image[256:256+512, 256:256+512])

    # microtubules_sub_image3 = normalize_image(microtubules_image[(range_x_new-512):, :512])
    # microtubules_sub_image4 = normalize_image(microtubules_image[(range_x_new-512):, (range_y_new - 512):])

    image1 = Image.fromarray(microtubules_sub_image1)
    image1 = image1.convert('L')
    image1.save('Training_data/BioSRdata/%s%d.png' % ('microtubulesadd', i3 * 3 + 0))
    image2 = Image.fromarray(microtubules_sub_image2)
    image2 = image2.convert('L')
    image2.save('Training_data/BioSRdata/%s%d.png' % ('microtubulesadd', i3 * 3 + 1))
    image3 = Image.fromarray(microtubules_sub_image3)
    image3 = image3.convert('L')
    image3.save('Training_data/BioSRdata/%s%d.png' % ('microtubulesadd', i3 * 3 + 2))
    # image4 = Image.fromarray(microtubules_sub_image4)
    # image4 = image4.convert('L')
    # image4.save('Training_data/BioSRdata/%s%d.png' % ('microtubules', i3 * 4 + 3))
