from foci_detection import *
from configparser import ConfigParser
import os
import cv2
import pandas as pd
import logging

# =================== Load the configuration file ===================

parser = ConfigParser()
ini_path = os.path.dirname(os.path.abspath(__file__))+'/' # local path
parser.read(ini_path+'config.ini')

# configuration infomation
name = parser.get('Paths','name')
input_path = parser.get('Paths','input_image_folder')
output_path = parser.get('Paths','output_image_folder')
margin = int(parser.get('Parameters','border_margin'))
min_pixel = int(parser.get('Parameters','min_foci_size'))

# =================== log information ===================

logging.basicConfig(filename=output_path+'/logfile.log', level=logging.INFO, format='%(asctime)s, %(levelname)s, %(filename)s, %(message)s')



# =================== setting ===========================

imagesize = 500 # 500 pixels
number_foci = list()
radius = list()

# =================== find radius  ======================

try:
  print('find radius process...')
  for image in os.listdir(input_path):
    # print('find radius image > ', image)
    img_path = os.path.join(input_path,image)
    img = cv2.imread(img_path, flags = cv2.IMREAD_GRAYSCALE)

    inner_border = border_detection(img, imagesize, 0) # set margin = 0
    if str(inner_border) == 'not circle detected':
      continue
    else:
      a,b,r = inner_border
      radius.append(r)

  med_r = radius_selector(radius)

  # find radius log status
  logging.info('{}, find radius: success'.format(name))
except:
  logging.error('{}, find radius: failed'.format(name))


# =================== foci detection  ===================
try:
  print('foci detection process...')
  for image in os.listdir(input_path):
    try:
      # print('image > ', image)
      img_path = os.path.join(input_path,image)
      img = cv2.imread(img_path, flags = cv2.IMREAD_GRAYSCALE)
      inner_border = border_detection(img, imagesize, margin) # user can set this margin
      img_croped = crop_inner_circle(img, imagesize, inner_border,med_r)
      
      # detection
      binary_img = binary_Threshold(img_croped,inner_border)
      box = foci_detection(binary_img, inner_border, min_pixel)

      # count number of Foci
      id = name+'_'+image[:-4]
      number_foci.append([id,foci_count(box)])
      foci_info = num_pixels(box)
      foci_info.to_csv(output_path+'/'+image[:-4]+'.csv')

      # foci information log status
      logging.info('{}, {}, foci detected: success'.format(name,image))

    except:
      logging.error('{}, {}, foci detected: failed'.format(name,image))
    

  foci_count_result = pd.DataFrame(number_foci, columns=["id", "foci count"])
  foci_count_result.to_csv(output_path+'/foci_count_result.csv')

  # foci detection process log status
  logging.info('{}, detection process: complete'.format(name))

except:
  logging.error('{}, detection process: failed'.format(name))

