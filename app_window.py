## window version
# import module
from foci_detection import *
from configparser import ConfigParser
import logging

# =================== Load the configuration file ===================

parser = ConfigParser()
ini_path = os.path.dirname(os.path.abspath(__file__)) # local path
parser.read(ini_path+'\config.ini')

# configuration infomation
name = parser.get('Paths','plate_name')
input_path = parser.get('Paths','input_image_folder')
output_path = parser.get('Paths','output_image_folder')
margin = parser.getint('Parameters','border_margin')
min_pixel = parser.getint('Parameters','min_foci_size')

# =================== make directory ====================

# make new directory follow plate name 
main_path = os.path.join(output_path,name)
info_file = os.path.join(main_path,'foci_description')
binary_box_path = os.path.join(main_path,'binary_image')
raw_box_path = os.path.join(main_path,'raw_box')
raw_number = os.path.join(main_path,'raw_number')

try:
  os.mkdir(main_path)
  os.mkdir(info_file)
  os.mkdir(binary_box_path)
  os.mkdir(raw_box_path)
  os.mkdir(raw_number)
except:
  print("error: can't make directory")

# log information
logging.basicConfig(filename=main_path+'\logfile.log', level=logging.INFO, format='%(asctime)s, %(levelname)s, %(filename)s, %(message)s')

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
      img_crop = crop_inner_circle(img, imagesize, inner_border,med_r)
      
      # detection
      binary_img = binary_Threshold(img_crop,inner_border)
      box = foci_detection(binary_img, inner_border, min_pixel)

      # count number of Foci
      id = name+'_'+image[:-4]
      number_foci.append([id,foci_count(box)])
      foci_info = num_pixels(box)
      img_csv = image[:-4]+".csv"
      foci_info.to_csv(os.path.join(info_file,img_csv))

      #image result
      result_binary_box = foci_draw_detected(binary_img,imagesize,box)
      result_raw_box = foci_draw_detected(img_crop,imagesize,box)
      result_raw_number = write_numberOfFoci(img_crop,imagesize,box)

      #save file
      os.chdir(binary_box_path)
      cv2.imwrite(image, result_binary_box)
      os.chdir(raw_box_path)
      cv2.imwrite(image, result_raw_box)
      os.chdir(raw_number)
      cv2.imwrite(image, result_raw_number)

      # foci information log status
      logging.info('{}, {}, foci detected: success'.format(name,image))

    except:
      logging.error('{}, {}, foci detected: failed'.format(name,image))
    

  foci_count_result = pd.DataFrame(number_foci, columns=["id", "foci count"])
  foci_count_result.to_csv(os.path.join(main_path,'foci_count_result.csv'))

  # foci detection process log status
  logging.info('{}, detection process: complete'.format(name))

except:
  logging.error('{}, detection process: failed'.format(name))

