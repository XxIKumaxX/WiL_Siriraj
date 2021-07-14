import configparser as cf
from main_function import *

# =================== Load the configuration file ===================
setup = cf.ConfigParser()
path_ini = r"D:\internship\code\Python_configparser\parameter.ini"
setup.read(path_ini)

## paths ##
path_input = setup.get('paths', 'path_inputs')
path_output = setup.get('paths', 'path_outputs')
print(path_input, path_output, sep='\n')
## parameters ## 
plate_name = setup.get('parameters', 'plate_name')
imagesize = setup.getint('parameters', 'imagesize')
margin = setup.getint('parameters', 'margin')
min_pixels = setup.getint('parameters', 'min_pixels')

# ===================  application ===================

## make new directory follow plate name 
main_path = os.path.join(path_output,plate_name)
binary_box_path = os.path.join(main_path,"binary_image")
raw_box_path = os.path.join(main_path,"raw_box")
raw_number = os.path.join(main_path,"raw_number")
try: 
    os.mkdir(main_path)
    os.mkdir(binary_box_path)
    os.mkdir(raw_box_path)
    os.mkdir(raw_number)
except: 
    print("error: can't make directory")

## save files
try:
    med_r = get_median_radius(path_input, imagesize)
    for i in os.listdir(path_input): #i=filename.jpg
        #read image
        img_path = os.path.join(path_input,i)
        img = cv2.imread(img_path, flags = cv2.IMREAD_GRAYSCALE)
        #border detection, crop image
        inner_border = border_detection(img, imagesize, margin)
        img_crop = crop_inner_circle(img,imagesize,inner_border, med_r)
        #foci detection
        binary_img = binary_Threshold(img_crop,inner_border)
        box = foci_detection(binary_img, inner_border, min_pixels)
        #result
        result_binary_box = foci_draw_detected(binary_img,imagesize,box)
        result_raw_box = foci_draw_detected(img_crop,imagesize,box)
        result_raw_number = write_numberOfFoci(img_crop,imagesize,box)
        #save file
        os.chdir(binary_box_path)
        cv2.imwrite(i, result_binary_box)
        os.chdir(raw_box_path)
        cv2.imwrite(i, result_raw_box)
        os.chdir(raw_number)
        cv2.imwrite(i, result_raw_number)
    print("Process complete")
except: 
    print('error')