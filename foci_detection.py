import numpy as np
import pandas as pd
import cv2
import statistics
import os

# =================== circle detection session ===================
def border_detection(img, size, margin):
  width, height = size, size
  imgresize = cv2.resize(img,(width, height))
  gray_img = cv2.cvtColor(imgresize,cv2.COLOR_GRAY2BGR)
  img_blur = cv2.GaussianBlur(imgresize,(5,5),0)

  # Find center point
  # cv2.HoughCircles (image, method, dp, minDist, param1, param2, minRadius, maxRadius)
  circles = cv2.HoughCircles(img_blur,cv2.HOUGH_GRADIENT,1,size,param1=60,param2=60,minRadius=0,maxRadius=0)
  
  if circles is None:
    return print('not circle detected')
  else:
    circles = np.uint16(np.around(circles))

    #Draw circle border
    for i in circles[0,:]:
      # draw the outer circle
      # circle(image, centroid(x,y), radius, color, thickness)
      cv2.circle(gray_img,(i[0],i[1]),i[2],(0,255,0),2)
      # draw the inner circle
      cv2.circle(gray_img,(i[0],i[1]),i[2]-margin,(0,0,255),2)
      # cv2_imshow(gray_img)
      x = i[0]
      y = i[1]
      r = i[2]-margin
    return np.array([x,y,r])

def radius_selector(r):
  r_med = statistics.median(r)
  return int(r_med)


def crop_inner_circle(img,size,circle,radius):
  high, width = size, size
  imgresize = cv2.resize(img,(width, high))
  x,y,_ = circle
  r = radius
  # black rectangle: size = image size
  mask = np.zeros(imgresize.shape, dtype=np.uint8)

  # white circle on mask: center x, y radius: r
  cv2.circle(mask,(x,y),r,(255,255,255),-1,8,0)
  result_array = imgresize & mask 
  return result_array

# ======================= delete line =======================
def line_detection(img,size,circle, margin):
  high, width = size, size
  imgresize = cv2.resize(img,(width, high))
  x,y,r = circle
  mask = np.zeros(imgresize.shape, dtype=np.uint8)
  cv2.circle(mask,(x,y),r,(255,255,255),-1,8,0)
  cv2.circle(mask,(x,y),r-margin,(0,0,0),-1,8,0)
  result_array = imgresize & mask

  # line detection
  minLineLength = 0
  maxLineGap = 0
  lines = cv2.HoughLinesP(result_array,1,np.pi/180,40,minLineLength,maxLineGap)
  lines

  return lines

def delete_line(img,size,lines):
  high, width = size, size
  imgresize = cv2.resize(img,(width, high))
  if lines is not None:
    for i in range(0, len(lines)):
      l = lines[i][0]
      cv2.line(imgresize, (l[0], l[1]), (l[2], l[3]), (0,0,0), 5, cv2.LINE_AA)
  return imgresize

# ===================== foci detection session =====================

# Threshold
def binary_Threshold(img, circle):
  x,y,r = circle
  width, height = img.shape
  boxsize = int(0.154*width)
  binary_img = cv2.adaptiveThreshold(img, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,cv2.THRESH_BINARY_INV, boxsize, 17)
  
  _, labels = cv2.connectedComponents(binary_img)
  for i in range(labels.shape[0]):
    labels[i] = np.where(labels[i] <= 1, 0, labels[i])
    labels[i] = np.where(labels[i] > 1, 255, labels[i])
  labels = labels.astype(np.uint8)
  return labels



# Finding Foci position
def foci_detection(binary_img, circle, min_pixels):
  x,y,r = circle
  width, height = binary_img.shape
  
  _, _, boxes, centroid = cv2.connectedComponentsWithStats(binary_img)

  # first box is the background
  boxes = boxes[1:]
  centroid = centroid[1:]
  filtered_boxes = []

  # set min, max that can detected
  max_height = r*0.5
  max_width = r*0.5

  # circle area
  pi = 3.14
  cir_area = pi*r*r
  i=1 # foci id
  for (x,y,w,h,pixels),(centroid_x,centroid_y) in zip(boxes,centroid):
      if pixels > min_pixels and h < max_height and w < max_width:
          normal_foci_area = round((pixels/cir_area)*100, 4) #4 decimal
          filtered_boxes.append((x,y,w,h,i,pixels,int(centroid_x),int(centroid_y),normal_foci_area))
          i=i+1

  return filtered_boxes

# Count number of Foci 
def foci_count(boxes):
  return len(boxes)

# Count number of Foci 
def num_pixels(boxes):
  filtered_pixels = []
  for x,y,w,h,i,pixels,centroid_x,centroid_y,normal_foci_area in boxes:
    filtered_pixels.append([i,centroid_x,centroid_y,pixels,normal_foci_area])
  return pd.DataFrame(filtered_pixels, columns =['id','centroid_x','centroid_y','area_pixels','Normalize Area %'])

# =============== draw in formation on image session ===============

# Draw rectangle on original image
def foci_draw_detected(img,size,boxes):
  high, width = size, size
  imgresize = cv2.resize(img,(width, high))
  gray_img = cv2.cvtColor(imgresize,cv2.COLOR_GRAY2BGR)
  
  for x,y,w,h,i,pixels,centroid_x,centroid_y,normal_foci_area in boxes:
      cv2.rectangle(gray_img, (x,y), (x+w,y+h), (0,255,255),1)
  return gray_img

# write text on image
def write_numberOfFoci(img,size,boxes):
  high, width = size, size
  imgresize = cv2.resize(img,(width, high))
  gray_img = cv2.cvtColor(imgresize,cv2.COLOR_GRAY2BGR)

  # write text
  for x,y,w,h,i,pixels,centroid_x,centroid_y,normal_foci_area in boxes:
      fontScale = 0.0006*size
      font = cv2.FONT_HERSHEY_SIMPLEX
      n_foci = cv2.putText(gray_img, str(i), (centroid_x,centroid_y), font, fontScale, (0,255,255), 1, cv2.LINE_AA)
  return gray_img

#  ======================= program =======================

def get_median_radius(path,imagesize):
  margin = 0
  radius = list()
  for i in os.listdir(path):
    img_path = os.path.join(path,i)
    img = cv2.imread(img_path, flags = cv2.IMREAD_GRAYSCALE)

    inner_border = border_detection(img, imagesize, margin)
    a,b,r = inner_border
    radius.append(r)
  return radius_selector(radius)
  
def main(img,imagesize,margin,med_r,min_pixels):
  # border detection
  inner_border = border_detection(img, imagesize, 0)
  # image cropping
  img_croped = crop_inner_circle(img, imagesize, inner_border,med_r-margin)
  # foci detection
  binary_img = binary_Threshold(img_croped,inner_border)
  box = foci_detection(binary_img, inner_border, min_pixels)
  # foci count
  return foci_count(box)