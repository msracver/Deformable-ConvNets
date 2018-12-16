#--------------------------------------
# Script to crop object from a document
#--------------------------------------

"""
This class crops an object like table(row and column) from  icdar17 datasets  
"""

import cv2
import os
import cPickle
import numpy as np 
from PIL import ImageColor
from imdb import IMDB 
from PIL import Image


IMAGE_EXTENSIONS = ['.jpg', '.png', '.bmp']

src_path = "./Crop"
dirs = os.listdir(src_path)

output_path = "./Crop/CropImages"

def processCoordinates(coordinates):
    coords = coordinates.split(' ') # Separate based on space
    x1 = 10000
    x2 = 0
    y1= 10000
    y2 = 0
    for coord in coords:
        x, y = coord.split(',')
        x = float(x)
        y = float(y)
        x1 = min(x1, x)
        y1 = min(y1, y)
        x2 = max(x2, x)
        y2 = max(y2, y)

    # print (x1, y1, x2, y2)
    return x1, y1, x2, y2

def loadGTAnnotationsFromXML(xml_path):
    if not os.path.exists(xml_path):
        print ("Error: Unable to locate XML file %s" % (im_name))
        exit(-1)   
    tree = ET.parse(xml_path)
    objs = tree.findall('object')
    num_objs = len(objs)

    # Load object bounding boxes into a data frame.
    boundingBoxes = []
    for ix, obj in enumerate(objs):
        bbox = obj.find('bndbox')
        # Make pixel indexes 0-based
        x1 = float(bbox.find('xmin').text) - 1
        y1 = float(bbox.find('ymin').text) - 1
        x2 = float(bbox.find('xmax').text) - 1
        y2 = float(bbox.find('ymax').text) - 1
        #cls = self._class_to_ind[obj.find('name').text.lower().strip()]
        cls = obj.find('name').text.lower().strip()
        #objs =[obj for obj in objs if obj.find('name').text.lower().strip()]
        #cls = objs
        f_name = currimage.crop((x1,y1,x2,y2))            
        if cls in CLASSES:
            boundingBoxes.append([x1, y1, x2, y2, 1.0, cls])
        #elif USE_REJECT_CLASS:
        #    boundingBoxes.append([x1, y1, x2, y2, 1.0, CLASSES[-1]]) # Add reject class
        #else:
        #    print ("Error: Class not found in list")
        #    exit (-1)

    return boundingBoxes

def main():
#Load an Image
currimage_file = Image.open(os.path.join(dirs),'r')
for currimage in currimage_file:
    data = []
    currimage = currimage.strip()
    print ("Processing file: %s" %(currimage))

    found = False
    for ext in IMAGE_EXTENSIONS:
        currimage_with_ext = currimage + ext
        im_path = os.path.join(dirs, 'Images', currimage_with_ext)
        if os.path.exists(im_path):
            found = True
            break
        if not found:
            print ("Error: Unable to locate file %s" % (currimage))
            exit(-1)
# Load GT annotations
        xml_path = os.path.join(dirs, 'Annotations', currimage + '.xml')
        gtBBoxes = loadGTAnnotationsFromXML(xml_path)

        im = cv2.imread(im_path, cv2.IMREAD_COLOR | cv2.IMREAD_IGNORE_ORIENTATION)
        target_size = config.SCALES[0][0]
        max_size = config.SCALES[0][1]
        im, im_scale = resize(im, target_size, max_size, stride=config.network.IMAGE_STRIDE)
        im_tensor = transform(im, config.network.PIXEL_MEANS)
        im_info = np.array([[im_tensor.shape[2], im_tensor.shape[3], im_scale]], dtype=np.float32)
        data.append({'data': im_tensor, 'im_info': im_info})


#sSaving the crop image
data.save(os.path.join(outout_path, currimage[:currimage.rfind('.')] + ".png"))

if __name__ == '__main__':
    main()