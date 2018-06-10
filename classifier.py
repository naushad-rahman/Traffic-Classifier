from  tl_classifier import TLClassifier
import numpy as np
import cv2
from cv_bridge import CvBridge
import glob

light_classifier = TLClassifier(0.3 ,1.5 ,False)
pathlist = glob.glob("/home/student/Desktop/TrafficClassification/*.jpg")
print(pathlist)

#Path ='/home/student/Desktop/TrafficClassification/TEST.jpg'

for Path in pathlist:
	cv_image = cv2.imread(Path)
	
	processed_img = cv_image[0:600, 0:800] # was [20:400, 0:800]  
	#Convert image to RGB format
	processed_img = cv2.cvtColor(processed_img, cv2.COLOR_BGR2RGB)


	img_full_np = light_classifier.load_image_into_numpy_array(processed_img)

	print("Get in Localization-Classification")
	b, conf, cls_idx = light_classifier.get_localization_classification(img_full_np, visual=False)
	if np.array_equal(b, np.zeros(4)):
		print ('unknown')
		unknown = True
	else:
		if cls_idx == 1.0:
			print('Green', b)
			#light_state = TrafficLight.GREEN
		elif cls_idx == 2.0:
			print('Red', b)
			#light_state = TrafficLight.RED
		elif cls_idx == 3.0:
			print('Yellow', b)
			#light_state = TrafficLight.YELLOW
		elif cls_idx == 4.0:
			print('Unknown', b)
			#light_state = TrafficLight.UNKNOWN
		else:
			print('Really Unknown! Didn\'t process image well', b)
			#light_state = TrafficLight.UNKNOWN


	cv2.imshow('image',cv_image)
	cv2.waitKey(0)
cv2.destroyAllWindows()