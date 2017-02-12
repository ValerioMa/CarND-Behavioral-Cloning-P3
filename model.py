import csv
import numpy as np
from scipy.misc import imread
import matplotlib.pyplot as plt
from sklearn.utils import shuffle
import cv2 as cv
from keras.models import Sequential
from keras.models import load_model
from keras.layers import Dense, Dropout, Flatten, Lambda, ELU
from keras.layers.convolutional import Convolution2D
from keras.models import load_model
from keras import backend as K
import string
import os



# Windows \ Linux compatibility
if os.name=='nt':
   # Windows
   separator = '\\'
elif os.name=='posix':
   # Linux
   separator = '/'
else:
   raise ValueError('os.name not implemented')


# CSV data to load
# csv_fields "throttle","steering","right","center","left","brake","speed"
csv_files = [{'file_name':'Giro_sinistra.csv','rgt_offs' : 0.1,'cen_offs' : 0.3,'lft_offs' : 0.35},
				{'file_name':'Giro_centro.csv','rgt_offs' : -0.10,'cen_offs' : 0,'lft_offs' : +0.10},
				{'file_name':'Giro_destra.csv','rgt_offs' : -0.35,'cen_offs' : -0.3,'lft_offs' : -0.1},
				{'file_name':'45_destra_muso_centro.csv','rgt_offs' : 0.05,'cen_offs' : 0.10,'lft_offs' : 0.10},
				{'file_name':'45_sinistra_muso_centro.csv','rgt_offs' : -0.15,'cen_offs' : -0.10,'lft_offs' : -0.05}				
				]

#if raffina_modello==None       --> create new model
#if raffina_modello=='model.h5' --> load model.h5 weights
raffina_modello = 'model.h5'

# Network params
validation_ratio = 0.2
EPOCH_NUMBER = 1
BATCH_SIZE = 256

# Trasformation of img (Crop and conversion to YUV space)
def img_trasform(img):
	img = cv.resize(img, (320,160))
	img = img[30:(160-25),0:320]   # taglia la macchina in basso e l'orizzonte
	img = cv.resize(img, (200,66))   #resize secondo quando scritto in paper NVIDIA
	img = cv.cvtColor(img,cv.COLOR_RGB2BGR)
	img = cv.cvtColor(img,cv.COLOR_BGR2YUV)
	return img[:,:,:]
	
# Read csv data and apply offset to steering angle of the different (left, center, right)
def read_names(file_name,left_steering_offset,center_steering_offset,right_steering_offset):		
	center = []
	left = []
	right = []
	file_path = '.{0}data{0}CSV{0}{1}'.format(separator,file_name)	
	print(file_path)
	with open(file_path) as csvfile:		
		reader = csv.DictReader(csvfile)
		for row in reader:
			steer = float(row["steering"])
			center.append((row["center"].strip().replace('\\',separator), steer + center_steering_offset))
			right.append((row["right"].strip().replace('\\',separator), steer + right_steering_offset))
			left.append((row["left"].strip().replace('\\',separator), steer + left_steering_offset))
	return left, center, right
		
	
#Generator to load batch_size image
def gen(data, batch_size=128):
	imgs = []
	steers = []
	while(True):
		for i,(img_path, steer) in enumerate(data):	
			img = imread(img_path).astype(np.float32)
			img = img_trasform(img)	
			imgs.append(img)
			steers.append(steer)
			# Center image		
			if len(imgs)%batch_size == 0:			
			#if (i+1)%batch_size == 0:			
				yield np.array(imgs), np.array(steers)
				imgs  = []
				steers = []
		#if len(steers)>0:
		#	yield np.array(imgs), np.array(steers)
		

# LOAD DATA from csv
name_steer_list = []
print('Importing data from:')
for csv_file in csv_files:	
	print(' - {}'.format(csv_file['file_name']))
	left, center, right = read_names(csv_file['file_name'],csv_file['lft_offs'],csv_file['cen_offs'],csv_file['rgt_offs'])
	name_steer_list = name_steer_list+ left + center + right

# Shuffle the data
name_steer_list = shuffle(name_steer_list)
# Total nuber of data loaded
examples_number = len(name_steer_list)

# Validation data number
validation_size = (int)(examples_number * validation_ratio)

# Training data number
train_size = examples_number - validation_size

# Split data in train and validation
train_names, validation_names = name_steer_list[0:train_size], name_steer_list[train_size:examples_number]


if raffina_modello == None:
	# if there is no model to load 
	
	def get_model():  # Similar to NVIDIA network
	
		ch, row, col = 3, 66, 200  # camera format

		model = Sequential()
		#Normalize image
		model.add(Lambda(lambda x: x/127.5 - 1.,
			input_shape=(row, col, ch),
			output_shape=(row, col, ch)))
		
		#Convolution
		model.add(Convolution2D(24, 5, 5, init='he_normal', activation='relu',subsample=(2, 2), border_mode="same"))
		model.add(Convolution2D(36, 5, 5, init='he_normal', activation='relu',subsample=(2, 2), border_mode="same"))
		model.add(Convolution2D(48, 5, 5, init='he_normal', activation='relu',subsample=(2, 2), border_mode="same"))
		model.add(Convolution2D(64, 3, 3, init='he_normal', activation='relu',subsample=(1, 1), border_mode="same"))
		model.add(Convolution2D(64, 3, 3, init='he_normal', activation='relu',subsample=(1, 1), border_mode="same"))
		model.add(Flatten())	
		
		#Classificaiton
		model.add(Dense(1164,activation='relu'))	
		model.add(Dropout(.5))
		model.add(Dense(100,activation='relu'))	
		model.add(Dropout(.5))
		model.add(Dense(50,activation='relu'))	
		model.add(Dropout(.2))
		model.add(Dense(10,activation='sigmoid'))		
		model.add(Dense(1))

		model.compile(optimizer="adam", loss="mse")

		return model
	model = get_model()
else:
	# if there is a model to load, load it
	model = load_model(raffina_modello)


# Iniziamo il training
print('Session numbers')
print('- Training data: {}'.format(train_size))
print('- Validation data: {}'.format(validation_size))

# FInd the right nuber to avoid warning message
train_size = np.floor(train_size/BATCH_SIZE)*BATCH_SIZE

model.fit_generator(gen(train_names, BATCH_SIZE),
						samples_per_epoch= train_size, 
						nb_epoch=EPOCH_NUMBER, 
						validation_data = gen(validation_names, 1),   
						nb_val_samples = validation_size,   
						verbose=1)   #validation_size

model.save('model.h5')

del model

# prova per eliminare errore 
K.clear_session()

# Call me I am done
import winsound
Freq = 2500 #2500 # Set Frequency To 2500 Hertz
Dur = 2500 # Set Duration To 1000 ms == 1 second
winsound.Beep(Freq,Dur)
