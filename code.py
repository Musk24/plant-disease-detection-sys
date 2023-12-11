
# Plotting 12 images to check dataset
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('TkAgg',force=True)
from matplotlib import pyplot as plt
from matplotlib.image import imread
print("Switched to:",matplotlib.get_backend())
import cv2
import random
import os
from os import listdir
from PIL import Image
import tensorflow as tf
tf.keras.backend.set_floatx('float64')
from keras.preprocessing import image
from tensorflow import keras
from keras.utils import img_to_array, array_to_img
from keras.optimizers import Adam
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D
from keras.layers import Activation, Flatten, Dropout, Dense
from sklearn. model_selection import train_test_split
from keras.models import model_from_json
from keras.utils import to_categorical
tf.keras.callbacks.History()

dataset_path = r"C:\Users\KIIT\OneDrive\Documents\plant disease\dataset\plantvillage3"
plt.figure(figsize = (12, 12))
dataset_path = r"C:\Users\KIIT\OneDrive\Documents\plant disease\dataset\plantvillage3\Tomato_Early_blight"

for i in range(1, 17):
    plt.subplot(4, 4, i)
    plt.tight_layout()
    rand_img = imread(dataset_path +'/'+ random.choice(sorted(os.listdir(dataset_path))))
    plt.imshow(rand_img)
    plt.xlabel(rand_img.shape[1], fontsize = 10) # width of image
    plt.ylabel(rand_img.shape[0], fontsize = 10) # height of image

dataset_path = r"C:\Users\KIIT\OneDrive\Documents\plant disease\dataset\plantvillage3\Tomato_Bacterial_spot"

for i in range(1, 17):
    plt.subplot(4, 4, i)
    plt.tight_layout()
    rand_img = imread(dataset_path +'/'+ random.choice(sorted(os.listdir(dataset_path))))
    plt.imshow(rand_img)
    plt.xlabel(rand_img.shape[1], fontsize = 10) # width of image
    plt.ylabel(rand_img.shape[0], fontsize = 10) # height of image
    
dataset_path = r"C:\Users\KIIT\OneDrive\Documents\plant disease\dataset\plantvillage3\Tomato_Leaf_Mold"

for i in range(1, 17):
    plt.subplot(4, 4, i)
    plt.tight_layout()
    rand_img = imread(dataset_path +'/'+ random.choice(sorted(os.listdir(dataset_path))))
    plt.imshow(rand_img)
    plt.xlabel(rand_img.shape[1], fontsize = 10) # width of image
    plt.ylabel(rand_img.shape[0], fontsize = 10) # height of image



    # Converting Images to array 

def convert_image_to_array(image_dir): 
    try:
        image = cv2.imread(image_dir)
        if image is not None :
            image = cv2.resize(image, (256, 256))  
            return img_to_array(image)
        else :
            return np.array([])
    except Exception as e:
        print(f"Error : {e}")
        return None
dataset_path = (r"C:\Users\KIIT\OneDrive\Documents\plant disease\dataset\plantvillage3")
labels = os.listdir(dataset_path)

print(labels)

dataset_path = (r"C:\Users\KIIT\OneDrive\Documents\plant disease\dataset\plantvillage3")
root_dir = listdir(dataset_path)
image_list, label_list = [], []
all_labels = ['Tomato_Early_blight', 'Tomato_Bacterial_spot', 'Tomato_Leaf_Mold']
binary_labels = [0, 1, 2]
temp = -1

# Reading and converting image to numpy array
for directory in root_dir:
    plant_image_list = listdir(f"{dataset_path}/{directory}")
    temp += 1
    # Check if temp is within the valid range of indices
    if temp < len(binary_labels):
        for files in plant_image_list:
            image_path = f"{dataset_path}/{directory}/{files}"
            image_list.append(convert_image_to_array(image_path))
            label_list.append(binary_labels[temp])
    else:
        # Handle the case where temp is out of range
        print(f"Warning: temp ({temp}) is out of range for binary_labels")
        
#for directory in root_dir:
  #plant_image_list = listdir(f"{dataset_path}/{directory}")
  #temp += 1
  #for files in plant_image_list:
    #image_path = f"{dataset_path}/{directory}/{files}"
    #image_list.append(convert_image_to_array(image_path))
    #label_list.append(binary_labels[temp])*\

    # Visualize the number of classes count

label_counts = pd.DataFrame(label_list).value_counts()
print(label_counts)
label_counts.head()


# Next we will observe the shape of the image.

image_list[0].shape
# Checking the total number of the images which is the length of the labels list.

label_list = np.array(label_list)
label_list.shape

x_train, x_test, y_train, y_test = train_test_split(image_list, label_list, test_size=0.2, random_state = 10) 
# Now we will normalize the dataset of our images. As pixel values ranges from 0 to 255 so we will divide each image pixel with 255 to normalize the dataset.

x_train = np.array(x_train, dtype=np.csingle) / 225.0
x_test = np.array(x_test, dtype=np.csingle) / 225.0
x_train = x_train.reshape(-1, 256, 256, 3)
x_test = x_test.reshape(-1, 256, 256, 3)
#x_train = np.expand_dims(x_train, axis=-1)
#x_test = np.expand_dims(x_test, axis=-1)
y_train = to_categorical(y_train)
y_test = to_categorical(y_test)

model = Sequential()
#model.add(Flatten())
model.add(Conv2D(32, (3, 3), padding = "same",input_shape = (256, 256, 3), activation = "relu"))
model.add(MaxPooling2D(pool_size = (3, 3)))
model.add(Conv2D(16, (3, 3), padding = "same", activation = "relu"))
model.add(MaxPooling2D(pool_size = (2, 2)))
model.add(Flatten())
model.add(Dense(8, activation = "relu"))
model.add(Dense(3, activation = "softmax"))
model.build(input_shape = (256, 256, 3))
model.summary()

model.compile(loss = 'categorical_crossentropy', optimizer = Adam(0.0001), metrics = ['accuracy'])
# Splitting the training data set into training and validation data sets

x_train, x_val, y_train, y_val = train_test_split(x_train, y_train, test_size = 0.2, random_state = 10)
# Training the model
class_mode="categorical"
epochs = 50
batch_size = 128
history = model.fit(x_train, y_train, batch_size = batch_size, epochs = epochs, validation_data = (x_val, y_val))

model.save(r"C:\Users\KIIT\OneDrive\Documents\plant disease\plant_disease_model (1).h5")

# Plot the training history

plt.figure(figsize = (12, 5))
plt.plot(history.history['accuracy'], color = 'r')
plt.plot(history.history['val_accuracy'], color = 'b')
plt.title('Model Accuracy')
plt.ylabel('Accuracy')
plt.xlabel('Epochs')
plt.legend(['train', 'val'])
#plt.scatter()
plt.show()

print("Calculating model accuracy")

scores = model.evaluate(x_test, y_test)
print(f"Test Accuracy: {scores[1] * 100}")

y_pred = model.predict(x_test)

# Plotting image to compare
#Visualizing the original and predicted labels for the test images
img = array_to_img(x_test[11])
img
# Finding max value from predition list and comaparing original value vs predicted

print("Originally : ", all_labels[np.argmax(y_test[11])])
print("Predicted : ", all_labels[np.argmax(y_pred[4])])
print(y_pred[2])

for i in range(50):
    print (all_labels[np.argmax(y_test[i])], " ", all_labels[np.argmax(y_pred [1])])