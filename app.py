import joblib
import random
import streamlit as st
import numpy as np
from skimage.io import imread
from skimage.transform import resize
from PIL import Image
from sklearn.datasets import load_digits
digits = load_digits()
images = digits['images']
target_names = digits['target_names']
st.title('Digit Classfier using Machine Learning')
st.text('Upload the image of a digit to classify')
model = joblib.load('Digit Classifier')
img = random.choice(images)
flat_data = []
img = np.array(img)
img_resized = resize(img,(8,8,1)) 
temp = resize(img_resized,(400,400,3))
flat_data.append(img_resized.flatten())
flat_data = np.array(flat_data)

if st.button('CLICK TO UPLOAD RANDOM IMAGE FROM THE DATASET'):
  st.write("Image uploaded!")

if st.button('PREDICT'):
  st.image(img_resized,caption = 'Uploaded image',clamp = True)
  st.write('Result....')
  y_out = model.predict(flat_data)
  y_out = target_names[y_out[0]]
  st.title(f'Predicted output:{y_out}')
  q = model.predict_proba(flat_data)
  for index,item in enumerate(target_names):                  #prints the probability of all the categories
    st.write(f'{item} : {q[0][index]*100}%')