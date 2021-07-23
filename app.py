import joblib
import streamlit as st
import numpy as np
from skimage.io import imread
from skimage.transform import resize
from PIL import Image
st.title('Digit Classfier using Machine Learning')
st.text('Upload the image of a digit to classify')
model = joblib.load('Digit Classifier')
uploaded_file = st.file_uploader("Choose an Image....",type = 'png')

if uploaded_file is not None:
  img = Image.open(uploaded_file)
  st.image(img,caption = "Image Uploaded")

if st.button('PREDICT'):
  st.write('Result....')
  flat_data = []
  CATEGORIES = ['0','1','2','3','4','5','6','7','8','9']
  flat_data = []
  img = np.array(img)
  img_resized = resize(img,(150,150,3)) 
  flat_data.append(img_resized.flatten())
  flat_data = np.array(flat_data)
  y_out = model.predict(flat_data)
  y_out = CATEGORIES[y_out[0]]
  st.title(f'Predicted output:{y_out}')
  q = model.predict_proba(flat_data)
  for index,item in enumerate(CATEGORIES):                  #prints the probability of all the categories
    st.write(f'{item} : {q[0][index]*100}%')