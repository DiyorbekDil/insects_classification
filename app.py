import streamlit as st
from fastai.vision.all import *
import plotly.express as px
import pathlib
temp = pathlib.PosixPath
pathlib.PosixPath = pathlib.WindowsPath
#title
st.title('Hashoratlarni farqlovchi model')

#Rasmni joylash
file = st.file_uploader('Rasm yuklash (Hozircha faqat 3 ta hashorat rasmini taniydi bular ninachi, kapalak va ari. Iltimos hozircha boshqa rasm yuklamang)', type=['png', 'jpeg', 'jpg', 'gif', 'svg'])
if file:
   st.image(file)
   #PIL convert
   img = PILImage.create(file)
   #Modelni olish
   model = load_learner('insects_model.pkl')
   # prediction
   pred, pred_id, probs = model.predict(img)
   st.success(f'Bashorat: {pred}')
   st.info(f'Aniqlik: {probs[pred_id]*100:.3f}%')
   #plotting
   fig = px.bar(x=probs*100, y=model.dls.vocab)
   st.plotly_chart(fig)
