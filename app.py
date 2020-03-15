from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.mobilenet import preprocess_input
import numpy as np

model=load_model('cat_dog.h5')

print('model loading successfull ')
print('Starting App')

from flask import Flask, request, jsonify, render_template

app = Flask(__name__)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict',methods=['GET','POST'])
def predict():
    '''
    For rendering results on HTML GUI
    '''
    
    f = request.files['file']
    
    img=image.load_img(f.filename,target_size=(299,299))
    x = image.img_to_array(img)
    x = preprocess_input(x)
    x = np.expand_dims(x, axis=0)
    pred = model.predict(x)[0]

    if pred[0]>0.5:
    	prediction='Yes its a cat!'
    else :
    	prediction='Nah! looks like a dog :/'

    return render_template('index.html', prediction_text='Answer : {}'.format(prediction))

if __name__=="__main__":
    app.run(host='localhost',port=5010)
