#==================================================================================#
# Author       : Davide Mariani                                                    #
# Script Name  : app.py                                                            #
# Description  : food images classifier web app                                    #
#==================================================================================#
# App developed using Flask and deployed using heroku on the website               #
# https://foodimage-classifier.herokuapp.com/                                      #
#==================================================================================#

import pandas as pd
import pickle
from flask import Flask, request, render_template, redirect, url_for, send_from_directory

from pytorch_scripts.process import process_image
from pytorch_scripts.predict import predict_image

from PIL import Image
import io
import os

app = Flask(__name__)

current_dir = os.path.dirname(os.path.abspath(__file__))

IMAGE_UPLOADS = current_dir + "/storage/"

app.config["IMAGE_UPLOADS"] = IMAGE_UPLOADS

def save_dictionary(dict, datafolder, dict_name):
    """
    This function saves a dictionary to a pickle file named "dict_name.pkl"
    in the folder specified in "datafolder"
    """
    filepath = datafolder+dict_name+'.pkl'

    # Create target folder if it doesn't exist
    if not os.path.exists(datafolder):
        os.mkdir(datafolder)

    print("- Saving dictionary to {}".format(filepath))
    with open(filepath, "wb") as pickle_file:
            pickle.dump(dict, pickle_file)
    return filepath


@app.route("/")
def root():
    return render_template('index.html') #app.send_static_file('index.html')


@app.route("/upload/", methods=["POST", "GET"])
def upload_img():

    uploaded_image = False

    if request.method == "POST": #veryfying that the request method is POST

        if request.files: #veryfying that it contains files

            img_req = request.files['img'] #selecting the file with name = "img"

            #process the image for prediction
            processed_img = process_image(img_req)

            #predict
            prediction = predict_image(processed_img)#

            save_dictionary(prediction, IMAGE_UPLOADS, "pred_tmp")

            predicted_name = prediction['class_name'].replace("_", " ").capitalize()

            #storing the image
            filename = img_req.filename
            save_img_path = os.path.join(app.config["IMAGE_UPLOADS"], filename)
            img = Image.open(img_req)
            img.save(save_img_path)
            print("Image saved...")

            #send the image to 'uploaded_file' to be rendered on pred.html
            return redirect(url_for('uploaded_file', filename=filename, predicted_name = predicted_name))

        print(img)

    return render_template("index.html")

@app.route('/show/<filename>/<predicted_name>')
def uploaded_file(filename, predicted_name):
    prediction = pd.read_pickle(IMAGE_UPLOADS + "pred_tmp.pkl")

    class_prob = round(prediction['prob'][int(prediction['class'])]*100, 2) #print out how much the model is "sure" about the prediction

    other_three_idx = [prediction['prob'].index(x) for x in sorted(prediction['prob'], reverse=True)[1:4] if x>0.0] #find the 3 other highest values in probability from prediction values
    other_three_names = [prediction['class_names_mapping'][idx].replace("_", " ").capitalize() for idx in other_three_idx] #find the respective names
    other_three_perc = [round(prediction['prob'][idx]*100, 2) for idx in other_three_idx] #find the respective percentage of probability

    #text to send for displaying it on the app
    other_three_text = ["{} - {}%".format(other_three_names[i], other_three_perc[i]) for i in range(len(other_three_idx))]

    return render_template('pred.html', filename=filename, predicted_name=predicted_name, class_prob=class_prob, other_three_text = other_three_text)

@app.route('/uploads/<filename>')
def send_file(filename):
    return send_from_directory(IMAGE_UPLOADS, filename)


if __name__ == '__main__':
    app.run()
