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
            prediction = predict_image(processed_img)

            predicted_name = prediction['class_name'].replace("_", " ").capitalize() + " ({}% sure...)".format(round(prediction['prob'][int(prediction['class'])]*100, 2))

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
    return render_template('pred.html', filename=filename, predicted_name=predicted_name)

@app.route('/uploads/<filename>')
def send_file(filename):
    return send_from_directory(IMAGE_UPLOADS, filename)


if __name__ == '__main__':
    app.run()
