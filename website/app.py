from flask import Flask, request, render_template, redirect, url_for, send_from_directory
from PIL import Image
import io
import os

app = Flask(__name__)

IMAGE_UPLOADS = "storage/"

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
            #print(img_req)

            img = Image.open(img_req)

            filename = img_req.filename

            save_img_path = os.path.join(app.config["IMAGE_UPLOADS"], filename)

            img.save(save_img_path) #storing the image

            print("Image saved...")

            return redirect(url_for('uploaded_file', filename=filename))

    return render_template("index.html")

@app.route('/show/<filename>')
def uploaded_file(filename):
    return render_template('pred.html', filename=filename)

@app.route('/uploads/<filename>')
def send_file(filename):
    return send_from_directory(IMAGE_UPLOADS, filename)


if __name__ == '__main__':
    app.run()
