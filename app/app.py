from flask import Flask, render_template, request
import os
from deeplearning import predict

# __name__ refers to the name of the file (app.py)
app = Flask(__name__)

BASE_PATH = os.getcwd()
UPLOAD_PATH = os.path.join(BASE_PATH, 'static/upload/')

# Map the url to the function index()
# GET method makes sure nothing happens to webpage after submitting a form
# POST method allows user to submit a form 
@app.route('/', methods=['POST', 'GET'])
def index():
    if request.method == 'POST':
        # Retrieves uploaded form and stores it in variable
        upload_file = request.files['image_name']
        filename = upload_file.filename
        # Constructs path using uploaded form name and the path to the new folder that will save the image static/upload
        path_save = os.path.join(UPLOAD_PATH, filename)
        upload_file.save(path_save)
        plate = predict(path_save, filename)

        return render_template('index.html', upload=True, upload_image=filename, text=plate)

    # Renders the html on the webpage
    return render_template('index.html', upload=False)

if __name__ == "__main__":
    app.run(debug=True)