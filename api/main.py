import os

from flask import Flask, flash, request, redirect, url_for

from google.cloud import storage
import datetime
app = Flask(__name__)

from werkzeug.utils import secure_filename

UPLOAD_FOLDER = '/path/to/the/uploads'
ALLOWED_EXTENSIONS = {'txt', 'pdf', 'png', 'jpg', 'jpeg', 'gif','h5', 'pt'}
app.config['UPLOAD_FOLDER'] = os.getcwd()
def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@app.route('/')
def main():
    return 'Homepage'

@app.route('/upload', methods=['POST'])
def upload_file():
    # check if the post request has the file part
    if 'files[]' not in request.files:
        resp = jsonify({'message' : 'No file part in the request'})
    resp.status_code = 400
    return resp

files = request.files.getlist('files[]')

errors = {}
success = False

for file in files: 
    if file and allowed_file(file.filename):
        filename = secure_filename(file.filename)
        file.save(os.path.join(app.config['UPLOAD_FOLDER'], filename))
        success = True
    else :
        errors[file.filename] = 'File type is not allowed'

    if success and errors:
        errors['message'] = 'File(s) successfully uploaded'
        resp = jsonify(errors)
        resp.status_code = 500
        return resp
    if success:
        resp = jsonify({'message' : 'Files successfully uploaded'})
        resp.status_code = 201
        return resp
    else:
        resp = jsonify(errors)
        resp.status_code = 500
        return resp
# @app.route("/")
# def upload():
    
#     uploaded_file = flask.request.files.get('file')

#     filename = flask.request.form.get('filename')

#     gcs_client = storage.Client()
#     storage_bucket = gcs_client.get_bucket('tubes-klasifikasi-batik')
#     blob = storage_bucket.blob(uploaded_file.filename)

#     c_type = uploaded_file.content_type
#     blob.upload_from_string(uploaded_file.read(), content_type=c_type)

#     return 
    


if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=int(os.environ.get("PORT", 8080)))