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

@app.route('/', methods=['GET', 'POST'])
def upload_file():
    if request.method == 'POST':
        # check if the post request has the file part
        if 'file' not in request.files:
            flash('No file part')
            return redirect(request.url)
        file = request.files['file']
        # If the user does not select a file, the browser submits an
        # empty file without a filename.
        if file.filename == '':
            flash('No selected file')
            return redirect(request.url)
        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            file.save(os.path.join(app.config['UPLOAD_FOLDER'], filename))
            return redirect(url_for('download_file', name=filename))
    return '''
    <!doctype html>
    <title>Upload new File</title>
    <h1>Upload new File</h1>
    <form method=post enctype=multipart/form-data>
      <input type=file name=file>
      <input type=submit value=Upload>
    </form>
    '''
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