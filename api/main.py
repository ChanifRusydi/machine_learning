import os

from flask import Flask
from google.cloud import storage
import datetime
app = Flask(__name__)


@app.route("/")
def upload():
    
    uploaded_file = flask.request.files.get('file')

    filename = flask.request.form.get('filename')

    gcs_client = storage.Client()
    storage_bucket = gcs_client.get_bucket('tubes-klasifikasi-batik')
    blob = storage_bucket.blob(uploaded_file.filename)

    c_type = uploaded_file.content_type
    blob.upload_from_string(uploaded_file.read(), content_type=c_type)

    return 
    


if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=int(os.environ.get("PORT", 8080)))