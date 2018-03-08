# Copyright (c) 2018 Alaa BEN JABALLAH

import os
from flask import Flask, request, redirect, url_for, send_from_directory, render_template, flash
from werkzeug.utils import secure_filename
import net_training as nt

UPLOAD_FOLDER = 'uploads'
ALLOWED_EXTENSIONS = set(['png', 'jpg', 'jpeg'])

app = Flask(__name__)
network = nt.build_cnn('models/model.npz')
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@app.route('/', methods=['GET', 'POST'])
def upload_file():
    if request.method == 'POST':
        # check if the post request has the file part
        if 'webcam' not in request.files and 'file' not in request.files:
            flash('No file part')
            return redirect(request.url)

        if 'webcam' in request.files:
            file = request.files['webcam']
        else:
            file = request.files['file']

        if file and allowed_file(file.filename):
            filename = "%s.jpg" % str(abs(hash(file.stream)))
            q = file.save(os.path.join(app.config['UPLOAD_FOLDER'], filename))
        
        if 'webcam' in request.files:
            return url_for('uploaded_file',
                                filename=filename)
        return redirect(url_for('uploaded_file',
                            filename=filename))

    return render_template('index.html')

@app.route('/uploads/<filename>')
def send_file(filename):
    return send_from_directory(UPLOAD_FOLDER, filename)

@app.route('/emotions/<filename>')
def uploaded_file(filename):
    faces = nt.load_img(os.path.join(app.config['UPLOAD_FOLDER'], filename))
    tab = nt.evaluate(network, faces) * 100
    # return "anger = {:.2f}%\ncontempt = {:.2f}%\ndisgust = {:.2f}%\nfear = {:.2f}%\nhappy = {:.2f}%\nsadness = {:.2f}%\nsurprise = {:.2f}%".format(*tab[0]*100)
    return render_template('result.html', filename=filename, conf=tab[0],
                           emotions=["anger", "contempt", "disgust", "fear",
                                     "happy", "sadness", "surpsise"],
                           emojis=['angry', 'pensive', 'anguished', 'fearful',
                                   'smile', 'disappointed', 'open_mouth'])
