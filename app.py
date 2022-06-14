import os
import pickle

import numpy as np
import pandas as pd
import transformers
import torch

from flask import Flask, request, redirect, url_for, render_template, flash
from transformers import TextClassificationPipeline
from werkzeug.utils import secure_filename
import matplotlib.pyplot as plt

UPLOAD_FOLDER = 'upload'

ALLOWED_EXTENSIONS = set(['csv', 'json'])

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
model = pickle.load(open('model/second_model.sav', 'rb'))
tokenizer = torch.hub.load('huggingface/pytorch-transformers', 'tokenizer', 'DeepPavlov/rubert-base-cased')
pipe = TextClassificationPipeline(model=model, tokenizer=tokenizer, return_all_scores=True)
labels = ['anger', 'sadness', 'neutral', 'joy', 'surprise', 'shame', 'disgust', 'fear']


def get_emotion_value(emotion):
    if emotion == 'anger':
        return -4
    if emotion == 'sadness':
        return -2
    if emotion == 'neutral':
        return 0
    if emotion == 'joy':
        return 2
    if emotion == 'surprise':
        return 5
    if emotion == 'shame':
        return -1
    if emotion == 'disgust':
        return -3.5
    if emotion == 'fear':
        return -3


@app.route("/")
def redirect_to_upload():
    return redirect(request.url + 'upload')


@app.route("/upload")
def get_upload():
    return render_template('upload.html')


@app.route("/result/<name>/<column_text>/<column_time>/<column_id>", methods=['GET', 'POST'])
def redirect_to_result(name, column_text, column_time, column_id):
    data = pd.read_csv('upload/' + name)
    fig, ax = plt.subplots(nrows=1, ncols=1)
    plt.yticks([-4, -3.5, -3, -2, -1, 0, 2, 5],
               ['злость', 'отвращ.', 'страх', 'грусть', 'стыд', 'нейтр.', 'счастье', 'удивл.'])
    plt.xlabel('время')

    user_count = data[column_id].nunique(dropna=False)
    dfs = dict(tuple(data.groupby(column_id)))
    print(user_count)
    for i in range(user_count):
        x = []
        y = []
        for index, row in dfs[i].iterrows():
            print(row[column_text])
            piper = pipe(row[column_text])
            list = [d['score'] for d in piper[0]]
            print(labels[np.argmax(list)])
            x.append(row[column_time])
            y.append(get_emotion_value(labels[np.argmax(list)]))
            plt.plot(x, y, marker="o", markersize=5)

    fig.savefig('static/plot.png')
    plt.close(fig)
    return render_template('result.html')


@app.route("/data/<name>", methods=['GET', 'POST'])
def post_result(name):
    file_format = name[name.rfind('.'):]
    current_name = name
    if file_format == '.csv':
        data = pd.read_csv('upload/' + name)
    else:
        data = pd.read_json('upload/' + name)
        current_name = name[:name.rfind('.')] + '.csv'
        data.to_csv('upload/' + current_name)

    columns = data.columns.values.tolist()

    if request.method == 'POST':
        return redirect(url_for('redirect_to_result', name=current_name, column_text=request.form.get('columnText'),
                                column_time=request.form.get('columnTime'), column_id=request.form.get('columnId')))

    return render_template('data.html', tables=[data.to_html()], titles=[''], columns=columns)


def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


@app.route('/upload', methods=['GET', 'POST'])
def upload_file():
    if request.method == 'POST':
        if 'file' not in request.files:
            print('No file part')
            return redirect(request.url)
        file = request.files['file']
        if file.filename == '':
            print('No selected file')
            return redirect(request.url)
        if allowed_file(file.filename):
            filename = secure_filename(file.filename)
            file.save(os.path.join(app.config['UPLOAD_FOLDER'], filename))
            return redirect(url_for('post_result', name=filename))
    return


@app.route('/')
def hello_world():
    return 'Hello World!'


@app.errorhandler(500)
def page_not_found(e):
    return render_template('error_500.html')


if __name__ == '__main__':
    app.run()
