import os
from flask import Flask, request,render_template, redirect, url_for
from werkzeug.utils import secure_filename
from werkzeug.datastructures import  FileStorage
import tensorflow as tf
import requests
from os.path import basename
import random

ALLOWED_EXTENSIONS = set(['jpg', 'jpeg'])

app = Flask('Loading Time')
app = Flask(__name__)

ALLOWED_EXTENSIONS = set(['jpg', 'jpeg'])

class Load():
    def __init__(self):
        self.dir_path = os.path.dirname(os.path.realpath(__file__))
        self.label_lines = [line.rstrip() for line
                            in tf.io.gfile.GFile(os.path.join(self.dir_path, "models/retrained_labels.txt"))]
        self.create_graph()
        self.sess = tf.compat.v1.Session()

    def create_graph(self):
        with tf.compat.v1.gfile.FastGFile(os.path.join(self.dir_path, "models/retrained_graph.pb"), 'rb') as f:
            graph_def = tf.compat.v1.GraphDef()
            graph_def.ParseFromString(f.read())
            _ = tf.import_graph_def(graph_def, name='')

    def run(self, image_path):
        image_data = tf.compat.v1.gfile.FastGFile(image_path, 'rb').read()
        # Feed the image_data as input to the graph and get first prediction
        softmax_tensor = self.sess.graph.get_tensor_by_name('final_result:0')

        predictions = self.sess.run(softmax_tensor,  {'DecodeJpeg/contents:0': image_data})

        # Sort to show labels of first prediction in order of confidence
        top_k = predictions[0].argsort()[-len(predictions[0]):][::-1]

        #print('Result for %s' % (image_path))
        ret = ''

        #human_string = self.label_lines[node_id]
        # score = predictions[0][node_id]

        score_0 = predictions[0][0]
        score_1 = predictions[0][1]
        score_2 = predictions[0][2]

        if score_0 > score_1 and score_0 > score_2:
            human_string = self.label_lines[0]
            ret += human_string
        elif score_1 > score_0 and score_1 > score_2:
            human_string = self.label_lines[1]
            ret += human_string
        elif score_2 > score_1 and score_2 > score_0:
            human_string = self.label_lines[2]
            ret += human_string
        #print(ret)

        return ret


app.load = Load()


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/", methods=['GET','POST'])
def uploaded_file():

    image = request.form['data']

    #print(image)

    thisis = app.load.run(image)
    print(thisis)

    #response = requests.post('http://localhost:2323/', data={'data': image})

    #ret = response.json()
    return thisis

app.debug=True
app.run(host='0.0.0.0', port=3000)