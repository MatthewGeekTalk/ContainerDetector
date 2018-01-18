# coding=utf-8
from flask import Flask, request,jsonify
import os
import cv2
import matplotlib.image as Image
import tensorflow as tf
from Graph import Graph
from plateRec import PlateRec

app = Flask(__name__)

FREEZE_MODEL_PATH_BC = os.path.abspath('./frozen_module/bc-cnn')
FREEZE_MODEL_PATH_CHAR = os.path.abspath('./frozen_module/char-cnn')
UPLOAD_FOLDER = os.path.abspath('./upload')
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

def load_graph(frozen_graph):
    with tf.gfile.GFile(frozen_graph, "rb") as f:
        graph_def = tf.GraphDef()
        graph_def.ParseFromString(f.read())

    with tf.Graph().as_default() as graph:
        tf.import_graph_def(graph_def, name='prefix')
    return graph

def inference(file_name):
    ret_string = ''
    img = cv2.imread(file_name, cv2.COLOR_BGR2RGB)
    plate_rec = PlateRec()
    plate_rec.img = img
    plate_rec.main()

    if plate_rec._plate_str is not None:
        format_string = ''
        for plate_str in plate_rec._plate_str:
            format_string += plate_str
        return format_string
    else:
        ret_string += 'No chars inside picture<BR>'
        return ret_string

@app.route("/container_detect", methods=['GET', 'POST'])
def get_tasks():
    if request.method == 'POST':
        file = request.files['file']
        materials = os.listdir(os.path.abspath(UPLOAD_FOLDER))
        new_filename = 'pic' + str(len(materials)) + '.jpg'
        f = os.path.join(app.config['UPLOAD_FOLDER'], new_filename)
        file.save(f)
        out_html = inference(f)
    return jsonify({'container_detect': out_html})

if __name__ == "__main__":
    print('start')
    print(os.path.abspath('./'))
    graph_bc = load_graph(FREEZE_MODEL_PATH_BC \
                          + '/frozen_model.pb')
    graph_char = load_graph(FREEZE_MODEL_PATH_CHAR \
                            + '/frozen_model.pb')
    sess_bc = tf.Session(graph=graph_bc)
    sess_char = tf.Session(graph=graph_char)
    graph = Graph()
    graph.graph_bc = graph_bc
    graph.sess_bc = sess_bc
    graph.graph_char = graph_char
    graph.sess_char = sess_char
    port = int(os.getenv("PORT", 8088))
    app.run(host='0.0.0.0', port=port)
