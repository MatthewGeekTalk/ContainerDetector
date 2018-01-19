# coding=utf-8
import os
from flask import Flask, request
import uuid
import cv2
import matplotlib.image as Image
import tensorflow as tf
from Graph import Graph
from plateRec import PlateRec
from util import rotateImage
from util import img_cutter

ALLOWED_EXTENSIONS = ['jpg', 'JPG', 'jpeg', 'JPEG', 'png']
UPLOAD_FOLDER = os.path.abspath('./static')

FREEZE_MODEL_PATH_BC = os.path.abspath('./frozen_module/bc-cnn')
FREEZE_MODEL_PATH_CHAR = os.path.abspath('./frozen_module/char-cnn')

app = Flask(__name__)
app._static_folder = UPLOAD_FOLDER


def load_graph(frozen_graph):
    with tf.gfile.GFile(frozen_graph, "rb") as f:
        graph_def = tf.GraphDef()
        graph_def.ParseFromString(f.read())

    with tf.Graph().as_default() as graph:
        tf.import_graph_def(graph_def, name='prefix')
    return graph


def allowed_files(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1] in ALLOWED_EXTENSIONS


def rename_filename(old_file_name):
    basename = os.path.basename(old_file_name)
    name, ext = os.path.splitext(basename)
    new_name = str(uuid.uuid1()) + ext
    return new_name

def inference(file_name):
    ret_string = ''
    img = cv2.imread(file_name, cv2.COLOR_BGR2RGB)
    plate_rec = PlateRec()
    # Rotate and cut image
    img = rotateImage.docRot(img)
    ratio = [2, 5]
    img_cut = img_cutter.img_cutter(ratio)
    _,img = img_cut.cut(img)
    plate_rec.img = img
    plate_rec.main()

    img_path1 = os.path.abspath('./static')
    new_name = rename_filename(file_name)
    img_path = img_path1 + os.sep + new_name
    if plate_rec._plate_str is not None:
        org_img = cv2.cvtColor(plate_rec._img, cv2.COLOR_BGR2RGB)
        new_url = '/static/%s' % os.path.basename(img_path)
        # new_url = '/static/%s' % os.path.basename(file_name)
        image_tag = '<img src="%s" width=650px></img><p>'
        new_tag = image_tag % new_url
        Image.imsave(img_path, org_img)
        format_string = ''
        for plate_str in plate_rec._plate_str:
            format_string += 'Chars: %s' % plate_str
        ret_string = new_tag + format_string + '<BR>'
        return ret_string
    else:
        ret_string += 'No chars inside picture<BR>'
        return ret_string


@app.route("/", methods=['GET', 'POST'])
def root():
    result = """
    <!doctype html>
    <title>Container recognition</title>
    <h1>Feed your container image here</h1>
    <form action="" method=post enctype=multipart/form-data>
      <p><input type=file name=file value='Choose image'>
         <input type=submit value='Submit'>
    </form>
    <p>%s</p>
    """ % "<br>"
    if request.method == 'POST':
        file = request.files['file']
        old_file_name = file.filename
        if file and allowed_files(old_file_name):
            file_path = os.path.join(UPLOAD_FOLDER, old_file_name)
            file.save(file_path)
            out_html = inference(file_path)

            return result + out_html
    return result

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
    port = int(os.getenv("PORT", 9099))
    # print('listening on port 50060')
    app.run(host='0.0.0.0', port=port)
