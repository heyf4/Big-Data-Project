import tensorflow as tf
import os
import time
from flask import Flask, request, url_for, send_from_directory
import socket
import numpy as np
import keras
from keras.models import load_model
from PIL import Image
from cassandra.cluster import Cluster
from cassandra.query import SimpleStatement

KEYSPACE = "myminstkeyspace"
# cluster = Cluster(contact_points=['127.0.0.1'],port=9043)
# session = cluster.connect()
cluster = Cluster(contact_points=['host.docker.internal'],port=9043)
session = cluster.connect()

session.execute("DROP KEYSPACE %s"%KEYSPACE)

session.execute("""
    CREATE KEYSPACE %s
    WITH replication = { 'class': 'SimpleStrategy', 'replication_factor': '2' }
    """ % KEYSPACE)

session.set_keyspace(KEYSPACE)

session.execute(
    """
    CREATE TABLE minst_table(
        insert_time text,
        insert_name text,
        predict_result int,
        PRIMARY KEY (insert_time)
    )
    """)

if not os.path.exists("uploads"):
    os.mkdir("uploads") 

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = os.getcwd()+"/uploads"

html = '''
    <!DOCTYPE html>
    <title>Upload File</title>
    <h1>PICTURE UPLOAD</h1>
    <form method=post enctype=multipart/form-data>
         <input type=file name=file>
         <input type=submit value=上传>
    </form>
    <p>Hello!</p>
    <p>The predict result of the picture</p>
    <p>{name}</p>
    <p>which is uploaded at {time} is</p>
    <p>{ans}</p>
    '''
@app.route('/', methods=['GET', 'POST'])
def upload_file():
    if request.method == 'POST':
        file = request.files['file']
        if file:
            timestamp=time.strftime("%Y-%m-%d-%H_%M_%S",time.localtime(time.time()))
            filename=timestamp +".png"
            file.save(os.path.join(app.config['UPLOAD_FOLDER'], filename))
            size=(28,28)
            dir_img="uploads/"
            img_name=filename
            
            ori_image = Image.open(dir_img+img_name)
            ori_image= ori_image.convert('L')
            pre_image = ori_image.resize(size, Image.ANTIALIAS)

            pre_image=np.array(pre_image).reshape((1,28,28,1))

            keras.backend.clear_session() 
            myModel = load_model('mnistmodel.h5')

            predict = myModel.predict(pre_image)
            predict = np.argmax(predict)
            
            session.execute(
                """
                INSERT INTO minst_table (insert_time, insert_name, predict_result)
                VALUES (%(insert_time)s, %(insert_name)s, %(predict_result)s)
                """,
                {'insert_time': timestamp, 'insert_name': img_name, 'predict_result': predict}
            )

            return html.format(name=img_name,time=timestamp,ans=predict)
    return html

if __name__ == "__main__":
    app.run(host='0.0.0.0', port=80)