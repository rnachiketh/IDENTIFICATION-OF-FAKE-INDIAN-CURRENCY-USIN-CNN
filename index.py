from flask import Flask, render_template, request,flash,session
from DBConfig import DBConnection
from werkzeug.utils import secure_filename
import os
import sys
import pickle
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences
from prediction import image_prediction
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

import warnings
warnings.filterwarnings('ignore')

app = Flask(__name__)
app.secret_key = "123"

@app.route('/')
def index():
    return render_template('index.html')

@app.route("/admin")
def admin():
    return render_template("admin_login.html")

@app.route("/adminlogin_check",methods =["GET", "POST"])
def adminlogin():

        uid = request.form.get("unm")
        pwd = request.form.get("pwd")
        if uid=="admin" and pwd=="admin":

            return render_template("admin_home.html")
        else:
            return render_template("admin_login.html",msg="Invalid Credentials")

@app.route('/admin_home')
def admin_home():
    return render_template('admin_home.html')

@app.route("/dl_evaluations")
def dl_evaluations():

    return render_template("dl_evaluations.html")


@app.route("/user_login")
def user_login():
    return render_template("user_login.html")

@app.route('/user_home')
def user_home():
    return render_template('user_home.html')

@app.route("/newuser")
def newuser():
    return render_template("user_register.html")

@app.route("/user_register",methods =["GET", "POST"])
def user_register():
    try:
        sts=""
        name = request.form.get('name')
        uid = request.form.get('unm')
        pwd = request.form.get('pwd')
        mno = request.form.get('mno')
        email = request.form.get('email')
        database = DBConnection.getConnection()
        cursor = database.cursor()
        sql = "select count(*) from register where userid='" + uid + "'"
        cursor.execute(sql)
        res = cursor.fetchone()[0]
        if res > 0:
            sts = 0
        else:
            sql = "insert into register values(%s,%s,%s,%s,%s)"
            values = (name,uid, pwd,email,mno)
            cursor.execute(sql, values)
            database.commit()
            sts = 1

        if sts==1:
            return render_template("user_login.html", msg="Registered Successfully..! Login Here.")
        else:
            return render_template("user_register.html", msg="User name already exists..!")
    except Exception as e:
        print(e)

    return ""

@app.route("/userlogin_check",methods =["GET", "POST"])
def userlogin_check():

        uid = request.form.get("unm")
        pwd = request.form.get("pwd")
        database = DBConnection.getConnection()
        cursor = database.cursor()
        sql = "select count(*) from register where userid='" + uid + "' and password='" + pwd + "'"
        cursor.execute(sql)
        res = cursor.fetchone()[0]
        if res > 0:
            session['uid'] = uid
            return render_template("user_home.html")
        else:
            return render_template("user_login.html", msg2="Invalid Credentials")
        return ""

@app.route("/detection")
def detection():
    return render_template("detection.html")

@app.route("/prediction",methods =["GET", "POST"])
def prediction():

    image = request.files['image']
    imgdata = secure_filename(image.filename)
    filename = image.filename
    filelist = [f for f in os.listdir("testimg")]
    # print(filelist)
    for f in filelist:
        os.remove(os.path.join("testimg", f))

    image.save(os.path.join("testimg", imgdata))
    image_path = "..\\fake_currency\\testimg\\" + filename
    result = image_prediction(image_path)
    #print("result.....", result)

    return render_template("detection_result.html", result=result)




if __name__ == '__main__':
    app.run(host="localhost", port=4683, debug=True)
