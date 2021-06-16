#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jan  9 16:59:03 2019

@author: onee
"""

from flask import Flask, render_template, request
from werkzeug import secure_filename

import os

import face_test

app = Flask(__name__)

@app.route('/')
def indexPage():
    return render_template('index.html')
#	return '<h1>Hello world!</h1>'

@app.route("/registPage", methods=['GET']) 
def registPage():
    return render_template('regist_page.html')

@app.route("/registPage/regist", methods=['POST']) 
def registFace():
    if not(os.path.isdir("data")):
        os.makedirs(os.path.join("data"))
        
    name = request.form['face_name']
    if not(os.path.isdir(os.path.join("data", str(name)))):
        os.makedirs(os.path.join("data", str(name)))
        
    file = request.files['face_img']
    file.save("data/"+name+"/"+secure_filename(file.filename))
        
        
    return render_template('regist_page.html') #팝업뜨게..로 변경

@app.route("/recogPage", methods=['GET'])
def recogPage():
    return render_template('recog_page.html')

    
@app.route("/recogPage/recog", methods=['POST']) 
def recogFace():
    file = request.files['input_img']
    
    file_name = "static/"+secure_filename(file.filename)
    file.save(file_name)
    
    person_names = face_test.find_face(file_name)
    person_names = list(set(person_names))#일단 하나씩 출력되도록
    
    dic = {}
    for i in range(len(person_names)):
        dic['name'+str(i)] = person_names[i]
    
#    dic = {'img_src': '../'+file_name, 'names': person_names}
        
    return render_template('recog_result.html', img_src='../'+file_name, names=dic)


if __name__ == '__main__':
	app.run(debug=True)