from flask import Flask, render_template,send_from_directory,redirect
import os
from app import *

@server.route('/') 
def home():
    return render_template('home.html')

@server.route('/static/<path:filename>')
def serve_static(filename):
    return server.send_static_file(filename)

@server.route('/introduction')
def introduction():
    return render_template('introduction.html')

@server.route("/dashboard2")
def dashboard():
    return redirect("/dashboard2/")

if __name__ == '__main__':
    server.run(debug=True)