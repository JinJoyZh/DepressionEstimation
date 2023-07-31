# Filename : example.py
# Copyright : 2020 By Nhooo
# Author by : www.cainiaojc.com
# Date : 2020-08-08
from flask import Flask

app = Flask(__name__)
@app.route('/')
def hello_world():
    return 'Hello World'
if __name__ == '__main__':
    app.run()
