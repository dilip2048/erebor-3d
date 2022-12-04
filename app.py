from flask import Flask

app = Flask(__name__)


@app.route('/')
def run_script():
    file = open(r'/Users/dilipyadav/githome/erebor/face_capture.py', 'r').read()
    return exec(file)


if __name__ == "__main__":
    app.run(debug=True)
