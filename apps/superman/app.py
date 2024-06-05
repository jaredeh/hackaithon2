from flask import Flask, render_template
import requests

app = Flask(__name__)


@app.route('/')
def index():
    response = requests.get("http://127.0.0.1:5000/get", json={"key": "lego-superman.jpeg", "service": "superman"})
    image = response.json()
    return render_template('index.html', image=image, title="Batman")

if __name__ == '__main__':
    app.run(debug=True, port=5001)
