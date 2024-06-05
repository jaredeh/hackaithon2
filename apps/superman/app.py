from flask import Flask, render_template
import requests

app = Flask(__name__)


@app.route('/')
def index():
    response = requests.post("http://127.0.0.1:5000/get", json={"key": "lego-superman.jpeg", "service": "superman"})
    
    data = response.json()  # This is where the JSONDecodeError can occur
    print(data)

    image = data.get('file_path')
    return render_template('index.html', image=image, title="Superman")

if __name__ == '__main__':
    app.run(debug=True, port=5002)
