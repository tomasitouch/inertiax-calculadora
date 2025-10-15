from flask import Flask, render_template, request
import pandas as pd
import os

app = Flask(__name__)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload():
    try:
        file = request.files['file']
        if not file:
            return "No se subió ningún archivo.", 400

        df = pd.read_csv(file)
        table_html = df.to_html(classes='table table-striped table-bordered display nowrap', index=False)

        return render_template('index.html', table_html=table_html, filename=file.filename)
    except Exception as e:
        return render_template('index.html', error=f"Error al procesar el archivo: {e}")

if __name__ == '__main__':
    port = int(os.environ.get("PORT", 5000))
    app.run(host='0.0.0.0', port=port)
