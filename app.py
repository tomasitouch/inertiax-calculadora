from flask import Flask, render_template, request
import pandas as pd
import os

app = Flask(__name__)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload():
    try:
        file = request.files['file']
        if not file:
            return "No se subió ningún archivo.", 400

        # Leer el CSV con pandas
        df = pd.read_csv(file)
        # Convertir a HTML (Bootstrap para mejor estilo)
        table_html = df.to_html(classes='table table-striped table-bordered', index=False)

        return render_template('index.html', table_html=table_html, filename=file.filename)

    except Exception as e:
        return f"Error al procesar el archivo: {e}", 500


if __name__ == '__main__':
    port = int(os.environ.get("PORT", 5000))
    app.run(host='0.0.0.0', port=port)
