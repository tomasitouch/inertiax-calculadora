from flask import Flask, render_template, request, jsonify, session
import pandas as pd
import os
from openai import OpenAI

app = Flask(__name__)
app.secret_key = "inertiax_secret_key"

# --- Cliente IA ---
client = OpenAI(
    base_url="https://openrouter.ai/api/v1",
    api_key=os.getenv("OPENROUTER_KEY")
)

# --- Página principal ---
@app.route('/')
def index():
    return render_template('index.html')

# --- Subir CSV ---
@app.route('/upload', methods=['POST'])
def upload():
    try:
        file = request.files['file']
        if not file:
            return "No se subió ningún archivo.", 400

        df = pd.read_csv(file)
        session['csv_data'] = df.to_json(orient='records')  # Guardamos datos en sesión temporal

        table_html = df.to_html(classes='table table-striped table-bordered display nowrap', index=False)
        return render_template('index.html', table_html=table_html, filename=file.filename)
    except Exception as e:
        return render_template('index.html', error=f"Error al procesar el archivo: {e}")

# --- Chat con IA ---
@app.route('/chat', methods=['POST'])
def chat():
    try:
        user_message = request.json.get("message", "")
        if not user_message:
            return jsonify({"response": "Por favor, escribe un mensaje."})

        # Recuperamos los datos del CSV si existen
        csv_json = session.get('csv_data', None)
        csv_summary = ""

        if csv_json:
            df = pd.read_json(csv_json)
            # Resumen compacto para el contexto
            csv_summary = f"Columnas: {list(df.columns)}. Primeras filas: {df.head(3).to_dict(orient='records')}."

        prompt = (
            "Eres un analista de rendimiento deportivo. "
            "Responde con claridad y precisión sobre los datos CSV proporcionados.\n\n"
            f"Resumen de datos: {csv_summary}\n\n"
            f"Usuario: {user_message}"
        )

        completion = client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system", "content": "Eres un asistente experto en análisis de datos deportivos."},
                {"role": "user", "content": prompt}
            ]
        )

        reply = completion.choices[0].message.content
        return jsonify({"response": reply})

    except Exception as e:
        return jsonify({"response": f"Error en la IA: {e}"}), 500


if __name__ == '__main__':
    port = int(os.environ.get("PORT", 5000))
    app.run(host='0.0.0.0', port=port)
