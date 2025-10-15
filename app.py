from flask import Flask, render_template, request, jsonify
import pandas as pd
import os
from openai import OpenAI

app = Flask(__name__)

# --- Configuración del cliente de IA ---
client = OpenAI(
    base_url="https://openrouter.ai/api/v1",
    api_key=os.getenv("OPENROUTER_KEY")  # define tu clave en Render
)

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

@app.route('/chat', methods=['POST'])
def chat():
    try:
        user_message = request.json.get("message", "")
        if not user_message:
            return jsonify({"response": "Por favor, escribe un mensaje."})

        completion = client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system", "content": "Eres un asistente especializado en analizar datos de rendimiento deportivo."},
                {"role": "user", "content": user_message}
            ]
        )

        reply = completion.choices[0].message.content
        return jsonify({"response": reply})

    except Exception as e:
        return jsonify({"response": f"Error en la IA: {e}"}), 500

if __name__ == '__main__':
    port = int(os.environ.get("PORT", 5000))
    app.run(host='0.0.0.0', port=port)
