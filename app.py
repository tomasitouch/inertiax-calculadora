from flask import Flask, render_template, request, session, redirect, url_for, send_file, jsonify
import pandas as pd
import numpy as np
import os
import io
import json
from datetime import datetime
import logging
from werkzeug.utils import secure_filename
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
import warnings
warnings.filterwarnings('ignore')

# Configuración de la aplicación
app = Flask(__name__)
app.secret_key = 'inertiax_secret_key_2024'
app.config['UPLOAD_FOLDER'] = 'uploads'
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max file

# Crear directorio de uploads si no existe
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

# Configurar logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# =============================================================================
# FUNCIONES DE DETECCIÓN Y ANÁLISIS PARA ENCODER HORIZONTAL
# =============================================================================

def detect_encoder_horizontal_file_type(df, filename):
    """
    Detecta automáticamente el tipo de archivo del Encoder Horizontal
    """
    try:
        # Convertir columnas a string para búsqueda
        columns_str = ' '.join(df.columns.astype(str)).lower()
        
        # Verificar por nombre de archivo primero
        filename_lower = filename.lower()
        
        if 'velocidad' in filename_lower and 'distancia' in filename_lower:
            return 'velocidad_vs_distancia'
        elif 'velocidad' in filename_lower:
            return 'velocidad'
        elif 'aceleracion' in filename_lower or 'aceleración' in filename_lower:
            return 'aceleracion'
        elif 'distancia' in filename_lower:
            return 'distancia'
        elif 'resumen' in filename_lower or 'sprint_data' in filename_lower:
            return 'resumen'
        
        # Patrones para identificar tipos de archivos por columnas
        patterns = {
            'resumen': [
                'fecha', 'velocidad_max', 'aceleracion_max', 'aceleracion_media',
                'tiempo_reaccion', 'velocidad/peso'
            ],
            'velocidad': ['tiempo', 'velocidad'],
            'aceleracion': ['tiempo', 'aceleracion'],
            'distancia': ['tiempo', 'distancia'],
            'velocidad_vs_distancia': ['distancia', 'velocidad']
        }
        
        # Verificar por columnas
        for file_type, pattern_list in patterns.items():
            matches = sum(1 for pattern in pattern_list if pattern in columns_str)
            if matches >= 2:  # Al menos 2 coincidencias
                return file_type
        
        # Detección por estructura de datos
        if len(df.columns) >= 6 and any('fecha' in col.lower() for col in df.columns):
            return 'resumen'
        elif len(df.columns) == 2:
            col1, col2 = df.columns[0].lower(), df.columns[1].lower()
            if 'tiempo' in col1 and 'velocidad' in col2:
                return 'velocidad'
            elif 'tiempo' in col1 and 'aceleracion' in col2:
                return 'aceleracion'
            elif 'tiempo' in col1 and 'distancia' in col2:
                return 'distancia'
            elif 'distancia' in col1 and 'velocidad' in col2:
                return 'velocidad_vs_distancia'
        
        return 'desconocido'
        
    except Exception as e:
        logger.error(f"Error en detección de tipo de archivo: {str(e)}")
        return 'desconocido'

def analyze_encoder_horizontal_data(df, file_type, filename):
    """
    Análisis profesional con IA para datos del Encoder Horizontal
    """
    analysis = {
        'tipo_archivo': file_type,
        'nombre_archivo': filename,
        'timestamp_analisis': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
        'estadisticas': {},
        'insights_ia': [],
        'recomendaciones': [],
        'metricas_clave': {},
        'clasificacion_rendimiento': {}
    }
    
    try:
        if file_type == 'resumen':
            return analyze_resumen_data(df, analysis)
        elif file_type == 'velocidad':
            return analyze_velocidad_data(df, analysis)
        elif file_type == 'aceleracion':
            return analyze_aceleracion_data(df, analysis)
        elif file_type == 'distancia':
            return analyze_distancia_data(df, analysis)
        elif file_type == 'velocidad_vs_distancia':
            return analyze_velocidad_distancia_data(df, analysis)
        else:
            analysis['insights_ia'].append("Tipo de archivo no reconocido. Análisis básico aplicado.")
            return analyze_generic_data(df, analysis)
            
    except Exception as e:
        analysis['error'] = f"Error en análisis: {str(e)}"
        logger.error(f"Error en análisis de datos: {str(e)}")
        return analysis

def analyze_resumen_data(df, analysis):
    """Análisis para archivos de resumen"""
    try:
        # Estadísticas básicas
        analysis['estadisticas'] = {
            'total_mediciones': len(df),
            'columnas': df.columns.tolist(),
            'rango_fechas': {
                'inicio': str(df.iloc[0, 0]) if len(df) > 0 else 'N/A',
                'fin': str(df.iloc[-1, 0]) if len(df) > 0 else 'N/A'
            }
        }
        
        # Buscar columnas de velocidad (pueden tener diferentes nombres)
        velocidad_col = None
        for col in df.columns:
            if 'velocidad' in col.lower() and 'max' in col.lower():
                velocidad_col = col
                break
        
        # Análisis de métricas de velocidad (convertir de m/s a km/h)
        if velocidad_col and velocidad_col in df.columns:
            velocidades = pd.to_numeric(df[velocidad_col], errors='coerce').dropna()
            if len(velocidades) > 0:
                velocidades_kmh = velocidades * 3.6  # Convertir a km/h
                analysis['metricas_clave']['velocidad'] = {
                    'maxima_kmh': float(velocidades_kmh.max()),
                    'media_kmh': float(velocidades_kmh.mean()),
                    'minima_kmh': float(velocidades_kmh.min()),
                    'desviacion_kmh': float(velocidades_kmh.std())
                }
                
                # Insights de velocidad
                if velocidades_kmh.max() > 30:  # 30 km/h
                    analysis['insights_ia'].append("🚀 Excelente velocidad máxima detectada (>30 km/h) - Nivel de élite")
                elif velocidades_kmh.max() > 25:
                    analysis['insights_ia'].append("💪 Buena velocidad máxima (25-30 km/h) - Nivel competitivo")
                else:
                    analysis['insights_ia'].append("📈 Oportunidad de mejora en velocidad máxima")
        
        # Buscar columnas de aceleración
        aceleracion_col = None
        for col in df.columns:
            if 'aceleracion' in col.lower() and 'max' in col.lower():
                aceleracion_col = col
                break
        
        # Análisis de aceleración
        if aceleracion_col and aceleracion_col in df.columns:
            aceleraciones = pd.to_numeric(df[aceleracion_col], errors='coerce').dropna()
            if len(aceleraciones) > 0:
                analysis['metricas_clave']['aceleracion'] = {
                    'maxima_mss': float(aceleraciones.max()),
                    'media_mss': float(aceleraciones.mean()),
                    'minima_mss': float(aceleraciones.min())
                }
                
                # Clasificación por aceleración
                avg_accel = aceleraciones.mean()
                if avg_accel > 8:
                    analysis['clasificacion_rendimiento']['aceleracion'] = 'Excelente'
                elif avg_accel > 6:
                    analysis['clasificacion_rendimiento']['aceleracion'] = 'Buena'
                elif avg_accel > 4:
                    analysis['clasificacion_rendimiento']['aceleracion'] = 'Regular'
                else:
                    analysis['clasificacion_rendimiento']['aceleracion'] = 'Necesita mejora'
        
        # Recomendaciones basadas en el análisis
        analysis['recomendaciones'] = [
            "📊 Considera realizar análisis específicos por atleta para personalizar el entrenamiento",
            "🔄 Monitorea la evolución de las métricas clave semanalmente",
            "🎯 Identifica atletas con potencial de mejora en aceleración o velocidad máxima"
        ]
        
        return analysis
        
    except Exception as e:
        analysis['error'] = f"Error en análisis de resumen: {str(e)}"
        return analysis

def analyze_velocidad_data(df, analysis):
    """Análisis para datos de velocidad vs tiempo"""
    try:
        if len(df.columns) >= 2:
            # Convertir a numérico y limpiar datos
            tiempo = pd.to_numeric(df.iloc[:, 0], errors='coerce').dropna()
            velocidad = pd.to_numeric(df.iloc[:, 1], errors='coerce').dropna()
            
            # Asegurarse de que tienen la misma longitud
            min_len = min(len(tiempo), len(velocidad))
            tiempo = tiempo.iloc[:min_len]
            velocidad = velocidad.iloc[:min_len]
            
            if len(tiempo) > 0 and len(velocidad) > 0:
                velocidad_kmh = velocidad * 3.6  # Convertir a km/h
                
                analysis['estadisticas'] = {
                    'puntos_datos': len(df),
                    'duracion_total_s': float(tiempo.max()),
                    'frecuencia_muestreo_hz': 10  # Asumido del código
                }
                
                analysis['metricas_clave'] = {
                    'velocidad_maxima_kmh': float(velocidad_kmh.max()),
                    'tiempo_alcanzar_max_s': float(tiempo[velocidad_kmh == velocidad_kmh.max()].iloc[0]),
                    'velocidad_promedio_kmh': float(velocidad_kmh.mean())
                }
                
                # Calcular aceleración media si hay suficientes datos
                if analysis['metricas_clave']['tiempo_alcanzar_max_s'] > 0:
                    analysis['metricas_clave']['aceleracion_media_kmh_s'] = float(
                        (velocidad_kmh.max() - velocidad_kmh.iloc[0]) / 
                        analysis['metricas_clave']['tiempo_alcanzar_max_s']
                    )
                
                # Análisis de curva de velocidad
                if velocidad_kmh.max() > 0:
                    tiempo_80_percent = tiempo[velocidad_kmh >= 0.8 * velocidad_kmh.max()]
                    if len(tiempo_80_percent) > 0:
                        analysis['metricas_clave']['tiempo_80_porciento_max_s'] = float(tiempo_80_percent.min())
                
                # Insights de IA
                analysis['insights_ia'] = [
                    f"⏱️ Velocidad máxima: {velocidad_kmh.max():.1f} km/h",
                    f"📈 Tiempo para alcanzar máximo: {analysis['metricas_clave']['tiempo_alcanzar_max_s']:.1f}s",
                    f"💡 Aceleración media: {analysis['metricas_clave'].get('aceleracion_media_kmh_s', 0):.1f} km/h/s"
                ]
                
                # Recomendaciones
                analysis['recomendaciones'] = [
                    "🏋️ Trabajar potencia de arranque si el tiempo a velocidad máxima es alto",
                    "🔍 Analizar la técnica de carrera en los primeros 20m",
                    "📊 Comparar con sesiones anteriores para medir progreso"
                ]
        
        return analysis
        
    except Exception as e:
        analysis['error'] = f"Error en análisis de velocidad: {str(e)}"
        return analysis

def analyze_aceleracion_data(df, analysis):
    """Análisis para datos de aceleración vs tiempo"""
    try:
        if len(df.columns) >= 2:
            # Convertir a numérico y limpiar datos
            tiempo = pd.to_numeric(df.iloc[:, 0], errors='coerce').dropna()
            aceleracion = pd.to_numeric(df.iloc[:, 1], errors='coerce').dropna()
            
            # Asegurarse de que tienen la misma longitud
            min_len = min(len(tiempo), len(aceleracion))
            tiempo = tiempo.iloc[:min_len]
            aceleracion = aceleracion.iloc[:min_len]
            
            if len(tiempo) > 0 and len(aceleracion) > 0:
                analysis['estadisticas'] = {
                    'puntos_datos': len(df),
                    'duracion_total_s': float(tiempo.max())
                }
                
                analysis['metricas_clave'] = {
                    'aceleracion_maxima_mss': float(aceleracion.max()),
                    'aceleracion_media_mss': float(aceleracion.mean()),
                    'tiempo_aceleracion_max_s': float(tiempo.iloc[aceleracion.argmax()]),
                    'fuerza_relativa_estimada': float(aceleracion.max() * 75)  # Asumiendo 75kg promedio
                }
                
                # Detección de fatiga
                if len(aceleracion) > 10:
                    primer_tercio = aceleracion[:len(aceleracion)//3].mean()
                    ultimo_tercio = aceleracion[-(len(aceleracion)//3):].mean()
                    if primer_tercio > 0:
                        fatiga = ((primer_tercio - ultimo_tercio) / primer_tercio) * 100
                        analysis['metricas_clave']['porcentaje_fatiga'] = float(fatiga)
                
                # Insights de IA
                analysis['insights_ia'] = [
                    f"💥 Aceleración máxima: {aceleracion.max():.1f} m/s²",
                    f"📊 Fuerza relativa estimada: {analysis['metricas_clave']['fuerza_relativa_estimada']:.0f} N",
                    f"🔋 Porcentaje de fatiga: {analysis['metricas_clave'].get('porcentaje_fatiga', 0):.1f}%"
                ]
                
                # Recomendaciones
                analysis['recomendaciones'] = [
                    "💪 Enfocar en ejercicios de potencia si aceleración máxima es baja",
                    "🔄 Mejorar resistencia a la fatiga con entrenamiento interválico",
                    "🎯 Trabajar técnica de salida y primeros pasos"
                ]
        
        return analysis
        
    except Exception as e:
        analysis['error'] = f"Error en análisis de aceleración: {str(e)}"
        return analysis

def analyze_distancia_data(df, analysis):
    """Análisis para datos de distancia vs tiempo"""
    try:
        if len(df.columns) >= 2:
            # Convertir a numérico y limpiar datos
            tiempo = pd.to_numeric(df.iloc[:, 0], errors='coerce').dropna()
            distancia = pd.to_numeric(df.iloc[:, 1], errors='coerce').dropna()
            
            # Asegurarse de que tienen la misma longitud
            min_len = min(len(tiempo), len(distancia))
            tiempo = tiempo.iloc[:min_len]
            distancia = distancia.iloc[:min_len]
            
            if len(tiempo) > 0 and len(distancia) > 0:
                analysis['estadisticas'] = {
                    'puntos_datos': len(df),
                    'distancia_total_m': float(distancia.max()),
                    'tiempo_total_s': float(tiempo.max())
                }
                
                # Cálculo de velocidades instantáneas
                if len(distancia) > 1:
                    velocidades = []
                    for i in range(1, len(distancia)):
                        if tiempo.iloc[i] != tiempo.iloc[i-1]:
                            v = (distancia.iloc[i] - distancia.iloc[i-1]) / (tiempo.iloc[i] - tiempo.iloc[i-1])
                            velocidades.append(v * 3.6)  # m/s a km/h
                    
                    if velocidades:
                        analysis['metricas_clave'] = {
                            'velocidad_promedio_kmh': float(np.mean(velocidades)),
                            'velocidad_maxima_kmh': float(np.max(velocidades)),
                            'eficiencia_movimiento': float(distancia.max() / tiempo.max())  # m/s
                        }
                
                # Análisis de splits
                splits = {20: None, 30: None, 40: None, 50: None}
                for dist in splits.keys():
                    if any(distancia >= dist):
                        idx = (distancia >= dist).idxmax()
                        splits[dist] = float(tiempo.iloc[idx])
                
                analysis['metricas_clave']['splits_tiempo'] = splits
                
                # Insights de IA
                analysis['insights_ia'] = [
                    f"⏱️ Tiempo total para {distancia.max():.0f}m: {tiempo.max():.2f}s",
                    f"📏 Velocidad promedio: {analysis['metricas_clave']['velocidad_promedio_kmh']:.1f} km/h",
                    f"🎯 Eficiencia de movimiento: {analysis['metricas_clave']['eficiencia_movimiento']:.2f} m/s"
                ]
                
                # Recomendaciones
                analysis['recomendaciones'] = [
                    "📊 Analizar tiempos parciales para identificar segmentos débiles",
                    "💨 Mejorar velocidad en segmentos específicos según splits",
                    "🔄 Trabajar transiciones entre fases de la carrera"
                ]
        
        return analysis
        
    except Exception as e:
        analysis['error'] = f"Error en análisis de distancia: {str(e)}"
        return analysis

def analyze_velocidad_distancia_data(df, analysis):
    """Análisis para datos de velocidad vs distancia"""
    try:
        if len(df.columns) >= 2:
            # Convertir a numérico y limpiar datos
            distancia = pd.to_numeric(df.iloc[:, 0], errors='coerce').dropna()
            velocidad = pd.to_numeric(df.iloc[:, 1], errors='coerce').dropna()
            
            # Asegurarse de que tienen la misma longitud
            min_len = min(len(distancia), len(velocidad))
            distancia = distancia.iloc[:min_len]
            velocidad = velocidad.iloc[:min_len]
            
            if len(distancia) > 0 and len(velocidad) > 0:
                velocidad_kmh = velocidad * 3.6  # Convertir a km/h
                
                analysis['estadisticas'] = {
                    'puntos_datos': len(df),
                    'rango_distancia_m': f"{distancia.min():.1f} - {distancia.max():.1f}",
                    'rango_velocidad_kmh': f"{velocidad_kmh.min():.1f} - {velocidad_kmh.max():.1f}"
                }
                
                analysis['metricas_clave'] = {
                    'velocidad_maxima_kmh': float(velocidad_kmh.max()),
                    'distancia_velocidad_max_m': float(distancia.iloc[velocidad_kmh.argmax()])
                }
                
                # Calcular aceleración promedio
                if analysis['metricas_clave']['distancia_velocidad_max_m'] > distancia.iloc[0]:
                    analysis['metricas_clave']['aceleracion_promedio_kmh_m'] = float(
                        (velocidad_kmh.max() - velocidad_kmh.iloc[0]) / 
                        (analysis['metricas_clave']['distancia_velocidad_max_m'] - distancia.iloc[0])
                    )
                
                # Análisis de eficiencia
                if len(velocidad_kmh) > 5:
                    # Buscar punto de máxima eficiencia (mayor velocidad por distancia)
                    eficiencia = velocidad_kmh / (distancia + 0.1)  # Evitar división por cero
                    idx_max_eff = eficiencia.idxmax()
                    analysis['metricas_clave']['punto_maxima_eficiencia'] = {
                        'distancia_m': float(distancia.iloc[idx_max_eff]),
                        'velocidad_kmh': float(velocidad_kmh.iloc[idx_max_eff]),
                        'indice_eficiencia': float(eficiencia.max())
                    }
                
                # Insights de IA
                analysis['insights_ia'] = [
                    f"🎯 Velocidad máxima de {velocidad_kmh.max():.1f} km/h alcanzada a {analysis['metricas_clave']['distancia_velocidad_max_m']:.1f}m",
                    f"📈 El atleta mantiene buena aceleración hasta los {analysis['metricas_clave'].get('punto_maxima_eficiencia', {}).get('distancia_m', 0):.1f}m",
                    "🔍 Gráfico ideal para analizar estrategia de carrera y distribución de esfuerzo"
                ]
                
                # Recomendaciones
                analysis['recomendaciones'] = [
                    "⚡ Optimizar fase de aceleración basado en curva velocidad-distancia",
                    "🎪 Trabajar mantenimiento de velocidad en distancias específicas",
                    "📋 Ajustar estrategia de carrera según perfil velocidad-distancia"
                ]
        
        return analysis
        
    except Exception as e:
        analysis['error'] = f"Error en análisis de velocidad vs distancia: {str(e)}"
        return analysis

def analyze_generic_data(df, analysis):
    """Análisis genérico para archivos no reconocidos"""
    analysis['estadisticas'] = {
        'filas': len(df),
        'columnas': df.columns.tolist(),
        'tipos_datos': df.dtypes.astype(str).to_dict()
    }
    
    analysis['insights_ia'] = [
        "🔍 Archivo procesado con análisis genérico",
        "💡 Considera usar el formato estándar del Encoder Horizontal para análisis más detallados"
    ]
    
    analysis['recomendaciones'] = [
        "📝 Verifica que el archivo tenga el formato correcto",
        "🔄 Intenta exportar nuevamente desde la app Encoder Horizontal"
    ]
    
    return analysis

# =============================================================================
# RUTAS DE LA APLICACIÓN
# =============================================================================

@app.route('/')
def index():
    """Página principal"""
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload_file():
    """Procesar archivo subido"""
    try:
        if 'file' not in request.files:
            return render_template('index.html', error='No se seleccionó ningún archivo')
        
        file = request.files['file']
        if file.filename == '':
            return render_template('index.html', error='No se seleccionó ningún archivo')
        
        # Leer datos del formulario
        nombre_entrenador = request.form.get('nombre_entrenador', '')
        origen_app = request.form.get('origen_app', '')
        codigo_invitado = request.form.get('codigo_invitado', '')
        
        # Guardar datos del formulario en sesión
        session['form_data'] = {
            'nombre_entrenador': nombre_entrenador,
            'origen_app': origen_app,
            'codigo_invitado': codigo_invitado
        }
        
        # Procesar el archivo
        filename = secure_filename(file.filename)
        file_extension = filename.lower().split('.')[-1]
        
        if file_extension == 'csv':
            df = pd.read_csv(file)
        elif file_extension in ['xls', 'xlsx']:
            df = pd.read_excel(file)
        else:
            return render_template('index.html', error='Formato de archivo no soportado. Use CSV o Excel.')
        
        # Guardar datos en sesión
        session['uploaded_data'] = df.to_dict('list')
        session['filename'] = filename
        
        # Procesamiento según el tipo de app
        if origen_app == 'encoder_horizontal':
            # Detección automática y análisis con IA
            file_type = detect_encoder_horizontal_file_type(df, filename)
            analysis = analyze_encoder_horizontal_data(df, file_type, filename)
            
            # Guardar análisis en sesión
            session['encoder_analysis'] = analysis
            
            # Generar tabla HTML para previsualización
            table_html = df.head(20).to_html(
                classes='table table-striped table-dark', 
                index=False, 
                border=0,
                table_id='dataPreview'
            )
            
            return render_template('index.html', 
                                table_html=table_html,
                                filename=filename,
                                encoder_analysis=analysis,
                                form_data=session['form_data'])
        
        else:
            # Procesamiento para Encoder Vertical (código existente)
            table_html = df.head(20).to_html(
                classes='table table-striped table-dark', 
                index=False, 
                border=0,
                table_id='dataPreview'
            )
            
            return render_template('index.html', 
                                table_html=table_html,
                                filename=filename,
                                form_data=session['form_data'])
            
    except Exception as e:
        logger.error(f"Error en upload_file: {str(e)}")
        return render_template('index.html', error=f'Error procesando archivo: {str(e)}')

@app.route('/generate_report')
def generate_report():
    """Generar reporte completo"""
    try:
        if 'uploaded_data' not in session:
            return redirect('/')
        
        # Aquí iría la lógica para generar el reporte completo
        # Por ahora solo redirigimos a la página principal
        return render_template('index.html', 
                             mensaje='Reporte generado exitosamente',
                             form_data=session.get('form_data', {}))
        
    except Exception as e:
        logger.error(f"Error en generate_report: {str(e)}")
        return render_template('index.html', error=f'Error generando reporte: {str(e)}')

@app.route('/preview_analysis')
def preview_analysis():
    """Vista previa del análisis"""
    try:
        if 'uploaded_data' not in session:
            return redirect('/')
        
        return render_template('index.html', 
                             mensaje='Vista previa del análisis generada',
                             form_data=session.get('form_data', {}))
        
    except Exception as e:
        logger.error(f"Error en preview_analysis: {str(e)}")
        return render_template('index.html', error=f'Error en vista previa: {str(e)}')

@app.route('/clear_session')
def clear_session():
    """Limpiar sesión"""
    session.clear()
    return redirect('/')

# Manejo de errores
@app.errorhandler(413)
def too_large(e):
    return render_template('index.html', error='Archivo demasiado grande. Máximo 16MB.'), 413

@app.errorhandler(404)
def not_found(e):
    return render_template('index.html', error='Página no encontrada'), 404

@app.errorhandler(500)
def internal_error(e):
    return render_template('index.html', error='Error interno del servidor'), 500

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)
