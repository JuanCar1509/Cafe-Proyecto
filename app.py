import numpy as np
import json
import os
from flask import Flask, render_template, request, redirect, url_for, jsonify
from datetime import datetime
import yfinance as yf

app = Flask(__name__)

# --- CONSTANTES Y CONFIGURACIÓN ---
BODEGAS = ['Sevilla', 'Tuluá', 'Caicedonia']
CAFES = ['Castillo', 'Caturra', 'Borbón']

# Archivos de datos (Persistencia)
FILE_INVENTARIO = 'inventario.json'
FILE_PRECIOS = 'precios.json'
FILE_MATRICES = 'matrices.json'
FILE_HISTORIAL = 'historial.log'
FILE_ESTADO_BOLSA = 'estado_bolsa.json' # Nuevo: Para persistir datos de mercado

# --- GESTIÓN DE DATOS ---

def cargar_json(filepath, default=None):
    """Carga datos de un JSON de forma segura."""
    if not os.path.exists(filepath):
        return default if default is not None else []
    try:
        with open(filepath, 'r') as f:
            return json.load(f)
    except:
        return default

def guardar_json(filepath, data):
    """Guarda datos en un JSON."""
    with open(filepath, 'w') as f:
        json.dump(data, f)

def leer_historial_log():
    """
    Lee el archivo de log y devuelve una lista invertida (más reciente primero).
    CORRECCIÓN: Se agrega errors='replace' para evitar fallos por tildes o caracteres especiales.
    """
    if not os.path.exists(FILE_HISTORIAL):
        return []
    
    # 'errors="replace"' reemplaza caracteres corruptos con ? en lugar de bloquear la app
    with open(FILE_HISTORIAL, 'r', encoding='utf-8', errors='replace') as f:
        lines = f.readlines()
    return list(reversed(lines))

def get_matrices():
    """Carga todas las matrices necesarias como arrays de NumPy."""
    inv_data = cargar_json(FILE_INVENTARIO, [[0,0,0],[0,0,0],[0,0,0]])
    precios_data = cargar_json(FILE_PRECIOS, [0,0,0])
    matrices_data = cargar_json(FILE_MATRICES, {"costos_transporte": [[0,0,0],[0,0,0],[0,0,0]]})
    
    return {
        "inventario": np.array(inv_data),
        "precios": np.array(precios_data),
        "costos_transporte": np.array(matrices_data["costos_transporte"]),
        "calidad": np.array(matrices_data.get("calidad_variedades", [1,1,1]))
    }

def registrar_historial(mensaje):
    """Escribe un evento en el log con fecha y hora."""
    # Usamos utf-8 para asegurar compatibilidad
    with open(FILE_HISTORIAL, 'a', encoding='utf-8') as f:
        f.write(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] {mensaje}\n")

# --- LÓGICA MATEMÁTICA (NIVEL 5) ---

def proyeccion_minimos_cuadrados(cafe_idx):
    """
    Aplica Regresión Lineal (Mínimos Cuadrados) para proyectar agotamiento.
    Resuelve: Beta = (X^T * X)^-1 * X^T * Y
    """
    mats = get_matrices()
    total_actual = np.sum(mats["inventario"][:, cafe_idx])
    
    # Simulación de datos históricos para demostración académica
    dias = np.array(range(10)) # Eje X
    consumo_promedio_diario = 5 
    
    stock_historico = []
    for i in dias:
        # Generar historia hacia atrás con algo de ruido aleatorio
        ruido = np.random.randint(-3, 4) 
        stock = total_actual + (consumo_promedio_diario * (9 - i)) + ruido
        stock_historico.append(max(0, stock))
        
    y = np.array(stock_historico) # Eje Y
    
    # 1. Matriz de Diseño X (Columna de días + Columna de 1s para intercepto)
    X = np.vstack([dias, np.ones(len(dias))]).T
    
    # 2. Ecuación Normal
    XT_X = X.T @ X
    
    try:
        XT_X_inv = np.linalg.inv(XT_X)
    except np.linalg.LinAlgError:
        return {"error": "Matriz singular (datos insuficientes)"}

    XT_y = X.T @ y
    beta = XT_X_inv @ XT_y
    
    m, b = beta # Pendiente (m), Intercepto (b)
    
    # 3. Cálculo del día cero (Agotamiento)
    if m >= 0:
        dias_restantes = 999 
    else:
        dia_cero = -b / m
        dias_restantes = max(0, dia_cero - 9) 
    
    return {
        "pendiente": round(m, 2),
        "intercepto": round(b, 2),
        "dias_agotamiento": round(dias_restantes, 1),
        "datos_x": dias.tolist(),
        "datos_y": y.tolist()
    }

# --- RUTAS FLASK ---

@app.route('/')
def index():
    mats = get_matrices()
    
    # Operación Matricial Básica: Valoración Total
    valor_bodegas = np.dot(mats["inventario"], mats["precios"])
    valor_total = np.sum(valor_bodegas)
    
    # Cargar datos adicionales para el Dashboard
    historial = leer_historial_log()
    estado_bolsa = cargar_json(FILE_ESTADO_BOLSA, {
        "precio_bolsa": 0, 
        "hora_bolsa": "No sincronizado", 
        "trm": 0
    })

    return render_template('index.html',
                           bodegas=BODEGAS,
                           cafes=CAFES,
                           inventario=mats["inventario"].tolist(),
                           precios=mats["precios"].tolist(),
                           valor_bodegas=valor_bodegas.tolist(),
                           valor_total=valor_total,
                           historial=historial,
                           estado_bolsa=estado_bolsa)

@app.route('/api/sincronizar-bolsa', methods=['POST'])
def sincronizar_bolsa():
    """
    Obtiene datos reales de Yahoo Finance y actualiza precios mediante Transformación Lineal.
    Retorna JSON para actualización dinámica sin recarga.
    """
    try:
        # 1. Obtener datos de API
        ticker_cafe = yf.Ticker("KC=F")
        hist = ticker_cafe.history(period="1d")
        
        if hist.empty:
             precio_bolsa_usd_lb = 2.45
             hora_mercado = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        else:
             precio_bolsa_usd_lb = hist['Close'].iloc[-1]
             # Obtener hora del índice del DataFrame
             hora_mercado = hist.index[-1].strftime("%Y-%m-%d %H:%M:%S")
        
        ticker_trm = yf.Ticker("COP=X")
        hist_trm = ticker_trm.history(period="1d")
        trm = hist_trm['Close'].iloc[-1] if not hist_trm.empty else 4100
        
        # 2. Transformación Lineal: T(v) = k * v
        lbs_por_saco = 154 # Aprox 70kg
        mats = get_matrices()
        vector_calidad = mats["calidad"]
        
        escalar_conversion = precio_bolsa_usd_lb * trm * lbs_por_saco
        nuevos_precios = vector_calidad * escalar_conversion
        nuevos_precios = np.round(nuevos_precios, 0).astype(int)
        
        # 3. Guardar cambios
        guardar_json(FILE_PRECIOS, nuevos_precios.tolist())
        
        datos_bolsa = {
            "precio_bolsa": round(precio_bolsa_usd_lb, 2),
            "hora_bolsa": hora_mercado,
            "trm": round(trm, 2)
        }
        guardar_json(FILE_ESTADO_BOLSA, datos_bolsa)
        
        registrar_historial(f"Sincronización Bolsa: {hora_mercado} - ${round(precio_bolsa_usd_lb, 2)}")
        
        return jsonify({
            "status": "success", 
            "nuevos_precios": nuevos_precios.tolist(),
            "datos_math": {
                "precio_bolsa": datos_bolsa["precio_bolsa"],
                "hora_bolsa": datos_bolsa["hora_bolsa"],
                "trm": datos_bolsa["trm"],
                "escalar_total": round(escalar_conversion, 2)
            }
        })
    except Exception as e:
        return jsonify({"status": "error", "message": str(e)}), 500

@app.route('/mover-inventario', methods=['POST'])
def mover_inventario():
    origen = int(request.form['origen'])
    destino = int(request.form['destino'])
    cafe = int(request.form['cafe'])
    cantidad = int(request.form['cantidad'])
    
    mats = get_matrices()
    inv = mats["inventario"]
    
    if inv[origen, cafe] >= cantidad:
        # Actualizar Inventario
        inv[origen, cafe] -= cantidad
        inv[destino, cafe] += cantidad
        guardar_json(FILE_INVENTARIO, inv.tolist())
        
        # Calcular Costo con Matriz de Adyacencia
        costo_unitario = mats["costos_transporte"][origen, destino]
        costo_total = cantidad * costo_unitario
        
        # Log formateado para la tabla del frontend
        msg = f"Logística: Mover {cantidad} {CAFES[cafe]} ({BODEGAS[origen]}->{BODEGAS[destino]}) | Costo: ${costo_total:,}"
        registrar_historial(msg)
        
        return redirect(url_for('index', mensaje_alerta=f"Movimiento exitoso. Costo Logístico: ${costo_total:,}"))
    
    return redirect(url_for('index', mensaje_error="Inventario insuficiente"))

@app.route('/agregar-inventario', methods=['POST'])
def agregar_inventario():
    bodega = int(request.form['bodega'])
    cafe = int(request.form['cafe'])
    cantidad = int(request.form['cantidad'])
    
    mats = get_matrices()
    inv = mats["inventario"]
    
    if cantidad > 0:
        inv[bodega, cafe] += cantidad
        guardar_json(FILE_INVENTARIO, inv.tolist())
        registrar_historial(f"Ingreso: {cantidad} {CAFES[cafe]} a {BODEGAS[bodega]}")
        
    return redirect(url_for('index'))

@app.route('/api/proyeccion/<int:cafe_idx>')
def api_proyeccion(cafe_idx):
    resultado = proyeccion_minimos_cuadrados(cafe_idx)
    return jsonify(resultado)

if __name__ == '__main__':
    app.run(debug=True, port=5000)