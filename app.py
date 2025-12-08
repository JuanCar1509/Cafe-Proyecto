import numpy as np
import json
import os
from flask import Flask, render_template, request, redirect, url_for, jsonify
from datetime import datetime
import yfinance as yf

app = Flask(__name__)

# --- CONSTANTES ---
BODEGAS = ['Sevilla', 'Tuluá', 'Caicedonia']
CAFES = ['Castillo', 'Caturra', 'Borbón']
FILE_INVENTARIO = 'inventario.json'
FILE_PRECIOS = 'precios.json'
FILE_MATRICES = 'matrices.json'
FILE_HISTORIAL = 'historial.log'

# --- GESTIÓN DE DATOS ---
def cargar_json(filepath, default=None):
    if not os.path.exists(filepath):
        return default if default is not None else []
    try:
        with open(filepath, 'r') as f:
            return json.load(f)
    except:
        return default

def guardar_json(filepath, data):
    with open(filepath, 'w') as f:
        json.dump(data, f)

def get_matrices():
    """
    Carga todas las matrices necesarias para las operaciones.
    Si no existen, crea matrices de ceros por defecto.
    """
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
    with open(FILE_HISTORIAL, 'a', encoding='utf-8') as f:
        f.write(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] {mensaje}\n")

# --- LÓGICA DE NEGOCIO Y ÁLGEBRA LINEAL ---

def proyeccion_minimos_cuadrados(cafe_idx):
    """
    NIVEL 5: Aplica Mínimos Cuadrados (Regresión Lineal).
    Matemática: Beta = (X.T * X)^-1 * X.T * Y
    """
    mats = get_matrices()
    # Sumar stock total de ese tipo de café en todas las bodegas
    total_actual = np.sum(mats["inventario"][:, cafe_idx])
    
    # --- SIMULACIÓN DE DATOS HISTÓRICOS ---
    # Para efectos del proyecto final, simulamos una tendencia de consumo
    # basada en el inventario actual para que la gráfica siempre tenga sentido.
    
    dias = np.array(range(10)) # Eje X: 0 a 9
    consumo_promedio_diario = 5 # Simulamos que gastan 5 sacos al día
    
    # Reconstruimos hacia atrás: 
    # Si hoy (día 9) hay X, hace 9 días había X + (9*5)
    stock_historico = []
    for i in dias:
        # Stock teórico + ruido aleatorio para que los puntos no sean perfectos
        ruido = np.random.randint(-3, 4) 
        stock = total_actual + (consumo_promedio_diario * (9 - i)) + ruido
        stock_historico.append(max(0, stock))
        
    y = np.array(stock_historico) # Eje Y
    
    # --- ÁLGEBRA LINEAL PURA ---
    # 1. Matriz de Diseño X (columna de días y columna de 1s para el intercepto)
    X = np.vstack([dias, np.ones(len(dias))]).T
    
    # 2. Ecuación Normal: beta = (X^T * X)^-1 * X^T * y
    XT_X = X.T @ X
    
    try:
        XT_X_inv = np.linalg.inv(XT_X)
    except np.linalg.LinAlgError:
        return {"error": "Matriz singular (datos insuficientes)"}

    XT_y = X.T @ y
    beta = XT_X_inv @ XT_y
    
    m, b = beta # Pendiente (m), Intercepto (b)
    
    # 3. Proyección: ¿Cuándo y será 0?  0 = mx + b  =>  x = -b / m
    if m >= 0:
        dias_restantes = 999 # Si la pendiente es positiva, el stock sube (no se acaba)
    else:
        dia_cero = -b / m
        # Restamos los 9 días que ya pasaron en la simulación
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
    # Valoración = Inventario (3x3) . Precios (3x1)
    # Resultado: Vector de 3 elementos (Valor total por bodega)
    valor_bodegas = np.dot(mats["inventario"], mats["precios"])
    valor_total = np.sum(valor_bodegas)
    
    return render_template('index.html',
                           bodegas=BODEGAS,
                           cafes=CAFES,
                           inventario=mats["inventario"].tolist(),
                           precios=mats["precios"].tolist(),
                           valor_bodegas=valor_bodegas.tolist(),
                           valor_total=valor_total)

@app.route('/api/sincronizar-bolsa', methods=['POST'])
def sincronizar_bolsa():
    """
    NIVEL 4: Transformación Lineal de precios.
    T(v) = k * v
    """
    try:
        # 1. Obtener datos de Yahoo Finance
        ticker_cafe = yf.Ticker("KC=F") # Futuros Café C
        hist = ticker_cafe.history(period="1d")
        
        if hist.empty:
             # Fallback si falla la API (valor promedio reciente)
             precio_bolsa_usd_lb = 2.45 
        else:
             precio_bolsa_usd_lb = hist['Close'].iloc[-1]
        
        ticker_trm = yf.Ticker("COP=X") # Tasa de Cambio Peso Colombiano
        hist_trm = ticker_trm.history(period="1d")
        trm = hist_trm['Close'].iloc[-1] if not hist_trm.empty else 4100
        
        # 2. Transformación Lineal
        lbs_por_saco = 154 # Aprox 70kg * 2.2
        mats = get_matrices()
        vector_calidad = mats["calidad"]
        
        # Escalar Compuesto (Scalar Multiplication)
        escalar_conversion = precio_bolsa_usd_lb * trm * lbs_por_saco
        
        # Operación: Vector * Escalar
        nuevos_precios = vector_calidad * escalar_conversion
        nuevos_precios = np.round(nuevos_precios, 0).astype(int)
        
        guardar_json(FILE_PRECIOS, nuevos_precios.tolist())
        registrar_historial(f"Bolsa Sincronizada: Café {round(precio_bolsa_usd_lb, 2)} USD/lb @ TRM {round(trm, 0)}")
        
        return jsonify({
            "status": "success", 
            "nuevos_precios": nuevos_precios.tolist(),
            "datos_math": {
                "precio_bolsa": round(precio_bolsa_usd_lb, 2),
                "trm": round(trm, 2),
                "escalar_total": round(escalar_conversion, 2),
                "vector_base": vector_calidad.tolist()
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
        # Actualizar cantidad (Resta/Suma elemental)
        inv[origen, cafe] -= cantidad
        inv[destino, cafe] += cantidad
        guardar_json(FILE_INVENTARIO, inv.tolist())
        
        # Calcular costo logístico (Uso de Matriz de Adyacencia/Costos)
        # Costo = Cantidad * Costo_Unitario[origen][destino]
        costo_unitario = mats["costos_transporte"][origen, destino]
        costo_total = cantidad * costo_unitario
        
        registrar_historial(f"Movimiento {cantidad} {CAFES[cafe]} {BODEGAS[origen]}->{BODEGAS[destino]}. Costo: {costo_total}")
        return redirect(url_for('index', mensaje_alerta=f"Movimiento exitoso. Costo Logístico: ${costo_total:,}"))
    
    return redirect(url_for('index', mensaje_error="Inventario insuficiente en origen"))

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
        registrar_historial(f"Ingreso {cantidad} {CAFES[cafe]} a {BODEGAS[bodega]}")
        
    return redirect(url_for('index'))

@app.route('/api/proyeccion/<int:cafe_idx>')
def api_proyeccion(cafe_idx):
    resultado = proyeccion_minimos_cuadrados(cafe_idx)
    return jsonify(resultado)

if __name__ == '__main__':
    app.run(debug=True, port=5000)