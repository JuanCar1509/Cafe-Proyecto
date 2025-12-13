import numpy as np  # Librería para matemáticas avanzadas (Matrices y Vectores)
import json         # Librería para leer y escribir archivos de texto .json
import os           # Librería para manejar rutas de archivos en el sistema operativo
from flask import Flask, render_template, request, redirect, url_for, jsonify
from datetime import datetime
import yfinance as yf # Librería para conectarse a la Bolsa de Valores (Yahoo Finance)

# Inicializamos la aplicación Flask 
app = Flask(__name__)

# --- CONSTANTES Y CONFIGURACIÓN ---
# Definimos los nombres fijos de nuestras bodegas y tipos de café
BODEGAS = ['Sevilla', 'Tuluá', 'Caicedonia']
CAFES = ['Castillo', 'Caturra', 'Borbón']

# Nombres de los archivos donde guardamos los datos 
FILE_INVENTARIO = 'inventario.json'
FILE_PRECIOS = 'precios.json'
FILE_MATRICES = 'matrices.json'
FILE_HISTORIAL = 'historial.log'
FILE_ESTADO_BOLSA = 'estado_bolsa.json' 

# --- GESTIÓN DE DATOS (LECTURA Y ESCRITURA) ---

def cargar_json(filepath, default=None):
    """
    Intenta abrir un archivo JSON. Si no existe o falla, devuelve un valor por defecto.
    Es como intentar leer un cuaderno: si no está, empezamos con una hoja en blanco.
    """
    if not os.path.exists(filepath):
        return default if default is not None else []
    try:
        with open(filepath, 'r') as f:
            return json.load(f)
    except:
        return default

def guardar_json(filepath, data):
    """
    Escribe datos en un archivo JSON.
    """
    with open(filepath, 'w') as f:
        json.dump(data, f)

def leer_historial_log():
    """
    Lee el archivo de historial y lo devuelve al revés (lo más nuevo primero).
    Usamos errors='replace' para que si hay una tilde extraña, no se rompa el programa.
    """
    if not os.path.exists(FILE_HISTORIAL):
        return []
    
    with open(FILE_HISTORIAL, 'r', encoding='utf-8', errors='replace') as f:
        lines = f.readlines()
    return list(reversed(lines))

def registrar_historial(mensaje):
    """
    Agrega una nueva línea al archivo de historial con la fecha y hora actual.
    """
    with open(FILE_HISTORIAL, 'a', encoding='utf-8') as f:
        f.write(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] {mensaje}\n")

def get_matrices():
    """
    FUNCIÓN CLAVE: Convierte los datos de los archivos JSON en MATRICES DE NUMPY.
    Esto es necesario para poder hacer las multiplicaciones matemáticas rápidas.
    """
    inv_data = cargar_json(FILE_INVENTARIO, [[0,0,0],[0,0,0],[0,0,0]])
    precios_data = cargar_json(FILE_PRECIOS, [0,0,0])
    matrices_data = cargar_json(FILE_MATRICES, {"costos_transporte": [[0,0,0],[0,0,0],[0,0,0]]})
    
    return {
        "inventario": np.array(inv_data),           # Matriz 3x3 (Bodegas x Cafés)
        "precios": np.array(precios_data),          # Vector de precios actuales
        "costos_transporte": np.array(matrices_data["costos_transporte"]), # Matriz de Adyacencia (Mapa de costos)
        "calidad": np.array(matrices_data.get("calidad_variedades", [1,1,1])) # Vector de calidad
    }

# --- LÓGICA MATEMÁTICA AVANZADA (NIVEL 5) ---

def proyeccion_minimos_cuadrados(cafe_idx):
    """
    EL CEREBRO PREDICTIVO:
    Aplica Regresión Lineal (Mínimos Cuadrados) para predecir cuándo se acabará el café.
    Usa la Ecuación Normal: Beta = (X^T * X)^-1 * X^T * Y
    """
    mats = get_matrices()
    
    # 1. Obtenemos el total actual de ese tipo de café (Suma de todas las bodegas)
    total_actual = np.sum(mats["inventario"][:, cafe_idx])
    
    # --- SIMULACIÓN DE DATOS ---
    # Creamos un historial ficticio de los últimos 10 días para poder hacer la gráfica
    dias = np.array(range(10)) # Eje X: Días 0 al 9
    consumo_promedio_diario = 5 
    
    stock_historico = []
    for i in dias:
        # Generamos datos hacia atrás con un poco de "ruido" aleatorio para que parezca real
        ruido = np.random.randint(-3, 4) 
        stock = total_actual + (consumo_promedio_diario * (9 - i)) + ruido
        stock_historico.append(max(0, stock))
        
    y = np.array(stock_historico) # Eje Y: Cantidad de Café
    
    # --- MATEMÁTICA PURA ---
    
    # 2. Creamos la Matriz de Diseño X
    # Apilamos una columna de días y una columna de unos (para el intercepto)
    X = np.vstack([dias, np.ones(len(dias))]).T
    
    # 3. Resolvemos la Ecuación Normal
    XT_X = X.T @ X  # Multiplicamos la Transpuesta por la Original
    
    try:
        XT_X_inv = np.linalg.inv(XT_X) # Calculamos la Inversa 
    except np.linalg.LinAlgError:
        return {"error": "Matriz singular (datos insuficientes)"}

    XT_y = X.T @ y
    beta = XT_X_inv @ XT_y # Calculamos Beta (El resultado final)
    
    m, b = beta # Desempaquetamos: m = Pendiente, b = Intercepto
    
    # 4. Interpretamos el futuro (Predicción)
    if m >= 0:
        dias_restantes = 999 # Si la pendiente es positiva, el inventario sube, nunca se acaba.
    else:
        # Fórmula: x = -b / m (Cuando y será 0)
        dia_cero = -b / m
        # Restamos 9 porque estamos en el día 9 de la simulación
        dias_restantes = max(0, dia_cero - 9) 
    
    return {
        "pendiente": round(m, 2),
        "intercepto": round(b, 2),
        "dias_agotamiento": round(dias_restantes, 1),
        "datos_x": dias.tolist(),
        "datos_y": y.tolist()
    }

# --- RUTAS DE LA PÁGINA WEB (FLASK) ---

@app.route('/')
def index():
    """Ruta principal: Muestra el Dashboard."""
    mats = get_matrices()
    
    # ÁLGEBRA LINEAL: Producto Punto para valoración
    # Multiplicamos (Matriz Inventario) . (Vector Precios) para saber cuánto vale cada bodega
    valor_bodegas = np.dot(mats["inventario"], mats["precios"])
    valor_total = np.sum(valor_bodegas)
    
    # Cargamos historial y estado de bolsa para mostrarlos
    historial = leer_historial_log()
    estado_bolsa = cargar_json(FILE_ESTADO_BOLSA, {
        "precio_bolsa": 0, 
        "hora_bolsa": "No sincronizado", 
        "trm": 0
    })

    # Enviamos todos los datos al HTML para que se pinten en pantalla
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
    EL AGENTE DE VIAJES:
    Se conecta a Yahoo Finance, trae el precio internacional y actualiza nuestros precios locales.
    """
    try:
        # 1. Obtener precio del Café (Futuros C) de NYBOT
        ticker_cafe = yf.Ticker("KC=F")
        hist = ticker_cafe.history(period="1d")
        
        if hist.empty:
             # Si la bolsa está cerrada, usamos un valor por defecto
             precio_bolsa_usd_lb = 2.45
             hora_mercado = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        else:
             # Tomamos el último precio de cierre
             precio_bolsa_usd_lb = hist['Close'].iloc[-1]
             hora_mercado = hist.index[-1].strftime("%Y-%m-%d %H:%M:%S")
        
        # 2. Obtener precio del Dólar (TRM)
        ticker_trm = yf.Ticker("COP=X")
        hist_trm = ticker_trm.history(period="1d")
        trm = hist_trm['Close'].iloc[-1] if not hist_trm.empty else 4100
        
        # 3. TRANSFORMACIÓN LINEAL: T(v) = k * v
        # Calculamos el nuevo precio base en Pesos
        lbs_por_saco = 154 # Un saco estándar son aprox 70kg (154 lbs)
        mats = get_matrices()
        vector_calidad = mats["calidad"]
        
        # Escalar mágico: Precio USD * Dólar * Libras
        escalar_conversion = precio_bolsa_usd_lb * trm * lbs_por_saco
        
        # Multiplicamos el vector de calidades por el escalar
        nuevos_precios = vector_calidad * escalar_conversion
        nuevos_precios = np.round(nuevos_precios, 0).astype(int) # Redondeamos a enteros
        
        # 4. Guardamos todo
        guardar_json(FILE_PRECIOS, nuevos_precios.tolist())
        
        datos_bolsa = {
            "precio_bolsa": round(precio_bolsa_usd_lb, 2),
            "hora_bolsa": hora_mercado,
            "trm": round(trm, 2)
        }
        guardar_json(FILE_ESTADO_BOLSA, datos_bolsa)
        
        registrar_historial(f"Sincronización Bolsa: {hora_mercado} - ${round(precio_bolsa_usd_lb, 2)}")
        
        # Respondemos con JSON para que la página se actualice sin recargar
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
    """
    LOGÍSTICA: Mueve inventario de una bodega a otra y calcula el costo de transporte.
    """
    # Obtenemos los datos del formulario web
    origen = int(request.form['origen'])
    destino = int(request.form['destino'])
    cafe = int(request.form['cafe'])
    cantidad = int(request.form['cantidad'])
    
    # Verificamos que no intenten mover a la misma bodega
    if origen == destino:
        return redirect(url_for('index', mensaje_error="Error Lógico: La bodega de origen y destino no pueden ser la misma."))
    
    mats = get_matrices()
    inv = mats["inventario"]
    
    # Verificamos si hay suficiente café en el origen
    if inv[origen, cafe] >= cantidad:
        # Ejecutamos el movimiento
        inv[origen, cafe] -= cantidad # Restamos del origen
        inv[destino, cafe] += cantidad # Sumamos al destino
        guardar_json(FILE_INVENTARIO, inv.tolist())
        
        # Calculamos Costo usando la MATRIZ DE ADYACENCIA (Costos de transporte)
        # Buscamos en la tabla de costos la intersección entre origen y destino
        costo_unitario = mats["costos_transporte"][origen, destino]
        costo_total = cantidad * costo_unitario
        
        # Registramos el evento
        msg = f"Logística: Mover {cantidad} {CAFES[cafe]} ({BODEGAS[origen]}->{BODEGAS[destino]}) | Costo: ${costo_total:,}"
        registrar_historial(msg)
        
        return redirect(url_for('index', mensaje_alerta=f"Movimiento exitoso. Costo Logístico: ${costo_total:,}"))
    
    # Si no hay suficiente inventario, damos error
    return redirect(url_for('index', mensaje_error="Error: Inventario insuficiente en la bodega de origen."))

@app.route('/agregar-inventario', methods=['POST'])
def agregar_inventario():
    """Ruta simple para añadir stock manualmente."""
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
    """API que llama a nuestra función matemática de predicción."""
    resultado = proyeccion_minimos_cuadrados(cafe_idx)
    return jsonify(resultado)

# Arrancar la aplicación
if __name__ == '__main__':
    app.run(debug=True, port=5000)