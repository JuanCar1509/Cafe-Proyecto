import numpy as np
import json
from flask import Flask, render_template, request, redirect, url_for
from datetime import datetime

app = Flask(__name__)

# --- Nombres para mostrar en la UI ---
BODEGAS = ['Sevilla', 'Tuluá', 'Caicedonia']
CAFES = ['Castillo', 'Caturra', 'Borbón']

# --- Funciones para manejar datos (igual que antes) ---
def cargar_datos():
    with open('inventario.json', 'r') as f:
        inventario = np.array(json.load(f))
    with open('precios.json', 'r') as f:
        precios = np.array(json.load(f))
    return inventario, precios

def guardar_inventario(inventario):
    with open('inventario.json', 'w') as f:
        json.dump(inventario.tolist(), f)

def guardar_precios(precios):
    with open('precios.json', 'w') as f:
        json.dump(precios.tolist(), f)
        
def registrar_movimiento(mensaje):
    # Escribir nuevos movimientos en UTF-8 para estandarizar la codificación
    with open('historial.log', 'a', encoding='utf-8') as f:
        f.write(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] {mensaje}\n")

def leer_historial(max_lineas=50):
    # Devuelve las últimas `max_lineas` del historial.log como lista de líneas (más reciente primero)
    try:
        # Leer en binario y decodificar intentando UTF-8, luego CP1252 (Windows),
        # y finalmente con replacement para evitar excepciones por bytes inválidos.
        with open('historial.log', 'rb') as f:
            raw = f.read()
        # Intentar UTF-8
        try:
            text = raw.decode('utf-8')
        except UnicodeDecodeError:
            # Intentar CP1252 (común en Windows)
            try:
                text = raw.decode('cp1252')
            except UnicodeDecodeError:
                # Como último recurso, decodificar reemplazando bytes inválidos
                text = raw.decode('utf-8', errors='replace')

        lineas = text.strip().splitlines()
        if not lineas:
            return []
        # Tomar las últimas `max_lineas` y devolver en orden cronológico descendente (más reciente primero)
        return list(reversed(lineas[-max_lineas:]))
    except FileNotFoundError:
        # Crear archivo vacío en UTF-8 para futuras escrituras
        with open('historial.log', 'w', encoding='utf-8'):
            pass
        return []

# --- Rutas de la Aplicación Web ---

@app.route('/')
def index():
    # Cargar datos frescos
    inventario, precios = cargar_datos()
    
    # --- ¡LA MAGIA DEL ÁLGEBRA LINEAL! ---
    # Multiplicamos la matriz de inventario (3x3) por el vector de precios (3x1)
    valor_por_bodega = np.dot(inventario, precios)
    valor_total = np.sum(valor_por_bodega)
    
    # Pasamos todos los datos a la plantilla HTML para que los muestre
    historial = leer_historial(200)
    return render_template('index.html', 
                           bodegas=BODEGAS, 
                           cafes=CAFES, 
                           inventario=inventario, 
                           precios=precios,
                           valor_por_bodega=valor_por_bodega,
                           valor_total=valor_total,
                           historial=historial)

@app.route('/actualizar-precios', methods=['POST'])
def actualizar_precios():
    # Obtenemos los nuevos precios del formulario
    nuevos_precios = [
        int(request.form['precio_castillo']),
        int(request.form['precio_caturra']),
        int(request.form['precio_borbon'])
    ]
    guardar_precios(np.array(nuevos_precios))
    return redirect(url_for('index')) # Redirigimos a la página principal

@app.route('/mover-inventario', methods=['POST'])
def mover_inventario():
    # Obtenemos los datos del formulario de movimiento
    bodega_origen_idx = int(request.form['origen'])
    bodega_destino_idx = int(request.form['destino'])
    cafe_idx = int(request.form['cafe'])
    cantidad = int(request.form['cantidad'])
    
    inventario, _ = cargar_datos()
    
    # Validamos que haya suficiente café para mover
    if inventario[bodega_origen_idx, cafe_idx] >= cantidad:
        # --- Operación matricial ---
        inventario[bodega_origen_idx, cafe_idx] -= cantidad
        inventario[bodega_destino_idx, cafe_idx] += cantidad
        guardar_inventario(inventario)
        # Log
        mensaje = f"Movimiento de {cantidad} sacos de {CAFES[cafe_idx]} desde {BODEGAS[bodega_origen_idx]} a {BODEGAS[bodega_destino_idx]}"
        registrar_movimiento(mensaje)

    return redirect(url_for('index'))

if __name__ == '__main__':
    app.run(debug=True) # El modo debug ayuda a ver errores al instante