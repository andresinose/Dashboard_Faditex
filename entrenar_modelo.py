import pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler
import joblib

def main():
    print("Cargando datos del CSV")
    # Cargar los datos
    df = pd.read_csv('dataset_faditex_limpio_y_analizado.csv')
    
    # Seleccionar solo las 5 columnas que usa tu modelo web
    caracteristicas = ['co2', 'humedad', 'ruido', 'temperatura', 'tvoc']
    
    # Eliminar posibles filas con valores nulos para evitar errores
    df = df.dropna(subset=caracteristicas)
    
    print(" Normalizando los datos")
    # Crear y entrenar el escalador
    scaler = StandardScaler()
    X_escalado = scaler.fit_transform(df[caracteristicas])
    
    print("Entrenando modelo Isolation Forest")
    # IMPORTANTE: Reducimos la 'contaminacion' del 5% al 1% para que el modelo
    # sea menos estricto y arroje menos "falsos positivos" (alertas rojas inválidas).
    modelo_if = IsolationForest(contamination=0.01, random_state=42)
    modelo_if.fit(X_escalado)
    
    print(" Guardando los modelos en formato .pkl")
    joblib.dump(scaler, 'escalador_faditex.pkl')
    joblib.dump(modelo_if, 'modelo_if_faditex.pkl')
    
    print("Los archivos 'escalador_faditex.pkl' y 'modelo_if_faditex.pkl' han sido creados en esta carpeta.")

if __name__ == "__main__":
    main()
