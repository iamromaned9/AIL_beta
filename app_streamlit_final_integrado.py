import streamlit as st
import pandas as pd
import numpy as np
import sqlite3
import pickle
import json
import datetime
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    accuracy_score, roc_auc_score, confusion_matrix,
    mean_absolute_error, r2_score
)

def init_user_db():
    conn = sqlite3.connect("usuarios.db")
    c = conn.cursor()
    c.execute('''CREATE TABLE IF NOT EXISTS usuarios (
                    usuario TEXT PRIMARY KEY,
                    password TEXT,
                    rol TEXT
                )''')
    conn.commit()
    conn.close()

def agregar_usuario(usuario, password, rol):
    conn = sqlite3.connect("usuarios.db")
    c = conn.cursor()
    c.execute("INSERT INTO usuarios (usuario, password, rol) VALUES (?, ?, ?)", (usuario, password, rol))
    conn.commit()
    conn.close()

def validar_credenciales(usuario, password):
    conn = sqlite3.connect("usuarios.db")
    c = conn.cursor()
    c.execute("SELECT rol FROM usuarios WHERE usuario=? AND password=?", (usuario, password))
    data = c.fetchone()
    conn.close()
    return data[0] if data else None

def init_log_db():
    conn = sqlite3.connect("resultados_modelos.db")
    c = conn.cursor()
    c.execute('''CREATE TABLE IF NOT EXISTS ejecuciones (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    usuario TEXT,
                    fecha TEXT,
                    tipo_modelo TEXT,
                    metrica1 TEXT,
                    valor1 REAL,
                    metrica2 TEXT,
                    valor2 REAL
                )''')
    conn.commit()
    conn.close()

def log_ejecucion(usuario, tipo, m1, v1, m2, v2):
    conn = sqlite3.connect("resultados_modelos.db")
    c = conn.cursor()
    c.execute("INSERT INTO ejecuciones (usuario, fecha, tipo_modelo, metrica1, valor1, metrica2, valor2) VALUES (?, ?, ?, ?, ?, ?, ?)",
              (usuario, str(datetime.datetime.now()), tipo, m1, v1, m2, v2))
    conn.commit()
    conn.close()

init_user_db()
init_log_db()

st.set_page_config(layout="centered")
st.title("Plataforma Predictiva Integral")
st.caption("Versión final con autenticación obligatoria")

# VALIDACIÓN DE LOGIN FUERTE
if 'usuario' not in st.session_state or st.session_state.usuario is None:
    st.session_state.usuario = None
    st.session_state.rol = None
    st.markdown("### Inicia sesión para continuar")

    with st.form("login"):
        usuario = st.text_input("Usuario")
        password = st.text_input("Contraseña", type="password")
        login = st.form_submit_button("Iniciar sesión")

        if login:
            rol = validar_credenciales(usuario, password)
            if rol:
                st.session_state.usuario = usuario
                st.session_state.rol = rol
                st.success(f"Bienvenido, {usuario} ({rol})")
                st.rerun()
            else:
                st.error("Credenciales inválidas")
    st.stop()

# PANEL ADMIN
if st.session_state.usuario and st.session_state.rol == "admin":
    st.sidebar.header("Registrar nuevo usuario")
    with st.sidebar.form("registro"):
        nuevo_usuario = st.text_input("Nuevo usuario")
        nueva_contra = st.text_input("Nueva contraseña", type="password")
        nuevo_rol = st.selectbox("Rol", ["admin", "analista"])
        registrar = st.form_submit_button("Registrar")

        if registrar:
            try:
                agregar_usuario(nuevo_usuario, nueva_contra, nuevo_rol)
                st.sidebar.success("Usuario registrado correctamente")
            except:
                st.sidebar.error("Ese usuario ya existe")

# FUNCIONALIDAD PRINCIPAL
st.markdown("### Paso 1: Cargar datos desde archivo Excel o CSV")
archivo = st.file_uploader("Selecciona un archivo (.csv o .xlsx)", type=["csv", "xlsx"])

if archivo:
    try:
        if archivo.name.endswith(".csv"):
            df = pd.read_csv(archivo)
        elif archivo.name.endswith(".xlsx"):
            df = pd.read_excel(archivo)
        else:
            st.error("Formato no soportado.")
            df = None

        if df is not None:
            st.session_state.df_api = df
            st.dataframe(df)
            st.success("Datos cargados correctamente.")
    except Exception as e:
        st.error(f"Error al cargar archivo: {str(e)}")

if 'df_api' in st.session_state:
    st.markdown("### Paso 2: Cargar modelo .pkl y ejecutar predicción")
    modelo_file = st.file_uploader("Cargar modelo .pkl", type=["pkl"])
    if modelo_file:
        model = pickle.load(modelo_file)
        df = st.session_state.df_api.copy()
        cols_excluir = ["ID", "Y_real"]
        X_modelo = df.drop(columns=[col for col in cols_excluir if col in df.columns])
        y_pred = model.predict(X_modelo)
        df["Prediccion"] = y_pred

        tipo = "Clasificación" if hasattr(model, "predict_proba") else "Regresión"
        st.write("Tipo de modelo:", tipo)
        st.dataframe(df)

        st.markdown("### Paso 3: Evaluación del modelo")
        if "Y_real" in df.columns:
            if tipo == "Regresión":
                mae = mean_absolute_error(df["Y_real"], df["Prediccion"])
                r2 = r2_score(df["Y_real"], df["Prediccion"])
                st.metric("MAE", f"{mae:.2f}")
                st.metric("R²", f"{r2:.2f}")
                conclusion = "Excelente ajuste." if r2 > 0.85 else "Ajuste aceptable." if r2 > 0.65 else "Modelo mejorable."
                fig, ax = plt.subplots()
                ax.scatter(df["Y_real"], df["Prediccion"], alpha=0.6)
                ax.plot([df["Y_real"].min(), df["Y_real"].max()],
                        [df["Y_real"].min(), df["Y_real"].max()], 'r--')
                ax.set_xlabel("Y Real")
                ax.set_ylabel("Predicción")
                st.pyplot(fig)
                st.info(f"Conclusión: {conclusion}")
                log_ejecucion(st.session_state.usuario, tipo, "MAE", mae, "R2", r2)

            else:
                acc = accuracy_score(df["Y_real"], df["Prediccion"])
                X_auc = df.drop(columns=["Y_real", "Prediccion"]) if "Prediccion" in df.columns else df.drop(columns=["Y_real"])
                auc = roc_auc_score(df["Y_real"], model.predict_proba(X_auc)[:,1]) if hasattr(model, "predict_proba") else 0
                st.metric("Accuracy", f"{acc:.2f}")
                st.metric("AUC", f"{auc:.2f}")
                cm = confusion_matrix(df["Y_real"], df["Prediccion"])
                fig, ax = plt.subplots()
                sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", ax=ax)
                st.pyplot(fig)
                conclusion = "Clasificador sobresaliente." if auc > 0.9 else "Aceptable." if auc > 0.75 else "Modelo débil."
                st.info(f"Conclusión: {conclusion}")
                log_ejecucion(st.session_state.usuario, tipo, "Accuracy", acc, "AUC", auc)
        else:
            st.warning("La columna 'Y_real' no existe. Solo se realizaron predicciones.")

        st.download_button("Descargar resultados", df.to_csv(index=False), file_name="predicciones.csv")

st.markdown("---")
st.subheader("Historial de ejecuciones")
conn = sqlite3.connect("resultados_modelos.db")
historico = pd.read_sql("SELECT * FROM ejecuciones ORDER BY fecha DESC", conn)
st.dataframe(historico)
