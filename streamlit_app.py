import streamlit as st
import pandas as pd
import numpy as np
from joblib import load
from streamlit_carousel_uui import uui_carousel

# --- CONFIGURACIÓN Y CARGA ---
st.set_page_config(page_title="SmartEatAI")

@st.cache_resource

def cargar_recursos():
    # Cargar archivos
    df = load("files/df_recetas.joblib")
    scaler = load("files/scaler.joblib")
    knn = load("files/knn.joblib")
    
    # Pre-escalar el dataset completo para evitar procesarlo en cada recomendación, para ahorrar CPU
    FEATURES = ['calories', 'fat_content', 'carbohydrate_content', 'protein_content']
    X_scaled_all = scaler.transform(df[FEATURES])
    
    return df, scaler, knn, X_scaled_all

df_recetas, scaler, knn, X_scaled_all = cargar_recursos()

FEATURES = ['calories', 'fat_content', 'carbohydrate_content', 'protein_content']
MACRO_WEIGHTS = np.array([1.5, 0.8, 1.0, 1.2]) # Cal, Fat, Carb, Prot

# --- LÓGICA ---

def recomendar_recetas(macros_obj, n=3):
    # Vector de usuario
    user_vec = np.array([[
        macros_obj["calories"],
        macros_obj["fat_content"],
        macros_obj["carbohydrate_content"],
        macros_obj["protein_content"]
    ]])
    
    # Escalar el vector de usuario
    user_scaled = scaler.transform(user_vec) * MACRO_WEIGHTS
    X_weighted = X_scaled_all * MACRO_WEIGHTS
    
    # Filtrado por dieta
    dietas_usuario = macros_obj.get("dietas", [])
    if dietas_usuario:
        mask = df_recetas["diet_type"].str.contains("|".join(dietas_usuario), case=False, na=False)
        indices_validos = np.where(mask)[0]
        X_search = X_weighted[indices_validos]
        df_search = df_recetas.iloc[indices_validos].copy()
    else:
        X_search = X_weighted
        df_search = df_recetas.copy()

    # Cálculo de distancia ##### REPASAR ESTO #####
    distancias = np.linalg.norm(X_search - user_scaled, axis=1)
    df_search["dist"] = distancias
    
    return df_search.sort_values("dist").head(n).reset_index(drop=True)

def cambiar_por_similar(receta_id, n_busqueda=11):
    # Localizar la receta actual en el DF global
    idx_list = df_recetas.index[df_recetas["id"] == receta_id].tolist()
    if not idx_list:
        return None
    
    # Extraer el vector de características ya escalado (usando nuestra matriz precargada)
    idx_global = idx_list[0]
    vec_receta = X_scaled_all[idx_global].reshape(1, -1)
    
    # El modelo KNN ya está entrenado, lo usamos para buscar vecinos
    dist, indices = knn.kneighbors(vec_receta, n_neighbors=n_busqueda)
    
    # Los resultados de kneighbors son índices del dataframe original
    # Saltamos el primero (que es la misma receta) y elegimos uno al azar de los otros 9
    idx_vecino = indices[0][np.random.randint(1, n_busqueda)]
    
    return df_recetas.iloc[idx_vecino].copy()


# --- FUNCIONES DE CÁLCULO ---
def estimar_pct_grasa(sexo, categoria):
    mapping = {
        "Hombre": {"Delgado": 12, "Normal": 18, "Relleno": 25, "Obeso": 32},
        "Mujer": {"Delgado": 20, "Normal": 26, "Relleno": 33, "Obeso": 40}
    }
    return mapping[sexo][categoria]

# Funcion para calcular las macros del usuario segun los datos de entrada
def calcular_macros(sexo, edad, altura, peso, pct_grasa, actividad, objetivo):
    masa_magra = peso * (1 - pct_grasa / 100)

    # Indice de Masa Corporal (IMC)
    if sexo == "Hombre":
        bmr = 10 * peso + 6.25 * altura - 5 * edad + 5
    else:
        bmr = 10 * peso + 6.25 * altura - 5 * edad - 161

    factores = {
        "Sedentario": 1.2,
        "Ligero": 1.375,
        "Moderado": 1.55,
        "Alto": 1.725,
        "Muy alto": 1.9
    }
    
    # Gasto Energetico Total Diario
    tdee = bmr * factores[actividad]


    # Recomendacion de calorias, proteinas y tipo de dieta
    if objetivo == "Ganar músculo":
        calorias = tdee * 1.1 + 150
        proteina = masa_magra * 2.2
        dietas = ["high_protein", "High Fiber"]
    elif objetivo == "Perder peso":
        calorias = tdee * 0.8
        proteina = masa_magra * 2.2
        dietas = ["low_carb", "Low Calorie"]
    else:
        calorias = tdee
        proteina = masa_magra * 2.0
        dietas = ["Vegetarian", "Vegan", "High Fiber"]

        #### NOTA: Se han cambiado los nombres de las dietas ###

    grasas = (calorias * 0.25) / 9
    carbos = (calorias - (proteina * 4 + grasas * 9)) / 4

    return {
        "calorias": round(calorias),
        "proteina": round(proteina),
        "grasa": round(grasas),
        "carbos": round(carbos),
        "dietas": dietas
    }

# --- INTERFAZ ---

st.title("🥗 SmartEatAI")
st.caption("Recomendador inteligente de comidas según tus macros")

st.header("Configuración de tu Perfil")

with st.form("form_usuario", border=True):
    # Fila 1: Datos Básicos (3 columnas para aprovechar el ancho)
    form_col1, form_col2, form_col3 = st.columns(3)
    
    with form_col1:
        sexo = st.selectbox("Sexo", ["Hombre", "Mujer"])
        altura = st.number_input("Altura (cm)", 140, 220, 175)

    with form_col2:
        edad = st.number_input("Edad", 15, 90, 30)
        peso = st.number_input("Peso (kg)", 40, 200, 75)
    with form_col3:
        n_comidas = st.number_input("Comidas/día", 3, 6, 3)
        cuerpo = st.selectbox("Complexión", ["Delgado", "Normal", "Relleno", "Obeso"])
    
    col_act, col_obj = st.columns(2)
    
    with col_act:
        actividad = st.selectbox(
            "Nivel de Actividad", 
            ["Sedentario", "Ligero", "Moderado", "Alto", "Muy alto"]
        )
    with col_obj:
        objetivo = st.selectbox(
            "Objetivo Principal", 
            ["Ganar músculo", "Perder peso", "Mantenimiento"]
        )
    
    # Botón centrado y destacado
    submit = st.form_submit_button("Generar Plan Personalizado", use_container_width=True, type="primary")

if submit:
    pct = estimar_pct_grasa(sexo, cuerpo)
    macros = calcular_macros(sexo, edad, altura, peso, pct, actividad, objetivo)
    st.session_state.macros = macros # Guardamos las macros en sesion
    
    # Buscamos las comidas dividiendo macros / n_comidas
    st.session_state.recetas = recomendar_recetas({
        "calories": macros["calorias"]/n_comidas, 
        "fat_content": macros["grasa"]/n_comidas,
        "carbohydrate_content": macros["carbos"]/n_comidas,
        "protein_content": macros["proteina"]/n_comidas,
        "dietas": macros["dietas"]
    }, n_comidas)

# --- DISPLAY ---
if "macros" in st.session_state:
    st.subheader("📊 Macros diarios")
    macros = st.session_state.macros
    total_protein = 0
    total_fat = 0
    total_cal = 0
    total_carb = 0
    if "recetas" in st.session_state:
        recetas_df = st.session_state.recetas
        total_protein = recetas_df["protein_content"].sum()
        if "fat_content" in recetas_df.columns:
            total_fat = recetas_df["fat_content"].sum()
        elif "FatContent" in recetas_df.columns:
            total_fat = recetas_df["FatContent"].sum()
        total_cal = recetas_df["calories"].sum()
        total_carb = recetas_df["carbohydrate_content"].sum()

    st.write("**Progreso de macros de las comidas recomendadas:**")
    def macro_bar(label, value, total, color):
        pct = min(1.0, value / total) if total > 0 else 0
        bar_html = f'''<div style="margin-bottom:8px"><b>{label}:</b> {value:.0f} / {total:.0f} <div style='background:#eee;width:100%;height:18px;border-radius:8px;overflow:hidden'><div style='width:{pct*100:.1f}%;height:100%;background:{color};'></div></div></div>'''
        st.markdown(bar_html, unsafe_allow_html=True)

    col1, col2 = st.columns(2)
    with col1:
        macro_bar("Calorías", total_cal, macros["calorias"], "#f39c12")  # naranja
        macro_bar("Grasa", total_fat, macros["grasa"], "#27ae60")  # verde
    with col2:
        macro_bar("Proteína", total_protein, macros["proteina"], "#e74c3c")  # rojo
        macro_bar("Carbohidratos", total_carb, macros["carbos"], "#2980b9")  # azul

    if "dietas" in macros and macros["dietas"]:
        st.write("**Tipos de dieta sugeridos:**")
        label_colors = ["#8e44ad", "#16a085", "#c0392b", "#2980b9", "#f39c12", "#27ae60"]
        labels_html = ""
        for i, dieta in enumerate(macros["dietas"]):
            color = label_colors[i % len(label_colors)]
            labels_html += f"<span style='display:inline-block;background:{color};color:#fff;padding:4px 12px;border-radius:12px;margin-right:8px;margin-bottom:4px;font-size:14px'>{dieta}</span>"
        st.markdown(labels_html, unsafe_allow_html=True)

if "recetas" in st.session_state:
    df_rec = st.session_state.recetas
    st.subheader("🍽️ Comidas recomendadas")

    # Mostrar recetas
    for idx, row in df_rec.iterrows():
    # Creamos un contenedor único para cada receta
        with st.container(border=True):
            st.subheader(f"Comida {idx+1}: {row['name']}")
            
            c1, c2 = st.columns([1, 2])
            
            with c1:
                # SOLUCIÓN AL DUPLICATE ID: Añadimos una key única basada en el ID y el índice
                imgs = row['images'].split(", ")
                slides = [{"image": url, "title": "", "description": ""} for url in imgs[:3]]
                
                uui_carousel(
                    items=slides, 
                    variant="sm", 
                    key=f"carousel_{row['id']}_{idx}"  # <--- Key única aquí
                )
                
            with c2:
                st.write(f"**🔥 Calorías:** {row['calories']} kcal")
                st.write(f"**🥩 Proteína:** {row['protein_content']}g | **🥑 Grasa:** {row['fat_content']}g | **🍞 Carbo:** {row['carbohydrate_content']}g")
                
                # Botón de intercambio con lógica segura
                if st.button(f"🔄 Cambiar por similar", key=f"btn_swp_{row['id']}_{idx}"):
                    nueva_receta = cambiar_por_similar(row['id'])
                    
                    if nueva_receta is not None:
                        # 1. Copiamos el DataFrame actual
                        df_temp = st.session_state.recetas.copy()
                        
                        # 2. Alineamos las columnas de la nueva receta con las del DataFrame
                        # Esto asegura que si existe la columna 'dist', no rompa el código
                        for col in df_temp.columns:
                            if col not in nueva_receta:
                                nueva_receta[col] = 0
                        
                        # 3. REEMPLAZO SEGURO: Usamos .values para evitar conflictos de índices
                        # Seleccionamos solo las columnas que ya existen en el DataFrame de la sesión
                        df_temp.iloc[idx] = nueva_receta[df_temp.columns].values
                        
                        # 4. Actualizamos y refrescamos
                        st.session_state.recetas = df_temp
                        st.rerun()