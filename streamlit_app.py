import streamlit as st
import pickle
import ast
import pandas as pd
import numpy as np
import gdown
from joblib import load

@st.cache_resource
def cargar_modelos():
    df = load("files/df_recetas.joblib")
    scaler = load("files/scaler.joblib")
    knn = load("files/knn.joblib")

    return df, scaler, knn


df_recetas, scaler, knn = cargar_modelos()

st.dataframe(df_recetas)

FEATURES = [
    'calories',
    'fat_content',
    'carbohydrate_content',
    'protein_content'
]

def recomendar_recetas(macros_obj, n=3):

    user_vec = scaler.transform([[
        macros_obj["calories"],
        macros_obj["protein_content"],
        macros_obj["fat_content"],
        macros_obj["carbohydrate_content"]
    ]])

    X_scaled = scaler.transform(df_recetas[FEATURES])

    distances, idxs = knn.kneighbors(
        user_vec,
        n_neighbors=min(n, len(df_recetas))
    )

    return df_recetas.iloc[idxs[0]]


def cambiar_por_similar(receta_id):

    idx = df_recetas[df_recetas["id"] == receta_id].index[0]

    X_rec = scaler.transform([df_recetas.loc[idx, FEATURES]])

    distances, idxs = knn.kneighbors(X_rec, n_neighbors=4)

    for i in idxs[0][1:]:
        return df_recetas.iloc[i]

    return None


def estimar_pct_grasa(sexo, categoria):
    if sexo == "Hombre":
        return {"Delgado": 12, "Normal": 18, "Relleno": 25, "Obeso": 32}[categoria]
    else:
        return {"Delgado": 20, "Normal": 26, "Relleno": 33, "Obeso": 40}[categoria]

def calcular_macros(sexo, edad, altura, peso, pct_grasa, actividad, objetivo):
    masa_magra = peso * (1 - pct_grasa / 100)

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

    tdee = bmr * factores[actividad]

    if objetivo == "Ganar músculo":
        calorias = tdee * 1.1 + 150
        proteina = masa_magra * 2.2
        dietas = ["alta en proteína", "balanceada"]
    elif objetivo == "Perder peso":
        calorias = tdee * 0.8
        proteina = masa_magra * 2.2
        dietas = ["low-carb", "keto"]
    else:
        calorias = tdee
        proteina = masa_magra * 2.0
        dietas = ["balanceada", "mediterránea"]

    grasas = (calorias * 0.25) / 9
    carbos = (calorias - (proteina * 4 + grasas * 9)) / 4

    return {
        "calorias": round(calorias),
        "proteina": round(proteina),
        "grasa": round(grasas),
        "carbos": round(carbos),
        "dietas": dietas
    }

st.set_page_config(
    page_title="SmartEatAI",
    layout="centered"
)

st.title("🥗 SmartEatAI")
st.caption("Recomendador inteligente de comidas según tus macros")

with st.form("form_usuario"):
    form_col1, form_col2 = st.columns(2)
    with form_col1:
        sexo = st.selectbox("Sexo", ["Hombre", "Mujer"])
        altura = st.number_input("Altura (cm)", 140, 220, 175)
        actividad = st.selectbox(
            "Nivel de actividad",
            ["Sedentario", "Ligero", "Moderado", "Alto", "Muy alto"]
        )

    with form_col2:
        edad = st.number_input("Edad", 10, 100, 28)
        peso = st.number_input("Peso (kg)", 40.0, 200.0, 75.0)
        cuerpo = st.selectbox(
            "Tipo de cuerpo",
            ["Delgado", "Normal", "Relleno", "Obeso"]
        )

    objetivo = st.selectbox(
            "Objetivo",
            ["Ganar músculo", "Perder peso", "Mantenimiento"]
        )
    submit = st.form_submit_button("Calcular y recomendar")

if submit:
    pct = estimar_pct_grasa(sexo, cuerpo)

    macros = calcular_macros(
        sexo, edad, altura, peso, pct, actividad, objetivo
    )

    st.session_state.macros = macros
    st.session_state.recetas = recomendar_recetas(
        {
            "calories": macros["calorias"] / 3,
            "protein_content": macros["proteina"] / 3,
            "fat_content": macros["grasa"] / 3,
            "carbohydrate_content": macros["carbos"] / 3
        },
        n=3
    )


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
    st.subheader("🍽️ Comidas recomendadas")

    recetas_df = st.session_state.recetas.copy()
    for idx, receta in recetas_df.iterrows():
        st.markdown(f"### {receta['name']}")
        st.write("**Macros:**")
        st.write(f"- Calorías: {receta['calories']} kcal")
        st.write(f"- Proteína: {receta['protein_content']} g")
        st.write(f"- Grasa: {receta.get('fat_content', receta.get('FatContent', 0))} g")
        st.write(f"- Carbohidratos: {receta['carbohydrate_content']} g")
        st.write("**Ingredientes:**")
        st.write(receta['recipe_ingredient_parts'])

        if st.button("Cambiar por similar", key=f"swap_{receta['id']}_{idx}"):
            nueva = cambiar_por_similar(receta["id"])
            if nueva is not None:
                st.session_state.recetas.loc[idx] = nueva
                st.session_state._rerun = True

if st.session_state.get('_rerun', False):
    st.session_state._rerun = False
    st.rerun()
