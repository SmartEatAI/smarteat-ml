import streamlit as st
import pickle
import pandas as pd
import numpy as np
import gdown

@st.cache_resource
def cargar_modelos():
    gdown.download(
        "https://drive.google.com/uc?id=10UuMfA8z1KukoWDPvDUdnFCn_xonQwew",
        "files/df_recetas.pkl", quiet=True)
    with open("files/df_recetas.pkl", "rb") as f:
        df = pickle.load(f)

    with open("files/scaler.pkl", "rb") as f:
        scaler = pickle.load(f)

    with open("files/knn.pkl", "rb") as f:
        knn = pickle.load(f)

    return df, scaler, knn


df_recetas, scaler, knn = cargar_modelos()

FEATURES = [
    "calorias", "proteina", "grasa", "carbos",
]

def recomendar_recetas(macros_obj, n=3):

    user_vec = scaler.transform([[
        macros_obj["calorias"],
        macros_obj["proteina"],
        macros_obj["grasa"],
        macros_obj["carbos"]
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
    sexo = st.selectbox("Sexo", ["Hombre", "Mujer"])
    edad = st.number_input("Edad", 10, 100, 28)
    altura = st.number_input("Altura (cm)", 140, 220, 175)
    peso = st.number_input("Peso (kg)", 40.0, 200.0, 75.0)
    actividad = st.selectbox(
        "Nivel de actividad",
        ["Sedentario", "Ligero", "Moderado", "Alto", "Muy alto"]
    )
    objetivo = st.selectbox(
        "Objetivo",
        ["Ganar músculo", "Perder peso", "Mantenimiento"]
    )
    cuerpo = st.selectbox(
        "Tipo de cuerpo",
        ["Delgado", "Normal", "Relleno", "Obeso"]
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
            "calorias": macros["calorias"] / 3,
            "proteina": macros["proteina"] / 3,
            "grasa": macros["grasa"] / 3,
            "carbos": macros["carbos"] / 3
        },
        n=3
    )


if "macros" in st.session_state:
    st.subheader("📊 Macros diarios")
    st.json(st.session_state.macros)

if "recetas" in st.session_state:
    st.subheader("🍽️ Comidas recomendadas")

    for i, receta in st.session_state.recetas.iterrows():
        st.markdown(f"### {receta['name']}")
        st.write("**Macros:**")
        st.write(f"- {float(receta['nutrition'][0])} kcal")
        # st.write(f"- Calorías: {float(receta['nutrition'][0])}")
        # st.write(f"- Proteína: {float(receta['nutrition'][4])} g")
        # st.write(f"- Grasa: {float(receta['nutrition'][1])} g")
        # st.write(f"- Carbohidratos: {float(receta['nutrition'][6])} g")
        st.write("**Ingredientes:**")
        st.write(receta["ingredientes"])

        if st.button("Cambiar por similar", key=f"swap_{receta['id']}"):
            nueva = cambiar_por_similar(
                receta["id"]
            )
            if nueva is not None:
                st.success(f"Alternativa: {nueva['name']}")
                st.write(f"{nueva}")
                st.write("**Macros:**")
                st.write(f"- Calorías: {float(nueva['nutrition'][0])}")
                st.write(f"- Proteína: {float(nueva['nutrition'][4])} g")
                st.write(f"- Grasa: {float(nueva['nutrition'][1])} g")
                st.write(f"- Carbohidratos: {float(nueva['nutrition'][6])} g")

                st.write(nueva["ingredientes"])
