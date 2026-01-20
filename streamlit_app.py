import streamlit as st

def calcular_macros_objetivo(sexo, edad, altura, peso, pct_grasa, actividad, objetivo):
    """
    Calcula calorías y macros ajustados según objetivo del usuario
    y devuelve una lista de dietas posibles.
    """
    # Masa magra
    masa_magra = peso * (1 - pct_grasa / 100)

    # BMR (Mifflin-St Jeor)
    if sexo.lower() == "hombre":
        bmr = 10 * peso + 6.25 * altura - 5 * edad + 5
    else:
        bmr = 10 * peso + 6.25 * altura - 5 * edad - 161

    # Factor de actividad
    factores = {
        "sedentario": 1.2,
        "ligero": 1.375,
        "moderado": 1.55,
        "alto": 1.725,
        "muy alto": 1.9
    }
    factor_act = factores.get(actividad.lower(), 1.55)

    # TDEE
    tdee = bmr * factor_act

    # Ajuste por objetivo
    if objetivo == "ganar_musculo":
        calorias = tdee * 1.15  # superávit ~15%
        proteina = round(masa_magra * 2.5)
        grasas = round(peso * 0.8)
    elif objetivo == "perder_peso":
        calorias = tdee * 0.85  # déficit ~15%
        proteina = round(masa_magra * 2.2)
        grasas = round(peso * 0.7)
    else:  # recomposición
        calorias = tdee
        proteina = round(masa_magra * 2.2)
        grasas = round(peso * 0.8)

    # Carbohidratos = resto de calorías
    carbos = round((calorias - (proteina*4 + grasas*9))/4)

    # -----------------------------
    # Lista de posibles dietas
    # -----------------------------
    if objetivo == "ganar_musculo":
        dietas_posibles = ["alta en proteína", "balanceada", "mediterránea"]
    elif objetivo == "perder_peso":
        if pct_grasa > 25:
            dietas_posibles = ["low-carb", "keto", "paleo"]
        else:
            dietas_posibles = ["balanceada", "mediterránea", "flexitariana"]
    else:  # recomposición
        dietas_posibles = ["balanceada", "mediterránea", "flexitariana", "alta en proteína"]

    return {
        "calorias": round(calorias),
        "proteina_g": proteina,
        "grasas_g": grasas,
        "carbos_g": carbos,
        "dietas_posibles": dietas_posibles
    }

usuario = {
    "sexo": "Hombre",
    "edad": 28,
    "altura": 175,
    "peso": 70,
    "%grasa": 18,
    "actividad": "moderado",
    "objetivo": "ganar_musculo"
}

resultado = calcular_macros_objetivo(**usuario)

st.write("Macros y calorías:")
st.write(f"Calorías: {resultado['calorias']}")
st.write(f"Proteína: {resultado['proteina_g']} g")
st.write(f"Grasas: {resultado['grasas_g']} g")
st.write(f"Carbohidratos: {resultado['carbos_g']} g")

st.write("Dietas posibles:")
st.write(resultado['dietas_posibles'])

# Si quieres permitir al usuario seleccionar una dieta:
seleccion_dieta = st.selectbox("Elige tu dieta preferida", resultado['dietas_posibles'])
st.write("Has seleccionado:", seleccion_dieta)
