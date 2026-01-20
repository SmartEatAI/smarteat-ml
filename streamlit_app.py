import streamlit as st

def calcular_macros_objetivo(sexo, edad, altura, peso, pct_grasa, actividad, objetivo):
    """
    Calcula calorías y macros ajustados según el objetivo del usuario
    y devuelve una lista de dietas posibles.

    Parámetros de entrada:
    - sexo: 'Hombre' o 'Mujer'
    - edad: años
    - altura: cm
    - peso: kg
    - pct_grasa: porcentaje de grasa corporal (%)
    - actividad: nivel de actividad física ('sedentario', 'ligero', 'moderado', 'alto', 'muy alto')
    - objetivo: 'ganar_musculo', 'perder_peso' o 'recomposición'

    Retorna un diccionario con:
    - calorias: calorías diarias necesarias (kcal)
    - proteina_g: gramos de proteína por día
    - grasas_g: gramos de grasa por día
    - carbos_g: gramos de carbohidratos por día
    - dietas_posibles: lista de tipos de dieta sugeridos
    """

    # -----------------------------
    # MASA MAGRA
    # -----------------------------
    # Masa libre de grasa: peso total * (1 - % de grasa)
    masa_magra = peso * (1 - pct_grasa / 100)

    # -----------------------------
    # BMR: TASA METABÓLICA BASAL
    # -----------------------------
    # BMR (Basal Metabolic Rate) = calorías que quemas en reposo
    if sexo.lower() == "hombre":
        bmr = 10 * peso + 6.25 * altura - 5 * edad + 5
    else:  # mujer
        bmr = 10 * peso + 6.25 * altura - 5 * edad - 161

    # -----------------------------
    # FACTOR DE ACTIVIDAD
    # -----------------------------
    # TDEE = Total Daily Energy Expenditure (gasto calórico total)
    factores = {
        "sedentario": 1.2,      # poca actividad
        "ligero": 1.375,        # ejercicio ligero 1-3 días/semana
        "moderado": 1.55,       # ejercicio moderado 3-5 días/semana
        "alto": 1.725,          # entrenamiento intenso 6-7 días/semana
        "muy_alto": 1.9         # trabajo físico intenso o doble entrenamiento
    }
    factor_act = factores.get(actividad.lower(), 1.55)  # 1.55 por defecto si no se reconoce

    # -----------------------------
    # TDEE: CALORÍAS DIARIAS
    # -----------------------------
    # TDEE = calorías necesarias para mantener peso con tu nivel de actividad
    tdee = bmr * factor_act

    # -----------------------------
    # AJUSTE SEGÚN OBJETIVO
    # -----------------------------
    if objetivo == "ganar_musculo":
        # Superávit calórico leve: 10% + 150 kcal extra para favorecer crecimiento muscular
        calorias = tdee * 1.10 + 150
        # Proteína alta: 2.2 g por kg de masa magra para preservar y ganar músculo
        proteina = round(masa_magra * 2.2)
        # Grasas = 25% de las calorías totales (1 g grasa = 9 kcal)
        grasas = round((calorias * 0.25) / 9)
    elif objetivo == "perder_peso":
        # Déficit calórico del 20%, mínimo 1200 kcal/día para seguridad
        calorias = max(tdee * 0.80, 1200)
        # Proteína alta para preservar músculo en déficit
        proteina = round(masa_magra * 2.2)
        # Grasas = 25% de las calorías totales
        grasas = round((calorias * 0.25) / 9)
    else:  # recomposición (mantener peso y mejorar composición corporal)
        calorias = tdee
        proteina = round(masa_magra * 2.0)  # proteína moderada
        grasas = round((calorias * 0.25) / 9)  # 25% calorías en grasa

    # -----------------------------
    # CARBOHIDRATOS
    # -----------------------------
    # Carbohidratos = resto de calorías
    carbos = round((calorias - (proteina * 4 + grasas * 9)) / 4)

    # -----------------------------
    # LISTA DE DIETAS POSIBLES
    # -----------------------------
    # Se sugiere un rango de dietas según objetivo y % de grasa corporal
    if objetivo == "ganar_musculo":
        dietas_posibles = ["alta en proteína", "balanceada", "mediterránea"]
    elif objetivo == "perder_peso":
        if pct_grasa > 25:
            dietas_posibles = ["low-carb", "keto", "paleo"]
        else:
            dietas_posibles = ["balanceada", "mediterránea", "flexitariana"]
    else:  # recomposición
        dietas_posibles = ["balanceada", "mediterránea", "flexitariana", "alta en proteína"]

    # -----------------------------
    # RETORNO
    # -----------------------------
    return {
        "calorias": round(calorias),      # kcal/día
        "proteina_g": proteina,           # g/día
        "grasas_g": grasas,               # g/día
        "carbos_g": carbos,               # g/día
        "dietas_posibles": dietas_posibles
    }

# -----------------------------
# EJEMPLO DE USUARIO
# -----------------------------
usuario = {
    "sexo": "Hombre",
    "edad": 28,
    "altura": 175,       # cm
    "peso": 70,          # kg
    "pct_grasa": 18,     # % grasa corporal
    "actividad": "muy_alto",
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
