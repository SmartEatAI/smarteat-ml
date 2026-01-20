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

def estimar_pct_grasa(sexo, categoria):
    if sexo.lower() == "hombre":
        mapa = {
            "Delgado": 12,
            "Normal": 18,
            "Relleno": 25,
            "Obeso": 32
        }
    else:
        mapa = {
            "Delgado": 20,
            "Normal": 26,
            "Relleno": 33,
            "Obeso": 40
        }

    return mapa.get(categoria, 18)

TODAS_LAS_DIETAS = [
    "alta en proteína",
    "balanceada",
    "mediterránea",
    "low-carb",
    "keto",
    "paleo",
    "flexitariana",
    "vegetariana",
    "vegana",
    "sin gluten",
    "halal"
]

# Aqui comienza la aplicación Streamlit

st.set_page_config(page_title="SmartEatAI", layout="centered")
st.title("SmartEatAI - Cálculo de Macros")

st.write("Introduce tus datos físicos y tu objetivo. Calcularemos tus macros diarios de forma personalizada.")

col1, col2 = st.columns(2)

with col1:
    sexo = st.selectbox("Sexo", ["Hombre", "Mujer"], key="sexo")
    edad = st.number_input("Edad (años)", min_value=10, max_value=100, value=28, key="edad")
    altura = st.number_input("Altura (cm)", min_value=120, max_value=230, value=175, key="altura")
    actividad = st.selectbox(
            "Nivel de actividad",
            ["Sedentario", "Ligero", "Moderado", "Alto", "Muy_alto"],
            key="actividad"
        )

with col2:
    peso = st.number_input("Peso (kg)", min_value=30.0, max_value=250.0, value=70.0, key="peso")

    modo_grasa = st.radio(
        "¿Cómo quieres indicar tu grasa corporal?",
        ["Seleccionar tipo de cuerpo", "Introducir % exacto"],
        key="modo_grasa"
    )

    if modo_grasa == "Seleccionar tipo de cuerpo":
        categoria_grasa = st.selectbox(
            "Tipo de cuerpo",
            ["Delgado", "Normal", "Relleno", "Obeso"],
            key="categoria_grasa"
        )
        pct_grasa_input = None
    else:
        pct_grasa_input = st.number_input(
            "Porcentaje de grasa corporal (%)",
            min_value=5.0,
            max_value=60.0,
            value=18.0,
            key="pct_grasa_input"
        )
        categoria_grasa = None

    

    objetivo = st.selectbox(
        "Objetivo",
        ["ganar_musculo", "perder_peso", "recomposición"],
        key="objetivo"
    )

calcular = st.button("Calcular macros")

if calcular:
    if modo_grasa == "Seleccionar tipo de cuerpo":
        pct_grasa = estimar_pct_grasa(sexo, categoria_grasa)
    else:
        pct_grasa = pct_grasa_input

    resultado = calcular_macros_objetivo(
        sexo=sexo,
        edad=edad,
        altura=altura,
        peso=peso,
        pct_grasa=pct_grasa,
        actividad=actividad,
        objetivo=objetivo
    )

    st.subheader("Resultados personalizados")

    colA, colB = st.columns(2)

    with colA:
        st.metric("Calorías (kcal/día)", resultado["calorias"])
        st.metric("Proteína (g/día)", resultado["proteina_g"])

    with colB:
        st.metric("Grasas (g/día)", resultado["grasas_g"])
        st.metric("Carbohidratos (g/día)", resultado["carbos_g"])

    st.caption(f"Porcentaje de grasa usado en el cálculo: {round(pct_grasa, 1)}%")

    st.subheader("Opciones de dieta")

    dietas_recomendadas = resultado["dietas_posibles"]

    # Construir etiquetas UI
    opciones_ui = []
    for dieta in TODAS_LAS_DIETAS:
        if dieta in dietas_recomendadas:
            opciones_ui.append(f"{dieta}  [RECOMENDADA]")
        else:
            opciones_ui.append(dieta)

    # Preseleccionar las recomendadas
    default_ui = [
        f"{dieta}  [RECOMENDADA]"
        for dieta in dietas_recomendadas
    ]

    seleccion_ui = st.multiselect(
        "Elige una o varias dietas según tus preferencias",
        opciones_ui,
        default=default_ui
    )


