import streamlit as st
import pandas as pd
import numpy as np
import joblib
import json

# -----------------------------
# FUNCIÓN: Calorías y Macros
# -----------------------------
def calcular_macros_objetivo(sexo, edad, altura, peso, pct_grasa, actividad, objetivo):
    """
    Calcula calorías y macros ajustados según objetivo del usuario
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
    # Asignación de tipo de dieta
    # -----------------------------
    dietas = ["keto", "low-carb", "balanceada", "alta en carbohidratos", 
              "alta en proteína", "mediterránea", "vegana", "paleo", "flexitariana"]

    # Reglas simples para seleccionar dieta
    if objetivo == "ganar_musculo":
        tipo_dieta = ["alta en proteína", "balanceada", "mediterránea"]
    elif objetivo == "perder_peso":
        if pct_grasa > 25:
            tipo_dieta = ["low-carb", "keto", "paleo"]
        else:
            tipo_dieta = ["balanceada", "mediterránea", "flexitariana"]
    else:  # recomposición
        tipo_dieta = ["balanceada", "mediterránea", "flexitariana", "alta en proteína"]

    return {
        "calorias": round(calorias),
        "proteina_g": proteina,
        "grasas_g": grasas,
        "carbos_g": carbos,
        "tipo_dieta": tipo_dieta
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

macros = calcular_macros_objetivo(**usuario)
print("Macros y tipo de dieta:")
print(macros)

st.write(macros)