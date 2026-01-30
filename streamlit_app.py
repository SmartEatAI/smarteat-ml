import streamlit as st
import pandas as pd
import numpy as np
import json
from joblib import load
from streamlit_carousel_uui import uui_carousel

# --------------------------------------------------
# CONFIG
# --------------------------------------------------
st.set_page_config(page_title="SmartEatAI", layout="wide")

FEATURES = ['calories', 'fat_content', 'carbohydrate_content', 'protein_content']
MACRO_WEIGHTS = np.array([1.5, 0.8, 1.0, 1.2])

DIET_LABELS = {
    "high_protein": "High Protein",
    "low_carb": "Low Carb",
    "vegan": "Vegan",
    "vegetarian": "Vegetarian",
    "low_calorie": "Low Calorie",
    "high_fiber": "High Fiber"
}

MEAL_COLORS = {
    "Breakfast": "#f39c12",
    "Lunch": "#2980b9",
    "Dinner": "#8e44ad",
    "Snack": "#16a085"
}

# --------------------------------------------------
# LOAD RESOURCES
# --------------------------------------------------
@st.cache_resource
def load_resources():
    df = load("files/df_recetas.joblib")
    scaler = load("files/scaler.joblib")
    knn = load("files/knn.joblib")

    X_scaled_all = scaler.transform(df[FEATURES])
    return df, scaler, knn, X_scaled_all

df_recetas, scaler, knn, X_scaled_all = load_resources()

# --------------------------------------------------
# UTILS
# --------------------------------------------------
def render_tags(tags, color="#34495e"):
    html = ""
    for tag in tags:
        html += f"<span style='background:{color};color:white;padding:4px 10px;border-radius:12px;margin-right:6px;font-size:13px'>{tag}</span>"
    st.markdown(html, unsafe_allow_html=True)

def render_diet_tags(diets):
    html = ""
    for d in diets:
        label = DIET_LABELS.get(d, d)
        html += f"<span style='background:#16a085;color:white;padding:4px 10px;border-radius:12px;margin-right:6px;font-size:13px'>{label}</span>"
    st.markdown(html, unsafe_allow_html=True)

def get_used_recipe_ids():
    if "recipes" not in st.session_state:
        return set()
    return set(st.session_state.recipes["id"].tolist())

# --------------------------------------------------
# RECOMMENDATION LOGIC
# --------------------------------------------------
def recommend_recipes(macros, diets, n):
    user_vec = np.array([[macros[c] for c in FEATURES]])
    user_scaled = scaler.transform(user_vec) * MACRO_WEIGHTS
    X_weighted = X_scaled_all * MACRO_WEIGHTS

    if diets:
        mask = df_recetas["diet_type"].str.contains("|".join(diets), case=False, na=False)
        df_search = df_recetas[mask].copy()
        X_search = X_weighted[mask.values]
    else:
        df_search = df_recetas.copy()
        X_search = X_weighted

    dist = np.linalg.norm(X_search - user_scaled, axis=1)
    df_search["dist"] = dist

    df_sorted = df_search.sort_values("dist")
    return df_sorted.head(n).reset_index(drop=True)

def swap_similar_unique(recipe_id, used_ids, max_tries=20):
    idx = df_recetas.index[df_recetas["id"] == recipe_id].tolist()
    if not idx:
        return None

    vec = X_scaled_all[idx[0]].reshape(1, -1)
    _, indices = knn.kneighbors(vec, n_neighbors=25)

    for _ in range(max_tries):
        candidate = df_recetas.iloc[np.random.choice(indices[0][1:])]
        if candidate["id"] not in used_ids:
            return candidate.copy()

    return None

# --------------------------------------------------
# USER MACROS
# --------------------------------------------------
def estimate_bodyfat(sex, category):
    mapping = {
        "Male": {"Lean": 12, "Normal": 18, "Stocky": 25, "Obese": 32},
        "Female": {"Lean": 20, "Normal": 26, "Stocky": 33, "Obese": 40}
    }
    return mapping[sex][category]

def calculate_macros(sex, age, height, weight, bf, activity, goal):
    lean_mass = weight * (1 - bf / 100)

    if sex == "Male":
        bmr = 10 * weight + 6.25 * height - 5 * age + 5
    else:
        bmr = 10 * weight + 6.25 * height - 5 * age - 161

    activity_factor = {
        "Sedentary": 1.2,
        "Light": 1.375,
        "Moderate": 1.55,
        "High": 1.725,
        "Very High": 1.9
    }[activity]

    tdee = bmr * activity_factor

    if goal == "Gain Muscle":
        calories = tdee * 1.1 + 150
        protein = lean_mass * 2.2
        diets = ["high_protein", "high_fiber"]
    elif goal == "Lose Weight":
        calories = tdee * 0.8
        protein = lean_mass * 2.2
        diets = ["low_carb", "low_calorie"]
    else:
        calories = tdee
        protein = lean_mass * 2.0
        diets = ["vegetarian", "high_fiber"]

    fat = (calories * 0.25) / 9
    carbs = (calories - (protein * 4 + fat * 9)) / 4

    return {
        "calories": round(calories),
        "protein": round(protein),
        "fat": round(fat),
        "carbs": round(carbs),
        "recommended_diets": diets
    }

# --------------------------------------------------
# UI
# --------------------------------------------------
st.title("🥗 SmartEatAI")
st.caption("Intelligent meal recommendations based on your macros")

st.header("Profile Setup")

with st.form("user_form", border=True):
    c1, c2, c3 = st.columns(3)

    with c1:
        sex = st.selectbox("Sex", ["Male", "Female"])
        height = st.number_input("Height (cm)", 140, 220, 175)

    with c2:
        age = st.number_input("Age", 15, 90, 30)
        weight = st.number_input("Weight (kg)", 40, 200, 75)

    with c3:
        meals_per_day = st.number_input("Meals per day", 3, 6, 3)
        body = st.selectbox("Body Type", ["Lean", "Normal", "Stocky", "Obese"])

    activity = st.selectbox("Activity Level", ["Sedentary", "Light", "Moderate", "High", "Very High"])
    goal = st.selectbox("Main Goal", ["Gain Muscle", "Lose Weight", "Maintenance"])

    submit = st.form_submit_button("Generate Personalized Plan", use_container_width=True)

# --------------------------------------------------
# PROCESS
# --------------------------------------------------
if submit:
    bf = estimate_bodyfat(sex, body)
    macros = calculate_macros(sex, age, height, weight, bf, activity, goal)
    st.session_state.macros = macros

# --------------------------------------------------
# DIET SELECTOR
# --------------------------------------------------
if "macros" in st.session_state:
    macros = st.session_state.macros
    recommended = macros["recommended_diets"]

    options = []
    for k, v in DIET_LABELS.items():
        if k in recommended:
            options.append(f"{v} [Recommended]")
        else:
            options.append(v)

    selected = st.multiselect(
        "Diet preferences",
        options,
        default=[o for o in options if "[Recommended]" in o]
    )

    def labels_to_keys(selected_labels):
        keys = []
        for s in selected_labels:
            clean = s.replace(" [Recommended]", "")
            for k, v in DIET_LABELS.items():
                if v == clean:
                    keys.append(k)
        return keys

    selected_diets = labels_to_keys(selected)

    st.session_state.selected_diets = selected_diets

    st.session_state.recipes = recommend_recipes(
        {
            "calories": macros["calories"] / meals_per_day,
            "fat_content": macros["fat"] / meals_per_day,
            "carbohydrate_content": macros["carbs"] / meals_per_day,
            "protein_content": macros["protein"] / meals_per_day,
        },
        selected_diets,
        meals_per_day
    )

# --------------------------------------------------
# DISPLAY RECIPES
# --------------------------------------------------
if "recipes" in st.session_state:
    st.header("🍽️ Recommended Meals")

    for idx, recipe in st.session_state.recipes.iterrows():
        images = recipe["images"].split(", ")
        slides = [{"image": img, "title": recipe["name"], "description": ""} for img in images]
        uui_carousel(slides, variant="md", key=f"carousel_{idx}")

        st.subheader(recipe["name"])

        # Meal types
        meal_types = recipe["meal_type"]
        for mt in meal_types:
            render_tags([mt], MEAL_COLORS.get(mt, "#34495e"))

        # Diets
        render_diet_tags(recipe["diet_type"])

        st.write("**Macros:**")
        st.write(f"Calories: {recipe['calories']} kcal")
        st.write(f"Protein: {recipe['protein_content']} g")
        st.write(f"Fat: {recipe.get('fat_content', 0)} g")
        st.write(f"Carbs: {recipe['carbohydrate_content']} g")

        st.write("**Ingredients:**")
        ingredients = recipe["ingredients"]
        if isinstance(ingredients, str):
            ingredients = json.loads(ingredients)
        for ing in ingredients:
            st.write(f"• {ing}")

        if st.button("Swap for similar", key=f"swap_{recipe['id']}"):
            used = get_used_recipe_ids()
            new_recipe = swap_similar_unique(recipe["id"], used)
            if new_recipe is not None:
                st.session_state.recipes.loc[idx] = new_recipe
                st.rerun()
