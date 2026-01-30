import streamlit as st
import pandas as pd
import numpy as np
import json
from joblib import load
from streamlit_carousel_uui import uui_carousel

# --- CONFIGURACIÓN Y CARGA ---
st.set_page_config(page_title="SmartEatAI")

@st.cache_resource

def load_resources():
    # Cargar archivos
    df = load("files/df_recetas.joblib")
    scaler = load("files/scaler.joblib")
    knn = load("files/knn.joblib")

    # Pre-escalar el dataset completo para evitar procesarlo en cada recomendación, para ahorrar CPU
    FEATURES = ['calories', 'fat_content', 'carbohydrate_content', 'protein_content']
    X_scaled_all = scaler.transform(df[FEATURES])

    return df, scaler, knn, X_scaled_all

df_recipes, scaler, knn, X_scaled_all = load_resources()

FEATURES = ['calories', 'fat_content', 'carbohydrate_content', 'protein_content']
MACRO_WEIGHTS = np.array([1.5, 0.8, 1.0, 1.2]) # Cal, Fat, Carb, Prot

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

LABEL_COLORS = ["#8e44ad", "#16a085", "#c0392b", "#2980b9", "#f39c12", "#27ae60"]

# --- UTILIDADES ---
def render_tags(tags, color="#34495e"):
    html = ""
    for tag in tags:
        html += f"<span style='background:{color};color:white;padding:4px 10px;border-radius:12px;margin-right:6px;font-size:13px'>{tag}</span>"
    st.markdown(html, unsafe_allow_html=True)

def render_diet_tags(diets):
    html = ""
    for i, d in enumerate(diets):
        label = DIET_LABELS.get(d, d)
        color = LABEL_COLORS[i % len(LABEL_COLORS)]
        html += f"<span style='background:{color};color:white;padding:4px 10px;border-radius:12px;margin-right:6px;font-size:13px'>{label}</span>"
    st.markdown(html, unsafe_allow_html=True)

def safe_to_list(value):
    """
    Convierte distintos formatos a lista de strings:
    - Lista real -> lista
    - JSON string -> lista
    - String separado por comas -> lista
    - None / NaN -> []
    """
    if value is None:
        return []

    # Si ya es lista
    if isinstance(value, list):
        return [str(v).strip() for v in value if str(v).strip()]

    # Si es string
    if isinstance(value, str):
        value = value.strip()

        if not value:
            return []

        # Intentar JSON
        try:
            parsed = json.loads(value)
            if isinstance(parsed, list):
                return [str(v).strip() for v in parsed if str(v).strip()]
        except Exception:
            pass

        # Fallback: separado por comas
        return [v.strip() for v in value.split(",") if v.strip()]

    return []

def get_used_recipe_ids(exclude_id=None):
    if (
        "recipes" not in st.session_state
        or st.session_state.recipes is None
        or st.session_state.recipes.empty
        or "id" not in st.session_state.recipes.columns
    ):
        return set()

    ids = set(st.session_state.recipes["id"].dropna().tolist())

    if exclude_id is not None:
        ids.discard(exclude_id)

    return ids

def recommend_recipes(macros_obj, diets, n=3, used_ids=None):
    if used_ids is None:
        used_ids = set()

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
    if diets:
        mask = df_recipes["diet_type"].str.contains("|".join(diets), case=False, na=False)
        valid_indices = np.where(mask)[0]
        X_search = X_weighted[valid_indices]
        df_search = df_recipes.iloc[valid_indices].copy()
    else:
        X_search = X_weighted
        df_search = df_recipes.copy()

    # Quitar recetas ya usadas
    if used_ids:
        mask_used = ~df_search["id"].isin(used_ids)
        df_search = df_search[mask_used]
        X_search = X_search[mask_used.values]

    if df_search.empty:
        return pd.DataFrame()

    # Cálculo de distancia ##### REPASAR ESTO #####
    distances = np.linalg.norm(X_search - user_scaled, axis=1)
    df_search["dist"] = distances

    # Eliminar duplicados por ID, manteniendo el de menor distancia
    df_search = df_search.sort_values("dist").drop_duplicates(subset=["id"], keep="first")

    return df_search.sort_values("dist").head(n).reset_index(drop=True)

def swap_for_similar(recipe_id, n_search=11, exclude_ids=None):
    if exclude_ids is None:
        exclude_ids = set()

    # Localizar la receta actual en el DF global
    idx_list = df_recipes.index[df_recipes["id"] == recipe_id].tolist()
    if not idx_list:
        return None

    # Extraer el vector de características ya escalado (usando nuestra matriz precargada)
    idx_global = idx_list[0]
    recipe_vec = X_scaled_all[idx_global].reshape(1, -1)

    # El modelo KNN ya está entrenado, lo usamos para buscar vecinos
    dist, indices = knn.kneighbors(recipe_vec, n_neighbors=n_search)

    # Filtrar vecinos que ya están siendo mostrados (excluyendo la receta actual)
    valid_neighbors = []
    for idx in indices[0][1:]:  # Saltamos el primero (que es la misma receta)
        neighbor_id = df_recipes.iloc[idx]["id"]
        if neighbor_id not in exclude_ids:
            valid_neighbors.append(idx)

    # Si no hay vecinos válidos, retornar None
    if not valid_neighbors:
        return None

    # Elegir uno al azar de los vecinos válidos
    neighbor_idx = valid_neighbors[np.random.randint(0, len(valid_neighbors))]

    return df_recipes.iloc[neighbor_idx].copy()


# --- FUNCIONES DE CÁLCULO ---
def estimate_bodyfat(sex, category):
    mapping = {
        "Male": {"Lean": 12, "Normal": 18, "Stocky": 25, "Obese": 32},
        "Female": {"Lean": 20, "Normal": 26, "Stocky": 33, "Obese": 40}
    }
    return mapping[sex][category]

# Funcion para calcular las macros del usuario segun los datos de entrada
def calculate_macros(sex, age, height, weight, bodyfat_pct, activity, goal):
    lean_mass = weight * (1 - bodyfat_pct / 100)

    # Indice de Masa Corporal (IMC)
    if sex == "Male":
        bmr = 10 * weight + 6.25 * height - 5 * age + 5
    else:
        bmr = 10 * weight + 6.25 * height - 5 * age - 161

    factors = {
        "Sedentary": 1.2,
        "Light": 1.375,
        "Moderate": 1.55,
        "High": 1.725,
        "Very High": 1.9
    }

    # Gasto Energetico Total Diario
    tdee = bmr * factors[activity]


    # Recomendacion de calorias, proteinas y tipo de dieta
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

        #### NOTA: Se han cambiado los nombres de las dietas ###

    fats = (calories * 0.25) / 9
    carbs = (calories - (protein * 4 + fats * 9)) / 4

    return {
        "calories": round(calories),
        "protein": round(protein),
        "fat": round(fats),
        "carbs": round(carbs),
        "recommended_diets": diets
    }

# --- INTERFAZ ---

st.title("🥗 SmartEatAI")
st.caption("Intelligent meal recommender based on your macros")

st.header("Profile Setup")

with st.form("user_form", border=True):
    # Fila 1: Datos Básicos (3 columnas para aprovechar el ancho)
    form_col1, form_col2, form_col3 = st.columns(3)

    with form_col1:
        sex = st.selectbox("Sex", ["Male", "Female"])
        height = st.number_input("Height (cm)", 140, 220, 175)

    with form_col2:
        age = st.number_input("Age", 15, 90, 30)
        weight = st.number_input("Weight (kg)", 40, 200, 75)
    with form_col3:
        meals_per_day = st.number_input("Meals/day", 3, 6, 3)
        body_type = st.selectbox("Body Type", ["Lean", "Normal", "Stocky", "Obese"])

    col_act, col_obj = st.columns(2)

    with col_act:
        activity = st.selectbox(
            "Activity Level", 
            ["Sedentary", "Light", "Moderate", "High", "Very High"]
        )
    with col_obj:
        goal = st.selectbox(
            "Main Goal",
            ["Gain Muscle", "Lose Weight", "Maintenance"]
        )

    # Botón centrado y destacado
    submit = st.form_submit_button("Generate Personalized Plan", use_container_width=True, type="primary")

if submit:
    bodyfat_pct = estimate_bodyfat(sex, body_type)
    macros = calculate_macros(sex, age, height, weight, bodyfat_pct, activity, goal)
    st.session_state.macros = macros # Guardamos las macros en sesion
    # Resetear el estado previo de dietas para que se regeneren recetas
    st.session_state.pop("prev_selected_diets", None)

# --- DIET SELECTOR ---
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
        default=[o for o in options if "[Recommended]" in o],
        key="diet_selector"
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

    if "prev_selected_diets" not in st.session_state:
        st.session_state.prev_selected_diets = selected_diets
        regenerate = True
    else:
        regenerate = selected_diets != st.session_state.prev_selected_diets

    if regenerate:
        used_ids = get_used_recipe_ids()
        st.session_state.recipes = recommend_recipes({
            "calories": macros["calories"]/meals_per_day,
            "fat_content": macros["fat"]/meals_per_day,
            "carbohydrate_content": macros["carbs"]/meals_per_day,
            "protein_content": macros["protein"]/meals_per_day
        }, selected_diets, meals_per_day, used_ids=used_ids)

        st.session_state.prev_selected_diets = selected_diets

    st.session_state.selected_diets = selected_diets

# --- DISPLAY ---
if "macros" in st.session_state:
    macros = st.session_state.macros
    total_protein = 0
    total_fat = 0
    total_cal = 0
    total_carb = 0
    if "recipes" in st.session_state:
        recipes_df = st.session_state.recipes
        if not recipes_df.empty:
            total_protein = recipes_df["protein_content"].sum()
            if "fat_content" in recipes_df.columns:
                total_fat = recipes_df["fat_content"].sum()
            elif "FatContent" in recipes_df.columns:
                total_fat = recipes_df["FatContent"].sum()
            total_cal = recipes_df["calories"].sum()
            total_carb = recipes_df["carbohydrate_content"].sum()

    st.write("**Macro progress for recommended meals:**")
    def macro_bar(label, value, total, color):
        pct = min(1.0, value / total) if total > 0 else 0
        bar_html = f'''<div style="margin-bottom:8px"><b>{label}:</b> {value:.0f} / {total:.0f} <div style='background:#eee;width:100%;height:18px;border-radius:8px;overflow:hidden'><div style='width:{pct*100:.1f}%;height:100%;background:{color};'></div></div></div>'''
        st.markdown(bar_html, unsafe_allow_html=True)

    col1, col2 = st.columns(2)
    with col1:
        macro_bar("Calories", total_cal, macros["calories"], "#f39c12")  # naranja
        macro_bar("Fat", total_fat, macros["fat"], "#27ae60")  # verde
    with col2:
        macro_bar("Protein", total_protein, macros["protein"], "#e74c3c")  # rojo
        macro_bar("Carbohydrates", total_carb, macros["carbs"], "#2980b9")  # azul

    if "recommended_diets" in macros and macros["recommended_diets"]:
        st.write("**Suggested diet types:**")
        render_diet_tags(macros["recommended_diets"])

if "recipes" in st.session_state:
    df_rec = st.session_state.recipes

    if df_rec.empty:
        st.info("⚠️ No recipes found matching your diet preferences. Try adjusting your selections.")
    else:
        st.subheader("🍽️ Recommended Meals")

        # Mostrar recetas
        for idx, row in df_rec.iterrows():
            # Creamos un contenedor único para cada receta
            with st.container(border=True):
                st.subheader(f"Meal {idx+1}: {row['name']}")

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
                    # Mostrar tipos de comida (Breakfast, Lunch, Dinner, Snack) si existen
                    meal_types = row.get("meal_type", [])
                    if isinstance(meal_types, str):
                        meal_types = json.loads(meal_types)

                    tags_html = ""
                    for mt in meal_types:
                        color = MEAL_COLORS.get(mt, "#34495e")
                        tags_html += (
                            f"<span style='background:{color};color:white;"
                            f"padding:4px 10px;border-radius:12px;"
                            f"margin-right:6px;font-size:13px'>{mt}</span>"
                        )
                    st.markdown(tags_html, unsafe_allow_html=True)

                    # Mostrar tipos de dieta si existen
                    diet_types = safe_to_list(row.get("diet_type"))
                    if diet_types:
                        render_diet_tags(diet_types)

                    st.write(f"**🔥 Calories:** {row['calories']} kcal")
                    st.write(f"**🥩 Protein:** {row['protein_content']}g | **🥑 Fat:** {row['fat_content']}g | **🍞 Carbs:** {row['carbohydrate_content']}g")

                    # Botón de intercambio con lógica segura
                    if st.button(f"🔄 Swap for similar", key=f"btn_swp_{row['id']}_{idx}"):
                        # Obtener IDs de las recetas actualmente mostradas (excepto la actual)
                        current_ids = set(st.session_state.recipes["id"].tolist())
                        current_ids.discard(row['id'])  # No excluir la receta actual del candidato

                        new_recipe = swap_for_similar(row['id'], exclude_ids=current_ids)

                        if new_recipe is not None:
                            # 1. Copiamos el DataFrame actual
                            df_temp = st.session_state.recipes.copy()

                            # 2. Alineamos las columnas de la nueva receta con las del DataFrame
                            # Esto asegura que si existe la columna 'dist', no rompa el código
                            for col in df_temp.columns:
                                if col not in new_recipe:
                                    new_recipe[col] = 0

                            # 3. REEMPLAZO SEGURO: Usamos .values para evitar conflictos de índices
                            # Seleccionamos solo las columnas que ya existen en el DataFrame de la sesión
                            df_temp.iloc[idx] = new_recipe[df_temp.columns].values

                            # 4. Actualizamos y refrescamos
                            st.session_state.recipes = df_temp
                            st.rerun()
                        else:
                            st.warning("No similar recipes found. Try swapping a different meal.")