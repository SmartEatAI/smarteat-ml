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

    # Pre-escalamos el dataset completo para evitar procesarlo en cada recomendación
    FEATURES = ['calories', 'fat_content', 'carbohydrate_content', 'protein_content']
    X_scaled_all = scaler.transform(df[FEATURES])

    return df, scaler, knn, X_scaled_all

df_recipes, scaler, knn, X_scaled_all = load_resources()
#st.dataframe(df_recipes)

FEATURES = ['calories', 'fat_content', 'carbohydrate_content', 'protein_content']
MACRO_WEIGHTS = np.array([1.5, 0.8, 1.0, 1.2])  # Cal, Fat, Carb, Prot

DIET_LABELS = {
    "high_protein": "High Protein",
    "low_carb": "Low Carb",
    "vegan": "Vegan",
    "vegetarian": "Vegetarian",
    "low_calorie": "Low Calorie",
    "high_fiber": "High Fiber",
    "high_carb": "High Carb"
}

MEAL_COLORS = {
    "Breakfast": "#f39c12",
    "Lunch": "#2980b9",
    "Dinner": "#8e44ad",
    "Snack": "#16a085"
}

LABEL_COLORS = ["#8e44ad", "#16a085", "#c0392b", "#2980b9", "#f39c12", "#27ae60", "#d35400"]

# --- UTILIDADES ---

def get_meal_order(n_meals):
    # Obtiene el orden de comidas según la cantidad diaria seleccionada
    mapping = {
        3: ["Breakfast", "Lunch", "Dinner"],
        4: ["Breakfast", "Lunch", "Snack", "Dinner"],
        5: ["Breakfast", "Snack", "Lunch", "Snack", "Dinner"],
        6: ["Breakfast", "Snack", "Lunch", "Snack", "Dinner", "Snack"]
    }
    return mapping[n_meals]

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
    Convierte varios formatos a una lista de strings:
    - Lista real -> lista
    - String JSON -> lista
    - String separado por comas -> lista
    - None / NaN -> []
    """
    if value is None:
        return []

    if isinstance(value, list):
        return [str(v).strip() for v in value if str(v).strip()]

    if isinstance(value, str):
        value = value.strip()

        if not value:
            return []

        try:
            parsed = json.loads(value)
            if isinstance(parsed, list):
                return [str(v).strip() for v in parsed if str(v).strip()]
        except Exception:
            pass

        return [v.strip() for v in value.split(",") if v.strip()]

    return []

def normalize_label(s):
    # Normaliza etiquetas para comparación consistente
    if s is None:
        return ""
    return (
        str(s)
        .lower()
        .replace(" ", "_")
        .replace("-", "_")
        .strip()
    )

def get_used_recipe_ids(exclude_id=None):
    # Obtiene los IDs de recetas ya mostradas en el plan
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
    # Recomienda recetas basadas en macros y preferencias dietéticas
    if used_ids is None:
        used_ids = set()

    meal_order = get_meal_order(n)
    final_recipes = []
    current_used_ids = used_ids.copy()

    user_diet_set = set(normalize_label(d) for d in diets)

    for meal_label in meal_order:
        user_vec = np.array([[ 
            macros_obj["calories"],
            macros_obj["fat_content"],
            macros_obj["carbohydrate_content"],
            macros_obj["protein_content"]
        ]])
        user_scaled = scaler.transform(user_vec) * MACRO_WEIGHTS

        # FILTRO INTELIGENTE
        # Filtro de dieta: la receta DEBE tener las dietas seleccionadas
        def check_diet(recipe_diets):
            r_diets = set(normalize_label(d) for d in safe_to_list(recipe_diets))
            return user_diet_set.issubset(r_diets)

        # Filtro de tipo de comida: debe incluir el tipo actual
        def check_meal(recipe_meals, target):
            r_meals = [m.lower().strip() for m in safe_to_list(recipe_meals)]
            return target.lower() in r_meals

        mask_diet = df_recipes["diet_type"].apply(check_diet)
        mask_meal = df_recipes["meal_type"].apply(lambda x: check_meal(x, meal_label))

        mask_combined = mask_diet & mask_meal
        valid_indices = np.where(mask_combined)[0]

        df_search = df_recipes.iloc[valid_indices].copy()
        df_search = df_search[~df_search["id"].isin(current_used_ids)]

        if not df_search.empty:
            # Re-calculamos distancias solo para los válidos
            X_search = X_scaled_all[df_search.index] * MACRO_WEIGHTS
            distances = np.linalg.norm(X_search - user_scaled, axis=1)
            df_search["dist"] = distances

            best_recipe = df_search.sort_values("dist").iloc[0].to_dict()
            best_recipe['assigned_meal_type'] = meal_label 
            final_recipes.append(best_recipe)
            current_used_ids.add(best_recipe["id"])
        else:
            # Si no hay match, intentamos buscar sin el filtro estricto de dieta para no dejar el plan vacío
            st.warning(f"No exact match for {meal_label} with all diet filters. Relaxing constraints...")

    return pd.DataFrame(final_recipes)

def swap_for_similar(
        recipe_id,
        meal_label, 
        recommended_diets, 
        selected_extra=None, 
        n_search=50,
        exclude_ids=None
):
    # Busca recetas similares por macros para reemplazar la actual
    if exclude_ids is None:
        exclude_ids = set()
    if selected_extra is None:
        selected_extra = []

    # Restricciones
    required_diets = set(normalize_label(d) for d in recommended_diets)
    extra_diets = set(normalize_label(d) for d in selected_extra)

    idx_list = df_recipes.index[df_recipes["id"] == recipe_id].tolist()
    if not idx_list:
        return None

    recipe_vec = X_scaled_all[idx_list[0]].reshape(1, -1)
    _, indices = knn.kneighbors(recipe_vec, n_neighbors=n_search)

    valid_candidates = []

    for idx in indices[0][1:]:
        candidate = df_recipes.iloc[idx]
        rid = candidate["id"]

        if rid == recipe_id or rid in exclude_ids:
            continue

        # Validar que sea del mismo tipo de comida
        candidate_meals = [m.lower().strip() for m in safe_to_list(candidate["meal_type"])]
        if meal_label.lower() not in candidate_meals:
            continue

        candidate_diets = set(
            normalize_label(d) for d in safe_to_list(candidate["diet_type"])
        )

        # Debe cumplir con todas las dietas recomendadas
        if not required_diets.issubset(candidate_diets):
            continue

        # Preferencia por dietas adicionales
        if extra_diets:
            if candidate_diets & extra_diets:
                valid_candidates.append(candidate)
        else:
            valid_candidates.append(candidate)

    if not valid_candidates:
        return None

    chosen = valid_candidates[np.random.randint(len(valid_candidates))]
    res = chosen.to_dict()
    res["assigned_meal_type"] = meal_label
    return res


# --- FUNCIONES DE CÁLCULO ---
def estimate_bodyfat(sex, category):
    # Estima el porcentaje de grasa corporal según tipo corporal
    mapping = {
        "Male": {"Lean": 12, "Normal": 18, "Stocky": 25, "Obese": 32},
        "Female": {"Lean": 20, "Normal": 26, "Stocky": 33, "Obese": 40}
    }
    return mapping[sex][category]

def calculate_macros(sex, age, height, weight, bodyfat_pct, activity, goal):
    # Calcula los macros diarios según perfil del usuario
    lean_mass = weight * (1 - bodyfat_pct / 100)

    # Metabolismo basal (BMR) según sexo
    if sex == "Male":
        bmr = 10 * weight + 6.25 * height - 5 * age + 5
    else:
        bmr = 10 * weight + 6.25 * height - 5 * age - 161

    # Factores de actividad física
    factors = {
        "Sedentary": 1.2,
        "Light": 1.375,
        "Moderate": 1.55,
        "High": 1.725,
        "Very High": 1.9
    }

    # Gasto energético diario total (TDEE)
    tdee = bmr * factors[activity]

    # Recomendaciones personalizadas según objetivo
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

    submit = st.form_submit_button("Generate Personalized Plan", use_container_width=True, type="primary")

if submit:
    bodyfat_pct = estimate_bodyfat(sex, body_type)
    macros = calculate_macros(sex, age, height, weight, bodyfat_pct, activity, goal)
    st.session_state.macros = macros  # Guarda los macros en el estado de la sesión
    # Reinicia las recetas previas
    st.session_state.pop("prev_selected_diets", None)

# --- SELECTOR DE DIETA ---
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
        # Convierte etiquetas de UI a claves de dieta
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

# --- VISUALIZACIÓN ---
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
        macro_bar("Calories", total_cal, macros["calories"], "#f39c12")  # orange
        macro_bar("Fat", total_fat, macros["fat"], "#27ae60")  # green
    with col2:
        macro_bar("Protein", total_protein, macros["protein"], "#e74c3c")  # red
        macro_bar("Carbohydrates", total_carb, macros["carbs"], "#2980b9")  # blue

    if "recommended_diets" in macros and macros["recommended_diets"]:
        st.write("**Suggested diet types:**")
        render_diet_tags(macros["recommended_diets"])

if "recipes" in st.session_state:
    df_rec = st.session_state.recipes

    if df_rec.empty:
        st.info("⚠️ No recipes found matching your diet preferences. Try adjusting your selections.")
    else:
        st.subheader("🍽️ Recommended Meals")

        for idx, row in df_rec.iterrows():
            meal_title = row.get('assigned_meal_type', f"Meal {idx+1}")

            with st.container(border=True):
                st.subheader(f"🍴 {meal_title}: {row['name']}")

                c1, c2 = st.columns([1, 2])

                with c1:
                    imgs = row['images'].split(", ")
                    slides = [{"image": url, "title": "", "description": ""} for url in imgs[:3]]

                    uui_carousel(
                        items=slides,
                        variant="sm",
                        key=f"carousel_{row['id']}_{idx}"
                    )

                with c2:
                    # Mostrar tipos de comida (Desayuno, Almuerzo, Cena, Merienda)
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

                    # Mostrar los tipos de dieta
                    diet_types = safe_to_list(row.get("diet_type"))
                    if diet_types:
                        render_diet_tags(diet_types)

                    st.write(f"**🔥 Calories:** {row['calories']} kcal")
                    st.write(f"**🥩 Protein:** {row['protein_content']}g | **🥑 Fat:** {row['fat_content']}g | **🍞 Carbs:** {row['carbohydrate_content']}g")
                    st.write(f"**🛒 Ingredients:** {row["recipe_ingredient_parts"]}")

                    # Lógica de swap de receta
                    if st.button(f"🔄 Swap for similar", key=f"btn_swp_{row['id']}_{idx}"):
                        # Obetener los ids actuales para excluirlos
                        current_ids = set(st.session_state.recipes["id"].tolist())
                        current_ids.discard(row['id'])

                        new_recipe_dict = swap_for_similar(
                            recipe_id=row['id'],
                            meal_label=meal_title, # Aquí pasamos si es Breakfast, Lunch, etc.
                            recommended_diets=macros["recommended_diets"]
                        )

                        if new_recipe_dict:
                            # Actualizamos el DataFrame en la sesión de forma segura
                            for col in st.session_state.recipes.columns:
                                if col in new_recipe_dict:
                                    st.session_state.recipes.at[idx, col] = new_recipe_dict[col]
                            st.rerun()
                        else:
                            st.error("Could not find a similar recipe for this meal type.")