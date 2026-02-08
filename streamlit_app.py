import streamlit as st
import pandas as pd
import numpy as np
import json
from joblib import load

# --- CONFIGURACIÓN Y CARGA ---
st.set_page_config(page_title="SmartEatAI", page_icon="🥗")

@st.cache_resource
def load_resources():
    """Carga los recursos del modelo y preprocesa el dataset completo"""
    df = load("files/df_recetas.joblib")
    scaler = load("files/scaler.joblib")
    knn = load("files/knn.joblib")

    FEATURES = [
        'cals_per_serving',
        'fat_per_serving',
        'protein_per_serving',
        'carbs_per_serving',
        'fiber_per_serving',
        'sugar_per_serving',
        'protein_ratio',
        'protein_carb_ratio'
    ]
    
    X_scaled_all = scaler.transform(df[FEATURES])

    return df, scaler, knn, X_scaled_all

df_recipes, scaler, knn, X_scaled_all = load_resources()

FEATURES = [
    'cals_per_serving',
    'fat_per_serving',
    'protein_per_serving',
    'carbs_per_serving',
    'fiber_per_serving',
    'sugar_per_serving',
    'protein_ratio',
    'protein_carb_ratio'
]

MACRO_WEIGHTS = np.array([1.5, 0.8, 1.2, 1.0, 0.5, 0.6, 1.3, 1.1])

DIET_LABELS = {
    "balanced": "Balanced",
    "vegetarian": "Vegetarian",
    "high fiber": "High Fiber",
    "low carb": "Low Carb",
    "vegan": "Vegan",
    "low sodium": "Low Sodium",
    "low fat": "Low Fat",
    "keto": "Keto",
    "paleo": "Paleo",
    "high protein": "High Protein"
}

MEAL_COLORS = {
    "breakfast": "#f39c12",
    "lunch": "#2980b9",
    "dinner": "#8e44ad",
    "snack": "#16a085"
}

LABEL_COLORS = {
    "balanced": "#8e44ad",
    "vegetarian": "#16a085",
    "high fiber": "#c0392b",
    "low carb": "#2980b9",
    "vegan": "#f39c12",
    "low sodium": "#27ae60",
    "low fat": "#d35400",
    "keto": "#9b59b6",
    "paleo": "#e67e22",
    "high protein": "#34495e"
}

# --- UTILIDADES ---

def get_meal_order(n_meals):
    """Obtiene el orden de comidas según la cantidad diaria"""
    mapping = {
        3: ["breakfast", "lunch", "dinner"],
        4: ["breakfast", "lunch", "snack", "dinner"],
        5: ["breakfast", "snack", "lunch", "snack", "dinner"],
        6: ["breakfast", "snack", "lunch", "snack", "dinner", "snack"]
    }
    return mapping[n_meals]

def render_diet_tags(diets):
    """Renderiza etiquetas de dieta con colores específicos"""
    html = ""
    for d in diets:
        label = DIET_LABELS.get(d, d.title())
        color = LABEL_COLORS.get(d, "#34495e")
        html += f"<span style='background:{color};color:white;padding:4px 10px;border-radius:12px;margin-right:6px;font-size:13px'>{label}</span>"
    st.markdown(html, unsafe_allow_html=True)

def safe_to_list(value):
    """Convierte varios formatos a lista de strings de forma segura"""
    if value is None:
        return []
    
    if isinstance(value, list):
        return [str(v).strip() for v in value if v and str(v).strip()]

    if pd.isna(value):
        return []

    if isinstance(value, str):
        value = value.strip()
        if not value or value == '[]':
            return []
        
        try:
            parsed = json.loads(value)
            if isinstance(parsed, list):
                return [str(v).strip() for v in parsed if v and str(v).strip()]
        except Exception:
            pass
        
        return [v.strip() for v in value.split(",") if v.strip()]

    return []

def normalize_label(s):
    """Normaliza etiquetas a minúsculas para comparación"""
    if s is None or pd.isna(s):
        return ""
    return str(s).lower().strip()

def get_used_recipe_ids(exclude_id=None):
    """Obtiene los IDs de recetas ya usadas en el plan"""
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

def calculate_derived_features(cals, protein, fat, carbs, fiber, sugar):
    """Calcula features derivadas para el modelo"""
    protein_ratio = (protein * 4 / (cals + 1)) * 100
    protein_carb_ratio = protein / (carbs + 1)
    
    return {
        'protein_ratio': protein_ratio,
        'protein_carb_ratio': protein_carb_ratio
    }

def recommend_recipes(macros_obj, diets, n=3, used_ids=None):
    """Recomienda recetas basadas en macros y preferencias dietéticas"""
    if used_ids is None:
        used_ids = set()

    meal_order = get_meal_order(n)
    final_recipes = []
    current_used_ids = used_ids.copy()

    user_diet_set = set(normalize_label(d) for d in diets) if diets else set()

    derived = calculate_derived_features(
        macros_obj["cals_per_serving"],
        macros_obj["protein_per_serving"],
        macros_obj["fat_per_serving"],
        macros_obj["carbs_per_serving"],
        macros_obj.get("fiber_per_serving", 8.0),
        macros_obj.get("sugar_per_serving", 10.0)
    )

    user_vec = np.array([[
        macros_obj["cals_per_serving"],
        macros_obj["fat_per_serving"],
        macros_obj["protein_per_serving"],
        macros_obj["carbs_per_serving"],
        macros_obj.get("fiber_per_serving", 8.0),
        macros_obj.get("sugar_per_serving", 10.0),
        derived['protein_ratio'],
        derived['protein_carb_ratio']
    ]])
    
    user_scaled = scaler.transform(user_vec) * MACRO_WEIGHTS

    for meal_label in meal_order:
        def check_diet(recipe_diets):
            if not user_diet_set:
                return True
            if recipe_diets is None:
                return False
            if isinstance(recipe_diets, list) and len(recipe_diets) == 0:
                return False
            if not isinstance(recipe_diets, list):
                if pd.isna(recipe_diets):
                    return False
                recipe_diets = safe_to_list(recipe_diets)
                if not recipe_diets:
                    return False
            
            r_diets = set(normalize_label(d) for d in recipe_diets)
            return user_diet_set.issubset(r_diets)

        def check_meal(recipe_meals):
            if recipe_meals is None:
                return False
            if isinstance(recipe_meals, list) and len(recipe_meals) == 0:
                return False
            if not isinstance(recipe_meals, list):
                if pd.isna(recipe_meals):
                    return False
                recipe_meals = safe_to_list(recipe_meals)
                if not recipe_meals:
                    return False
            
            r_meals = [normalize_label(m) for m in recipe_meals]
            return normalize_label(meal_label) in r_meals

        mask_diet = df_recipes["diet_type"].apply(check_diet)
        mask_meal = df_recipes["meal_type"].apply(check_meal)
        mask_combined = mask_diet & mask_meal
        valid_indices = np.where(mask_combined)[0]

        df_search = df_recipes.iloc[valid_indices].copy()
        df_search = df_search[~df_search["id"].isin(current_used_ids)]

        if not df_search.empty:
            X_search = X_scaled_all[df_search.index] * MACRO_WEIGHTS
            distances = np.linalg.norm(X_search - user_scaled, axis=1)
            df_search["dist"] = distances

            best_recipe = df_search.sort_values("dist").iloc[0].to_dict()
            best_recipe['assigned_meal_type'] = meal_label 
            final_recipes.append(best_recipe)
            current_used_ids.add(best_recipe["id"])
        else:
            mask_meal_only = df_recipes["meal_type"].apply(check_meal)
            df_fallback = df_recipes[mask_meal_only].copy()
            df_fallback = df_fallback[~df_fallback["id"].isin(current_used_ids)]
            
            if not df_fallback.empty:
                X_fallback = X_scaled_all[df_fallback.index] * MACRO_WEIGHTS
                distances = np.linalg.norm(X_fallback - user_scaled, axis=1)
                df_fallback["dist"] = distances
                
                best_recipe = df_fallback.sort_values("dist").iloc[0].to_dict()
                best_recipe['assigned_meal_type'] = meal_label
                final_recipes.append(best_recipe)
                current_used_ids.add(best_recipe["id"])
                
                st.info(f"ℹ️ No exact match for {meal_label.title()} with selected diets. Showing best nutritional match.")

    return pd.DataFrame(final_recipes)

def swap_for_similar(
        recipe_id,
        meal_label, 
        recommended_diets, 
        selected_extra=None, 
        n_search=100,
        exclude_ids=None
):
    """Encuentra receta similar con estrategia de fallback multinivel"""
    if exclude_ids is None:
        exclude_ids = set()
    if selected_extra is None:
        selected_extra = []

    required_diets = set(normalize_label(d) for d in recommended_diets) if recommended_diets else set()
    extra_diets = set(normalize_label(d) for d in selected_extra) if selected_extra else set()

    idx_list = df_recipes.index[df_recipes["id"] == recipe_id].tolist()
    if not idx_list:
        return None

    recipe_vec = X_scaled_all[idx_list[0]].reshape(1, -1)
    _, indices = knn.kneighbors(recipe_vec, n_neighbors=n_search)

    valid_candidates_strict = []
    valid_candidates_medium = []
    valid_candidates_relaxed = []

    for idx in indices[0][1:]:
        candidate = df_recipes.iloc[idx]
        rid = candidate["id"]

        if rid == recipe_id or rid in exclude_ids:
            continue

        candidate_meals = [normalize_label(m) for m in safe_to_list(candidate["meal_type"])]
        if normalize_label(meal_label) not in candidate_meals:
            continue

        candidate_diets = set(normalize_label(d) for d in safe_to_list(candidate["diet_type"]))

        valid_candidates_relaxed.append(candidate)

        if required_diets:
            if not required_diets.issubset(candidate_diets):
                continue
        
        valid_candidates_medium.append(candidate)

        if extra_diets:
            if candidate_diets & extra_diets:
                valid_candidates_strict.append(candidate)
        else:
            valid_candidates_strict.append(candidate)

    if valid_candidates_strict:
        chosen_candidates = valid_candidates_strict
    elif valid_candidates_medium:
        chosen_candidates = valid_candidates_medium
        st.info(f"ℹ️ Relaxed diet filters to find a match for {meal_label.title()}")
    elif valid_candidates_relaxed:
        chosen_candidates = valid_candidates_relaxed
        st.warning(f"⚠️ Using flexible search for {meal_label.title()} (nutritional similarity only)")
    else:
        mask_any_meal = df_recipes["meal_type"].apply(
            lambda meals: normalize_label(meal_label) in [normalize_label(m) for m in safe_to_list(meals)]
        )
        df_any_meal = df_recipes[mask_any_meal]
        df_any_meal = df_any_meal[~df_any_meal["id"].isin(exclude_ids)]
        df_any_meal = df_any_meal[df_any_meal["id"] != recipe_id]
        
        if not df_any_meal.empty:
            chosen = df_any_meal.sample(n=1).iloc[0]
            res = chosen.to_dict()
            res["assigned_meal_type"] = meal_label
            st.warning(f"⚠️ Limited options for {meal_label.title()}. Showing random match.")
            return res
        else:
            return None

    chosen = chosen_candidates[np.random.randint(len(chosen_candidates))]
    res = chosen.to_dict()
    res["assigned_meal_type"] = meal_label
    
    return res

# --- FUNCIONES DE CÁLCULO ---

def estimate_bodyfat(sex, category):
    """Estima porcentaje de grasa corporal"""
    mapping = {
        "Male": {"Lean": 12, "Normal": 18, "Stocky": 25, "Obese": 32},
        "Female": {"Lean": 20, "Normal": 26, "Stocky": 33, "Obese": 40}
    }
    return mapping[sex][category]

def calculate_macros(sex, age, height, weight, bodyfat_pct, activity, goal, meals_per_day):
    """Calcula macronutrientes diarios basados en perfil del usuario"""
    lean_mass = weight * (1 - bodyfat_pct / 100)

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

    tdee = bmr * factors[activity]

    if goal == "Gain Muscle":
        total_cals = tdee * 1.1 + 150
        protein = lean_mass * 2.2
        diets = ["high protein", "high fiber"]
    elif goal == "Lose Weight":
        total_cals = tdee * 0.8
        protein = lean_mass * 2.2
        diets = ["low carb", "low fat"]
    else:
        total_cals = tdee
        protein = lean_mass * 2.0
        diets = ["balanced", "high fiber"]

    fats = (total_cals * 0.25) / 9
    carbs = (total_cals - (protein * 4 + fats * 9)) / 4
    
    fiber_per_meal = 8.0
    sugar_per_meal = 10.0

    return {
        "cals_per_serving": round(total_cals / meals_per_day, 2),
        "protein_per_serving": round(protein / meals_per_day, 2),
        "fat_per_serving": round(fats / meals_per_day, 2),
        "carbs_per_serving": round(carbs / meals_per_day, 2),
        "fiber_per_serving": round(fiber_per_meal, 2),
        "sugar_per_serving": round(sugar_per_meal, 2),
        "total_protein": round(protein),
        "total_fat": round(fats),
        "total_cals": round(total_cals),
        "total_carbs": round(carbs),
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

    submit = st.form_submit_button("🚀 Generate Personalized Plan", use_container_width=True, type="primary")

if submit:
    bodyfat_pct = estimate_bodyfat(sex, body_type)
    macros = calculate_macros(sex, age, height, weight, bodyfat_pct, activity, goal, meals_per_day)
    st.session_state.macros = macros
    st.session_state.meals_per_day = meals_per_day
    st.session_state.pop("prev_selected_diets", None)
    st.rerun()

# --- SELECTOR DE DIETA ---

if "macros" in st.session_state:
    macros = st.session_state.macros
    meals_per_day = st.session_state.get("meals_per_day", 3)
    recommended = macros["recommended_diets"]

    options = []
    default_selections = []
    
    for diet_key in recommended:
        if diet_key in DIET_LABELS:
            option_text = f"{DIET_LABELS[diet_key]} [Recommended]"
            options.append(option_text)
            default_selections.append(option_text)
    
    for k, v in DIET_LABELS.items():
        if k not in recommended:
            options.append(v)

    diet_selector_key = f"diet_selector_{'_'.join(sorted(recommended))}"

    st.header("Diet preferences")
    selected = st.multiselect(
        "Diet preferences",
        options,
        default=default_selections,
        key=diet_selector_key,
        help="Recommended diets are based on your goal"
    )

    def labels_to_keys(selected_labels):
        keys = []
        for s in selected_labels:
            clean = s.replace(" [Recommended]", "").strip()
            for k, v in DIET_LABELS.items():
                if v == clean:
                    keys.append(k)
                    break
        return keys

    selected_diets = labels_to_keys(selected)

    if "prev_selected_diets" not in st.session_state:
        st.session_state.prev_selected_diets = selected_diets
        regenerate = True
    else:
        regenerate = selected_diets != st.session_state.prev_selected_diets

    if regenerate:
        used_ids = get_used_recipe_ids()
        st.session_state.recipes = recommend_recipes(
            macros, 
            selected_diets, 
            meals_per_day, 
            used_ids=used_ids
        )
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
            total_protein = recipes_df["protein_per_serving"].sum()
            total_fat = recipes_df["fat_per_serving"].sum()
            total_cal = recipes_df["cals_per_serving"].sum()
            total_carb = recipes_df["carbs_per_serving"].sum()

    st.header("Macro progress")
    
    def macro_bar(label, value, total, color):
        pct = min(1.0, value / total) if total > 0 else 0
        bar_html = f'''<div style="margin-bottom:8px"><b>{label}:</b> {value:.0f} / {total:.0f} <div style='background:#eee;width:100%;height:18px;border-radius:8px;overflow:hidden'><div style='width:{pct*100:.1f}%;height:100%;background:{color};'></div></div></div>'''
        st.markdown(bar_html, unsafe_allow_html=True)

    col1, col2 = st.columns(2)
    with col1:
        macro_bar("Calories", total_cal, macros["total_cals"], "#f39c12")
        macro_bar("Fat", total_fat, macros["total_fat"], "#27ae60")
    with col2:
        macro_bar("Protein", total_protein, macros["total_protein"], "#e74c3c")
        macro_bar("Carbohydrates", total_carb, macros["total_carbs"], "#2980b9")

    if "recommended_diets" in macros and macros["recommended_diets"]:
        st.write("**Suggested diet types:**")
        render_diet_tags(macros["recommended_diets"])

if "recipes" in st.session_state:
    df_rec = st.session_state.recipes

    if df_rec.empty:
        st.warning("⚠️ No recipes found matching your diet preferences. Try adjusting your selections.")
    else:
        st.subheader("🍽️ Recommended Meals")

        for idx, row in df_rec.iterrows():
            meal_title = row.get('assigned_meal_type', f"Meal {idx+1}").title()

            with st.container(border=True):
                st.subheader(f"🍴 {meal_title}: {row['recipe_name']}")

                c1, c2 = st.columns([1, 2])

                with c1:
                    img_url = row.get('image_url', '')
                    if img_url:
                        st.image(img_url, use_container_width=True)

                with c2:
                    meal_types = safe_to_list(row.get("meal_type", []))
                    if meal_types:
                        tags_html = ""
                        for mt in meal_types:
                            color = MEAL_COLORS.get(normalize_label(mt), "#34495e")
                            display_name = mt.title()
                            tags_html += f"<span style='background:{color};color:white;padding:4px 10px;border-radius:12px;margin-right:6px;font-size:13px'>{display_name}</span>"
                        st.markdown(tags_html, unsafe_allow_html=True)

                    diet_types = safe_to_list(row.get("diet_type", []))
                    if diet_types:
                        render_diet_tags(diet_types)

                    st.write(f"**🔥 Calories:** {row['cals_per_serving']:.0f} kcal")
                    
                    st.write(f"**🥩 Protein:** {row['protein_per_serving']:.1f}g | "
                             f"**🥑 Fat:** {row['fat_per_serving']:.1f}g | "
                             f"**🍞 Carbs:** {row['carbs_per_serving']:.1f}g")
                    
                    st.write(f"**🌾 Fiber:** {row['fiber_per_serving']:.1f}g | "
                             f"**🍯 Sugar:** {row['sugar_per_serving']:.1f}g")
                    
                    ingredients = safe_to_list(row.get("ingredients_clean", []))
                    if ingredients:
                        ingredients_capitalized = [ing.title() for ing in ingredients]
                        st.write(f"**🛒 Ingredients:** {', '.join(ingredients_capitalized)}")

                    if st.button(f"🔄 Swap for similar", key=f"btn_swp_{row['id']}_{idx}"):
                        current_ids = get_used_recipe_ids(exclude_id=row['id'])

                        new_recipe_dict = swap_for_similar(
                            recipe_id=row['id'],
                            meal_label=row.get('assigned_meal_type', meal_title.lower()),
                            recommended_diets=macros["recommended_diets"],
                            selected_extra=st.session_state.get("selected_diets", []),
                            exclude_ids=current_ids
                        )

                        if new_recipe_dict:
                            for col in st.session_state.recipes.columns:
                                if col in new_recipe_dict:
                                    st.session_state.recipes.at[idx, col] = new_recipe_dict[col]
                            st.rerun()
                        else:
                            st.error("❌ Could not find a similar recipe. Try adjusting your diet preferences.")