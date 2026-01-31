import streamlit as st
import pandas as pd
import numpy as np
import json
from joblib import load
from streamlit_carousel_uui import uui_carousel

# --- CONFIGURATION AND LOADING ---
st.set_page_config(page_title="SmartEatAI")

@st.cache_resource
def load_resources():
    # Load files
    df = load("files/df_recetas.joblib")
    scaler = load("files/scaler.joblib")
    knn = load("files/knn.joblib")

    # Pre-scale the entire dataset to avoid processing it in each recommendation, saving CPU
    FEATURES = ['calories', 'fat_content', 'carbohydrate_content', 'protein_content']
    X_scaled_all = scaler.transform(df[FEATURES])

    return df, scaler, knn, X_scaled_all

df_recipes, scaler, knn, X_scaled_all = load_resources()
st.dataframe(df_recipes)

FEATURES = ['calories', 'fat_content', 'carbohydrate_content', 'protein_content']
MACRO_WEIGHTS = np.array([1.5, 0.8, 1.0, 1.2])  # Cal, Fat, Carb, Prot

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

# --- UTILITIES ---

#### Añadido para feature/meal_type ####
def get_meal_order(n_meals):
    if n_meals == 3:
        return ["Breakfast", "Lunch", "Dinner"]
    elif n_meals == 4:
        return ["Breakfast", "Lunch", "Snack", "Dinner"]
    elif n_meals == 5:
        return ["Breakfast", "Snack", "Lunch", "Snack", "Dinner"]
    else:  # 6 comidas
        return ["Breakfast", "Snack", "Lunch", "Snack", "Dinner", "Snack"]

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
    Converts various formats to a list of strings:
    - Real list -> list
    - JSON string -> list
    - Comma-separated string -> list
    - None / NaN -> []
    """
    if value is None:
        return []

    # If it's already a list
    if isinstance(value, list):
        return [str(v).strip() for v in value if str(v).strip()]

    # If it's a string
    if isinstance(value, str):
        value = value.strip()

        if not value:
            return []

        # Try JSON
        try:
            parsed = json.loads(value)
            if isinstance(parsed, list):
                return [str(v).strip() for v in parsed if str(v).strip()]
        except Exception:
            pass

        # Fallback: comma-separated
        return [v.strip() for v in value.split(",") if v.strip()]

    return []

def normalize_label(s):
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

#### Modificado para feature/meal_type ####
def recommend_recipes(macros_obj, diets, n=3, used_ids=None):
    if used_ids is None:
        used_ids = set()

    # Obtener el orden de comidas
    meal_order = get_meal_order(n)
    
    final_recipes = []
    current_used_ids = used_ids.copy()

    # Iteramos para cada momento del día (Desayuno, Almuerzo, etc.)
    for meal_label in meal_order:
        # Vector de usuario para esta comida (macros divididos)
        user_vec = np.array([[ 
            macros_obj["calories"],
            macros_obj["fat_content"],
            macros_obj["carbohydrate_content"],
            macros_obj["protein_content"]
        ]])
        user_scaled = scaler.transform(user_vec) * MACRO_WEIGHTS

        # --- FILTRADO POR DIETA Y TIPO DE COMIDA ---
        mask_diet = pd.Series([True] * len(df_recipes))
        if diets:
            normalized_diets = set(normalize_label(d) for d in diets)
            mask_diet = df_recipes["diet_type"].apply(
                lambda x: set(normalize_label(dt) for dt in safe_to_list(x)) == normalized_diets
            )

        # FILTRO CRUCIAL: Que el meal_type contenga la etiqueta (ej: 'Breakfast')
        mask_meal = df_recipes["meal_type"].apply(
            lambda x: meal_label.lower() in [m.lower().strip() for m in safe_to_list(x)]
        )

        mask_combined = mask_diet & mask_meal
        valid_indices = np.where(mask_combined)[0]
        
        df_search = df_recipes.iloc[valid_indices].copy()
        X_search = (X_scaled_all[valid_indices] * MACRO_WEIGHTS)

        # Excluir IDs ya usados en este plan
        if current_used_ids:
            mask_used = ~df_search["id"].isin(current_used_ids)
            df_search = df_search[mask_used]
            X_search = X_search[mask_used.values]

        if not df_search.empty:
            distances = np.linalg.norm(X_search - user_scaled, axis=1)
            df_search["dist"] = distances
            best_recipe = df_search.sort_values("dist").iloc[0]
            
            # Guardamos la etiqueta real asignada para mostrarla en la UI
            best_recipe_dict = best_recipe.to_dict()
            best_recipe_dict['assigned_meal_type'] = meal_label 
            final_recipes.append(best_recipe_dict)
            current_used_ids.add(best_recipe["id"])

    return pd.DataFrame(final_recipes)


#### Modificado para feature/meal_type ####
def swap_for_similar(
    recipe_id,
    recommended_diets,
    meal_label,
    selected_extra=None,
    n_search=50,
    exclude_ids=None
):
    if exclude_ids is None:
        exclude_ids = set()
    if selected_extra is None:
        selected_extra = []

    recommended_set = set(normalize_label(d) for d in recommended_diets)
    extra_set = set(normalize_label(d) for d in selected_extra)

    idx_list = df_recipes.index[df_recipes["id"] == recipe_id].tolist()
    if not idx_list:
        return None

    recipe_vec = X_scaled_all[idx_list[0]].reshape(1, -1)
    _, indices = knn.kneighbors(recipe_vec, n_neighbors=n_search)

    valid_candidates = []

    for idx in indices[0][1:]:
        row = df_recipes.iloc[idx]
        # Validar tipo de comida
        candidate_meals = [m.lower().strip() for m in safe_to_list(row["meal_type"])]
        if meal_label.lower() not in candidate_meals:
            continue

        candidate_diets = set(normalize_label(d) for d in safe_to_list(row["diet_type"]))

        # Hard constraint: always include recommended diets
        if not recommended_set.issubset(candidate_diets):
            continue

        # Soft constraint: include some of the extra if any
        if extra_set:
            if candidate_diets & extra_set:
                valid_candidates.append(row)
        else:
            valid_candidates.append(row)

    if valid_candidates:
        return valid_candidates[np.random.randint(len(valid_candidates))].copy()

    return None


# --- CALCULATION FUNCTIONS ---
def estimate_bodyfat(sex, category):
    mapping = {
        "Male": {"Lean": 12, "Normal": 18, "Stocky": 25, "Obese": 32},
        "Female": {"Lean": 20, "Normal": 26, "Stocky": 33, "Obese": 40}
    }
    return mapping[sex][category]

# Function to calculate user macros based on input data
def calculate_macros(sex, age, height, weight, bodyfat_pct, activity, goal):
    lean_mass = weight * (1 - bodyfat_pct / 100)

    # Body Mass Index (BMI)
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

    # Total Daily Energy Expenditure
    tdee = bmr * factors[activity]

    # Recommendation of calories, proteins, and diet type
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

# --- INTERFACE ---

st.title("🥗 SmartEatAI")
st.caption("Intelligent meal recommender based on your macros")

st.header("Profile Setup")

with st.form("user_form", border=True):
    # Row 1: Basic Data (3 columns to make use of width)
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

    # Centered and highlighted button
    submit = st.form_submit_button("Generate Personalized Plan", use_container_width=True, type="primary")

if submit:
    bodyfat_pct = estimate_bodyfat(sex, body_type)
    macros = calculate_macros(sex, age, height, weight, bodyfat_pct, activity, goal)
    st.session_state.macros = macros  # Save macros in session
    # Reset previous diet state so recipes are regenerated
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

        # Show recipes
        for idx, row in df_rec.iterrows():
            meal_title = row.get('assigned_meal_type', f"Meal {idx+1}")

            # Create a unique container for each recipe
            with st.container(border=True):
                st.subheader(f"🍴 {meal_title}: {row['name']}")

                c1, c2 = st.columns([1, 2])

                with c1:
                    # SOLUTION TO DUPLICATE ID: Add a unique key based on ID and index
                    imgs = row['images'].split(", ")
                    slides = [{"image": url, "title": "", "description": ""} for url in imgs[:3]]

                    uui_carousel(
                        items=slides,
                        variant="sm",
                        key=f"carousel_{row['id']}_{idx}"  # <--- Unique key here
                    )

                with c2:
                    # Show meal types (Breakfast, Lunch, Dinner, Snack) if they exist
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

                    # Show diet types if they exist
                    diet_types = safe_to_list(row.get("diet_type"))
                    if diet_types:
                        render_diet_tags(diet_types)

                    st.write(f"**🔥 Calories:** {row['calories']} kcal")
                    st.write(f"**🥩 Protein:** {row['protein_content']}g | **🥑 Fat:** {row['fat_content']}g | **🍞 Carbs:** {row['carbohydrate_content']}g")
                    st.write(f"**🛒 Ingredients:** {row["recipe_ingredient_parts"]}")

                    # Swap button with safe logic
                    if st.button(f"🔄 Swap for similar", key=f"btn_swp_{row['id']}_{idx}"):
                        # Get IDs of currently shown recipes (except the current one)
                        current_ids = set(st.session_state.recipes["id"].tolist())
                        current_ids.discard(row['id'])  # Do not exclude the current recipe from candidate

                        new_recipe = swap_for_similar(
                            row['id'],
                            recommended_diets=macros["recommended_diets"],
                            meal_label=meal_title, # Pasamos el tipo de comida
                            selected_extra=[d for d in st.session_state.selected_diets if d not in macros["recommended_diets"]],
                            exclude_ids=current_ids
                        )


                        if new_recipe is not None:
                            # 1. Copy the current DataFrame
                            df_temp = st.session_state.recipes.copy()

                            # 2. Align columns of the new recipe with the DataFrame
                            # This ensures that if the 'dist' column exists, it doesn't break the code
                            for col in df_temp.columns:
                                if col not in new_recipe:
                                    new_recipe[col] = 0

                            # 3. SAFE REPLACEMENT: Use .values to avoid index conflicts
                            # Select only columns that already exist in the session DataFrame
                            df_temp.iloc[idx] = new_recipe[df_temp.columns].values

                            # 4. Update and refresh
                            st.session_state.recipes = df_temp
                            st.rerun()
                        else:
                            st.warning("No similar recipes found. Try swapping a different meal.")