import numpy as np
import pandas as pd
import streamlit as st
from joblib import load
from streamlit_carousel_uui import uui_carousel
import ast

# --- GLOBAL CONFIGURATION ---
st.set_page_config(page_title="SmartEatAI", layout="wide")

FEATURES = ["calories", "fat_content", "carbohydrate_content", "protein_content"]
MACRO_WEIGHTS = np.array([1.5, 0.8, 1.0, 1.2])  # Cal, Fat, Carb, Prot

# --- RESOURCE LOADING ---
@st.cache_resource(show_spinner="Loading models and data...")
def load_resources():
    df = load("files/df_recetas.joblib")
    scaler = load("files/scaler.joblib")
    knn = load("files/knn.joblib")

    missing_cols = [c for c in FEATURES if c not in df.columns]
    if missing_cols:
        raise ValueError(f"Missing columns in df_recetas: {missing_cols}")

    X_scaled_all = scaler.transform(df[FEATURES])
    X_weighted_all = X_scaled_all * MACRO_WEIGHTS

    return df, scaler, knn, X_scaled_all, X_weighted_all

df_recipes, scaler, knn, X_scaled_all, X_weighted_all = load_resources()

# --- DIET NORMALIZATION ---
def normalize_diet(diet: str) -> str:
    """Normalize a single diet string to atomic label."""
    return diet.strip().lower().replace(" ", "_")

def extract_atomic_diets(df: pd.DataFrame) -> set[str]:
    """Extract unique atomic diet labels from the dataset."""
    diets = set()
    for entry in df["diet_type"].dropna():
        for d in entry.split(","):
            diets.add(normalize_diet(d))
    return diets

# Precompute atomic diets
ATOMIC_DIETS = extract_atomic_diets(df_recipes)

# Display-friendly diets (capitalize and replace underscores)
DISPLAY_DIETS = sorted([d.replace("_", " ").title() for d in ATOMIC_DIETS])

# Mapping from display to atomic
DIET_MAPPING = {d.replace("_", " ").title(): d for d in ATOMIC_DIETS}

# --- CORE LOGIC FUNCTIONS ---
def normalize_diet_list(diets: list[str] | None) -> set[str]:
    """Normalize a list of diets to a set of atomic labels."""
    if not diets:
        return set()
    return {normalize_diet(d) for d in diets}

def recipe_matches_diets(row_diets: str, wanted_diets: set[str]) -> bool:
    """Check if recipe diets (as set) contain ANY of the wanted diets (OR logic)."""
    if not wanted_diets:
        return True

    # Parse recipe diets to set
    recipe_set = {normalize_diet(d) for d in str(row_diets or "").split(",")}

    # OR logic: recipe must have at least ONE wanted diet
    return bool(wanted_diets & recipe_set)

def recommend_recipes(macros_obj: dict, user_diets: list[str] | None = None, n: int = 3) -> pd.DataFrame:
    """Recommend recipes using KNN after filtering by diets (OR logic)."""
    user_diets_norm = normalize_diet_list(user_diets)

    # Debug
    st.session_state.debug_diets = user_diets_norm

    # User vector
    user_vec = np.array([[
        macros_obj["calories"],
        macros_obj["fat_content"],
        macros_obj["carbohydrate_content"],
        macros_obj["protein_content"],
    ]])
    user_scaled_weighted = scaler.transform(user_vec) * MACRO_WEIGHTS

    # Filter by diets (OR logic) BEFORE KNN
    if user_diets_norm:
        mask = df_recipes["diet_type"].fillna("").apply(
            lambda d: recipe_matches_diets(d, user_diets_norm)
        )
        df_search = df_recipes.loc[mask].copy()
        X_search = X_weighted_all[mask.values]
        st.session_state.debug_mask_count = mask.sum()
    else:
        df_search = df_recipes.copy()
        X_search = X_weighted_all

    if df_search.empty:
        st.warning("⚠️ No recipes found matching ANY selected diets!")
        return df_search.head(0)

    # KNN distances
    distances = np.linalg.norm(X_search - user_scaled_weighted, axis=1)
    df_search["dist"] = distances

    return df_search.sort_values("dist", ascending=True).head(n).reset_index(drop=True)

def swap_for_similar(recipe_id: int, n_search: int = 11):
    idx_list = df_recipes.index[df_recipes["id"] == recipe_id].tolist()
    if not idx_list:
        return None

    global_idx = idx_list[0]
    recipe_vec = X_scaled_all[global_idx].reshape(1, -1)

    dist, indices = knn.kneighbors(recipe_vec, n_neighbors=n_search)
    if indices.size <= 1:
        return None

    neighbor_idx = int(indices[0][np.random.randint(1, min(n_search, indices.shape[1]))])
    return df_recipes.iloc[neighbor_idx].copy()

def estimate_body_fat(sex: str, category: str) -> int:
    mapping = {
        "Male": {"Lean": 12, "Normal": 18, "Stocky": 25, "Obese": 32},
        "Female": {"Lean": 20, "Normal": 26, "Stocky": 33, "Obese": 40},
    }
    return mapping[sex][category]

def calculate_macros(sex, age, height, weight, body_fat, activity, goal):
    lean_mass = weight * (1 - body_fat / 100)

    if sex == "Male":
        bmr = 10 * weight + 6.25 * height - 5 * age + 5
    else:
        bmr = 10 * weight + 6.25 * height - 5 * age - 161

    activity_factors = {
        "Sedentary": 1.2,
        "Light": 1.375,
        "Moderate": 1.55,
        "High": 1.725,
        "Very High": 1.9,
    }

    tdee = bmr * activity_factors[activity]

    if goal == "Gain Muscle":
        calories = tdee * 1.1 + 150
        protein = lean_mass * 2.2
        diets = ["high_protein", "high_fiber"]

    elif goal == "Lose Weight":
        calories = tdee * 0.8
        protein = lean_mass * 2.2
        diets = ["low_calorie", "high_fiber"]

    else:
        calories = tdee
        protein = lean_mass * 2.0
        diets = ["balanced", "vegetarian", "vegan"]

    fats = (calories * 0.25) / 9
    carbs = (calories - (protein * 4 + fats * 9)) / 4

    return {
        "calories": round(calories),
        "protein": round(protein),
        "fat": round(fats),
        "carbs": round(carbs),
        "diets": diets,
    }

# --- USER INTERFACE ---
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
        weight = st.number_input("Weight (kg)", 40.0, 200.0, 75.0)

    with form_col3:
        meals_per_day = st.number_input("Meals/day", 3, 6, 3)
        body_type = st.selectbox("Body Type", ["Lean", "Normal", "Stocky", "Obese"])

    col_act, col_obj = st.columns(2)

    with col_act:
        activity = st.selectbox(
            "Activity Level",
            ["Sedentary", "Light", "Moderate", "High", "Very High"],
        )
    with col_obj:
        goal = st.selectbox(
            "Main Goal",
            ["Gain Muscle", "Lose Weight", "Maintenance"],
        )

    submit = st.form_submit_button(
        "Generate Personalized Plan", use_container_width=True, type="primary"
    )

if submit:
    body_fat = estimate_body_fat(sex, body_type)
    macros = calculate_macros(sex, age, height, weight, body_fat, activity, goal)
    st.session_state.macros = macros

    st.session_state.recipes = recommend_recipes(
        {
            "calories": macros["calories"] / meals_per_day,
            "fat_content": macros["fat"] / meals_per_day,
            "carbohydrate_content": macros["carbs"] / meals_per_day,
            "protein_content": macros["protein"] / meals_per_day,
            "diets": macros["diets"],
        },
        user_diets=st.session_state.get("selected_diets", []),
        n=meals_per_day,
    )

# --- MACROS DISPLAY ---
if "macros" in st.session_state:
    st.subheader("📊 Daily Macros")
    macros = st.session_state.macros

    total_protein = 0.0
    total_fat = 0.0
    total_cal = 0.0
    total_carb = 0.0

    if "recipes" in st.session_state and not st.session_state.recipes.empty:
        recipes_df = st.session_state.recipes
        total_protein = recipes_df["protein_content"].sum()
        total_fat = recipes_df["fat_content"].sum()
        total_cal = recipes_df["calories"].sum()
        total_carb = recipes_df["carbohydrate_content"].sum()

    st.write("**Macro progress for recommended meals:**")

    @st.cache_data(ttl=300)
    def macro_bar(label: str, value: float, total: float, color: str) -> None:
        pct = min(1.0, value / total) if total > 0 else 0
        bar_html = f"""
        <div style="margin-bottom:8px">
            <b>{label}:</b> {value:.0f} / {total:.0f}
            <div style="background:#eee;width:100%;height:18px;border-radius:8px;overflow:hidden">
                <div style="width:{pct*100:.1f}%;height:100%;background:{color};"></div>
            </div>
        </div>
        """
        st.markdown(bar_html, unsafe_allow_html=True)

    col1, col2 = st.columns(2)
    with col1:
        macro_bar("Calories", total_cal, macros["calories"], "#f39c12")
        macro_bar("Fat", total_fat, macros["fat"], "#27ae60")
    with col2:
        macro_bar("Protein", total_protein, macros["protein"], "#e74c3c")
        macro_bar("Carbohydrates", total_carb, macros["carbs"], "#2980b9")

    # --- DIET SELECTION ---
    st.write("**Select Diet Preferences:**")

    # Map goal to recommended diets
    goal_to_diets = {
        "Gain Muscle": ["High Protein", "High Fiber"],
        "Lose Weight": ["Low Calorie", "High Fiber"],
        "Maintenance": ["Balanced", "Vegetarian", "Vegan"]
    }

    recommended_display = [
        d for d in goal_to_diets.get(goal, [])
        if d in DISPLAY_DIETS
    ]

    selected_display_diets = st.multiselect(
        "Select diet preferences:",
        options=DISPLAY_DIETS,
        default=recommended_display,
    )

    # Convert to atomic diets
    selected_atomic_diets = [DIET_MAPPING[d] for d in selected_display_diets]

    # Update if changed
    if st.session_state.get("selected_diets") != selected_atomic_diets:
        st.session_state.selected_diets = selected_atomic_diets

        # Recalculate with selected diets
        st.session_state.recipes = recommend_recipes(
            {
                "calories": macros["calories"] / meals_per_day,
                "fat_content": macros["fat"] / meals_per_day,
                "carbohydrate_content": macros["carbs"] / meals_per_day,
                "protein_content": macros["protein"] / meals_per_day,
                "diets": selected_atomic_diets,
            },
            user_diets=selected_atomic_diets,
            n=meals_per_day,
        )
        st.rerun()

# --- DEBUG INFO ---
if "debug_diets" in st.session_state:
    with st.expander("🔍 Debug Info"):
        st.write(f"**Active diets:** {st.session_state.debug_diets}")
        if "debug_mask_count" in st.session_state:
            st.write(f"**Matching recipes:** {st.session_state.debug_mask_count}")

# --- RECIPES DISPLAY ---
if "recipes" in st.session_state and not st.session_state.recipes.empty:
    df_rec = st.session_state.recipes
    st.subheader("🍽️ Recommended Meals")

    for idx, row in df_rec.iterrows():
        with st.container(border=True):
            st.subheader(f"Meal {idx + 1}: {row['name']}")

            c1, c2 = st.columns([1, 2])

            with c1:
                imgs = str(row.get("images", "")).split(", ")
                slides = [{"image": url, "title": "", "description": ""} for url in imgs[:3] if url]
                if slides:
                    uui_carousel(
                        items=slides,
                        variant="sm",
                        key=f"carousel_{row['id']}_{idx}",
                    )

            with c2:
                st.write(f"**🔥 Calories:** {row['calories']} kcal")
                st.write(
                    f"**🥩 Protein:** {row['protein_content']}g | "
                    f"**🥑 Fat:** {row['fat_content']}g | "
                    f"**🍞 Carbs:** {row['carbohydrate_content']}g"
                )

                # Clean meal_type
                meal_type = row['meal_type']
                if isinstance(meal_type, list):
                    meal_type = ', '.join(meal_type)
                elif isinstance(meal_type, str) and meal_type.startswith('['):
                    try:
                        meal_list = ast.literal_eval(meal_type)
                        meal_type = ', '.join(meal_list)
                    except:
                        pass
                st.write(f"**🍽️ Meal Type:** {meal_type}")

                # Clean diet_type
                diet_type = row['diet_type']
                if isinstance(diet_type, list):
                    diet_type = ', '.join(diet_type)
                elif isinstance(diet_type, str) and diet_type.startswith('['):
                    try:
                        diet_list = ast.literal_eval(diet_type)
                        diet_type = ', '.join(diet_list)
                    except:
                        pass
                st.write(f"**🥗 Diet Type:** {diet_type}")

                ingredients = row.get("recipe_ingredient_parts", "")
                if isinstance(ingredients, list) and ingredients:
                    st.markdown("**🥘 Ingredients:**", unsafe_allow_html=True)
                    for ingredient in ingredients:
                        st.markdown(f"- {ingredient.capitalize()}")
                elif isinstance(ingredients, str) and ingredients.strip():
                    st.markdown(f"**🥘 Ingredients:**\n{ingredients.strip()}", unsafe_allow_html=True)
                else:
                    st.caption("*Ingredients not available*")

                if st.button("🔄 Swap for similar", key=f"btn_swp_{row['id']}_{idx}"):
                    new_recipe = swap_for_similar(int(row["id"]))

                    if new_recipe is not None:
                        df_temp = st.session_state.recipes.copy()

                        for col in df_temp.columns:
                            if col not in new_recipe:
                                new_recipe[col] = 0

                        df_temp.iloc[idx] = new_recipe[df_temp.columns].values
                        st.session_state.recipes = df_temp
                        st.rerun()

# --- VIEW DATASET ---
st.header("📊 Recommended Recipes Dataset")
if "recipes" in st.session_state and not st.session_state.recipes.empty:
    with st.expander("🔍 View recommended recipes data"):
        # Preprocess to handle mixed types in recipe_ingredient_parts
        df_to_show = st.session_state.recipes.copy()
        df_to_show['recipe_ingredient_parts'] = df_to_show['recipe_ingredient_parts'].astype(str)
        st.dataframe(df_to_show)
else:
    st.info("No recommendations available yet. Please select preferences and get recommendations.")
