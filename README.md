# 🥗 SmartEatAI - Recomendador Inteligente de Comidas

## Descripción

SmartEatAI es una aplicación web interactiva que proporciona recomendaciones personalizadas de recetas basadas en tu perfil nutricional, objetivos fitness y preferencias de número de comidas diarias.

Utiliza algoritmos avanzados de machine learning (KNN) para recomendar recetas que coincidan con tus macronutrientes estimados según ecuaciones científicas de nutrición.

Link a la [app desplegada en Streamlit](https://smarteat-ml-rec.streamlit.app/).

Link al [video explicativo](https://youtu.be/QQojlJ8lJp0).

---

## Características Principales

### Cálculo Personalizado de Macronutrientes
- **Ecuación Mifflin-St Jeor**: La más precisa para calcular metabolismo basal (BMR)
- **TDEE (Total Daily Energy Expenditure)**: Gasto energético total basado en nivel de actividad
- **Ajustes por objetivo**:
  - Ganancia muscular: Supervávit calórico + alto en proteína
  - Pérdida de peso: Déficit calórico + mantenimiento de proteína
  - Mantenimiento: Equilibrio calórico

### Selección Flexible de Comidas (3-6)
- **3 comidas**: Desayuno, Almuerzo, Cena
- **4-6 comidas**: Lo anterior + Snacks
- Distribución equitativa de macronutrientes entre comidas

### Algoritmo KNN Inteligente
- Recomendaciones basadas en similitud nutricional
- Recetas aleatorias y variadas (no repetitivas)
- Evita duplicados cuando cambias una receta

### Interfaz Moderna y Responsiva
- Diseño limpio y profesional
- Barras de progreso con indicadores de estado
- Cards de las recetas con carrusel de imágenes

### Dashboard Completo
- Métricas en tiempo real de macros
- Comparación visual con objetivos
- Información detallada de cada receta

---

## Instalación

### Requisitos
- Python 3.8+
- pip o conda
- Docker, Docker Compose (opcional)

### Pasos de Instalación

1. **Clonar el repositorio**
   ```bash
   git clone https://github.com/SmartEatAI/smarteat-ml.git
   cd smarteat-ml
   ```
- **Opción 1: Local**

   2. **Crear entorno virtual (recomendado)**
      ```bash
      python -m venv venv
      source venv/bin/activate  # En Windows: venv\Scripts\activate
      ```

   3. **Instalar dependencias**
      ```bash
      pip install -r requirements.txt
      ```

   4. **Ejecutar la aplicación**
      ```bash
      streamlit run streamlit_app.py
      ```

   La aplicación se abrirá en `http://localhost:8501`

- **Opción 2: Docker**

   2. **Construir la imagen Docker**
      Abre una terminal en la raíz del proyecto y ejecuta el siguiente comando para construir la imagen:
      ```bash
      docker compose build
      ```

   3. **Ejecutar el contenedor**
      Una vez construida la imagen, ejecuta el siguiente comando para iniciar la aplicación:
      ```bash
      docker compose up
      ```

   4. **Abrir la aplicación**
      Accede a la aplicación en tu navegador en la dirección:
      ```
      http://localhost:8501
      ```

   La aplicación se abrirá en `http://localhost:8501`


---

## Uso

### 1. Completar Tu Perfil
En el formulario inicial, ingresa:
- **Datos personales**: Sexo, Edad
- **Medidas**: Altura (cm), Peso (kg)
- **Estilo de vida**: Tipo de cuerpo, Nivel de actividad
- **Objetivo**: Mantenimiento, Ganar músculo, Perder peso
- **Número de comidas**: 3-6 comidas diarias

### 2. Generar Recomendación
Haz clic en "Generar Plan Personalizado"

### 3. Revisar Plan Nutricional
La aplicación mostrará:
- Macronutrientes calculados (Calorías, Proteína, Grasas, Carbohidratos)
- Barras de progreso comparativas
- Tipos de dieta sugeridos
- Información nutricional de las recetas sugeridas

### 4. Ver Comidas Recomendadas
Para cada comida:
- Nombre de la receta
- Carrusel de imágenes
- Macros detallados
- Lista de ingredientes
- Botón para cambiar por receta similar

### 5. Personalizar (Opcional)
Usa "Cambiar por receta similar" para:
- Obtener alternativas de la misma comida
- Evitar recetas que no te atraigan
- Aumentar variedad en tus opciones

---

## Interpretación de Resultados

### Barras de Progreso
- **Verde**: Los macros de las recetas están dentro del 90-110% del objetivo (✅ Óptimo)
- **Rojo**: Desviación significativa (>10% de diferencia)

### Métricas
Los deltas muestran la comparación:
- Positivo: Superávit respecto al objetivo
- Negativo: Déficit respecto al objetivo
- El valor está balanceado si está cercano a 0

---

## Fundamentos Científicos

### Ecuación Mifflin-St Jeor
```
Para Hombres:   BMR = 10 * peso(kg) + 6.25 * altura(cm) - 5 * edad + 5
Para Mujeres:   BMR = 10 * peso(kg) + 6.25 * altura(cm) - 5 * edad - 161
```

### TDEE (Total Daily Energy Expenditure)
```
TDEE = BMR × Factor Actividad

Factores:
- Sedentario: 1.2
- Ligero: 1.375
- Moderado: 1.55
- Alto: 1.725
- Muy alto: 1.9
```

### Distribución de Macros
```
Proteína: Según objetivo (2.0-2.2 g/kg de masa magra)
Grasas: 25% del total calórico
Carbohidratos: Calorías restantes / 4 kcal por gramo
```

---

## Estructura del Proyecto

```
smarteat-ml/
├── streamlit_app.py              # Aplicación principal
├── requirements.txt              # Dependencias Python
├── Dockerfile                    # Configuración Docker
├── docker-compose.yml            # Orquestación Docker
├── files/
│   ├── df_recetas.joblib        # DataFrame serializado
│   ├── knn.joblib               # Modelo KNN entrenado
│   └── scaler.joblib            # Escalador StandardScaler
└──
```

---

## Solución de Problemas

### "ModuleNotFoundError: No module named 'streamlit'"
```bash
pip install streamlit
pip install -r requirements.txt
```

### "Archivo de modelos no encontrado"
Asegúrate de que existen los archivos:
- `files/df_recetas.joblib`
- `files/knn.joblib`
- `files/scaler.joblib`

### "Las recetas no coinciden con los macros"
- Esto es normal debido a la naturaleza del algoritmo KNN
- Las recetas están en el rango de similitud más cercano
- Tolerancia aceptable: ±10% en macros principales

### Cambiar receta no funciona
- Puede significar que no hay más recetas similares disponibles
- Intenta con un número diferente de comidas
- Regenera la recomendación completa

---

## Licencia

Este proyecto es de uso educativo. Está basado en técnicas de machine learning y nutrición científica.

---

## Agradecimientos

- **Mifflin-St Jeor**: Ecuación para cálculo de BMR
- **Scikit-learn**: Implementación del algoritmo KNN
- **Streamlit**: Framework para la interfaz
- **Pandas/NumPy**: Procesamiento de datos

---

## Autores

- [Elías Robles Ruíz](https://github.com/eliasrrobles)
- [Cristina Vacas López](https://github.com/flashtime-dev)
- [Ruyi Xia Ye](https://github.com/rxy94)

---

⭐️ Si te gusta este proyecto, no dudes en darnos una estrellita!
