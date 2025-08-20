# Challenge — Flight Delay API

## Resumen
API para predecir retraso de vuelos usando un modelo de ML y exponerlo vía FastAPI. Entrena al iniciar con `data/data.csv` y valida el contrato de entrada antes de predecir.

## Decisiones clave
- **Modelo:** LogisticRegression (`solver=liblinear`, `C≈0.5–0.75`, `class_weight="balanced"`) con **umbral de probabilidad calibrado** en holdout para maximizar **F1 de la clase positiva**.
- **Por qué:** ligera, explicable, reproducible y con buen F1 al calibrar umbral, sin dependencias extra.
- **Features:** one-hot de `OPERA`, `TIPOVUELO`, `MES` con **orden fijo (10 columnas)** para mantener consistencia entre entrenamiento y serving.

## Supuestos
- Si `delay` no existe, se **deriva** con la regla de negocio: `min_diff >= 15`.
- Validaciones estrictas:
  - `MES ∈ [1..12]`
  - `TIPOVUELO ∈ {N,O}` (**se acepta** `"I"` y se **mapea** a `"O"` para compatibilidad)
  - `OPERA` debe ser conocida por el modelo (se mapean sinónimos comunes, p.ej. `"Grupo LATAM"` → `"LATAM Airlines Group"`)

## Cómo correr
```bash
# Crear y activar entorno (Unix/macOS)
python3 -m venv .venv
source .venv/bin/activate

# En Windows:
# py -3.10 -m venv .venv
# .\.venv\Scripts\activate

# Instalar dependencias (stack del enunciado)
pip install -r requirements.txt -c constraints.txt

# Ejecutar API (entrena en startup y expone /health y /predict)
uvicorn challenge.api:app --host 0.0.0.0 --port 8080 --reload
