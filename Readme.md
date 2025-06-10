convierte esto en un archivo md

1. Estimación de Densidades (Estimacion)

- Clase dedicada a estimar funciones de densidad mediante métodos no paramétricos. Incluye:

- Histogramas adaptables con ajuste de ancho de banda (h).

- Cuatro kernels para suavizado: Gaussiano (óptimo para datos normales), Uniforme (simple), Epanechnikov (minimiza error cuadrático) y Triangular (balanceado).

- Validación automática de datos vacíos y parámetros inválidos.

2\. Análisis Descriptivo (AnalisisDescriptivo)

- Herramienta para exploración de datos:

- Cálculo automático de métricas clave: media, mediana, cuartiles, varianza y desviación estándar.

- Visualizaciones integradas:

- QQ-Plots para comparar distribuciones con la normal.

- Histogramas y curvas de densidad superpuestas (usando kernels).

- Hereda métodos de Estimacion para estimar densidades.

3\. Generación de Datos (GeneradoraDeDatos)

- Simulador de distribuciones probabilísticas:

- Distribuciones clásicas: Normal (media y desviación ajustables), Uniforme (en intervalos personalizados), t-Student (grados de libertad).

- Distribución mixta "BS": Combina 50% normal estándar y 50% componentes uniformes.

- PDFs teóricas: Permite comparar datos simulados con sus distribuciones teóricas.

4\. Modelado Predictivo (RegresionLineal y RegresionLogistica)

- Regresión Lineal:

- Ajuste de modelos simples o múltiples.

- Diagnósticos gráficos: residuos vs predichos, QQ-Plots de    normalidad.

- Métricas: R², errores estándar de coeficientes, intervalos de confianza.

- Regresión Logística:

- Clasificación binaria con umbral ajustable.

- Evaluación mediante:

- Matriz de confusión.

- Curva ROC y cálculo del área bajo la curva (AUC).

- Sensibilidad/especificidad por umbral.
