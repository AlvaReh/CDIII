# Clase `Estimacion`

La clase `Estimacion` está diseñada para la estimación de densidades a partir de un conjunto de datos.

## 1. Inicialización

- Recibe un conjunto de datos `datos` que serán usados para estimar densidades.

## 2. Estimación mediante histograma

- `generar_histograma(h)`: Genera un histograma con ancho de bin `h`.
  - Calcula los intervalos (`bins`) para dividir los datos.
  - Cuenta la frecuencia absoluta de datos en cada bin.
  - Devuelve los bins y la estimación de densidad basada en el histograma.
- `evaluar_histograma(x, h)`: Evalúa la densidad del histograma en un conjunto de puntos `x`.

## 3. Estimación mediante kernel

Define varios kernels para estimar la densidad con suavizado:

- `kernel_gauss(u)`: Kernel gaussiano.
- `kernel_uni(u)`: Kernel uniforme.
- `kernel_epanechnikov(u)`: Kernel Epanechnikov.
- `kernel_tri(u)`: Kernel triangular.

## 4. Estimación de densidad con kernel

- `densidad_nucleo(x, h, kernel)`: Estima la densidad en puntos `x` usando un kernel específico y ancho de banda `h`.
  - Soporta kernels: `'uniforme'`, `'gaussiano'`, `'epanechnikov'` y `'triangular'`.
  - Calcula para cada punto la suma ponderada de los kernels centrados en los datos.

---

## Conclusión

La clase ofrece métodos para realizar estimación de densidad usando tanto histogramas como métodos basados en kernels, con diferentes opciones de kernels para adaptar el suavizado según la aplicación.

# Clase `AnalisisDescriptivo`

La clase `AnalisisDescriptivo` amplía la funcionalidad de `Estimacion` para realizar **análisis estadístico descriptivo** sobre un conjunto de datos. Incorpora estadísticas básicas, visualización y análisis gráfico para explorar la distribución de la muestra.

## 1. Inicialización

- Al instanciarse, convierte los datos a un arreglo de NumPy.
- Calcula y guarda automáticamente un **resumen numérico** con medidas clave como media, mediana, desviación estándar, varianza, cuartiles, mínimo y máximo.

## 2. Medidas de tendencia central y dispersión

Incluye métodos específicos para calcular estadísticas básicas:

- `calcular_media()`: Media aritmética.
- `calcular_mediana()`: Mediana.
- `calcular_desviacion_estandar()`: Desviación estándar muestral.
- `calcular_varianza()`: Varianza muestral.
- `calcular_cuartiles()`: Percentiles 25, 50 y 75.

> **Nota:** Dentro del método `resumen_numerico()`, hay un posible error porque se llaman funciones como `self.media()` en lugar de `self.calcular_media()`.

## 3. Resumen numérico

- `resumen_numerico()`: Devuelve un diccionario con las principales estadísticas descriptivas de los datos.

## 4. Análisis gráfico

- `mi_qqplot()`: Genera un **QQ plot** para comparar los cuantiles muestrales con los cuantiles teóricos de una normal estándar. Esto permite evaluar si los datos siguen una distribución normal.
- `graficar_histograma(h)`: Visualiza la **estimación de densidad por histograma** con ancho de bin `h`.
- `graficar_densidad_nucleo(h, kernel)`: Visualiza la **estimación de densidad por el método de núcleos**, permitiendo especificar el tipo de kernel y el ancho de banda.

---

## Conclusión

La clase `AnalisisDescriptivo` integra herramientas de estadística descriptiva numérica y gráfica. Está pensada para ofrecer una primera exploración estructurada de un conjunto de datos, combinando medidas resumen con representaciones visuales que ayudan a comprender la forma de la distribución. Además, hereda la capacidad de estimación de densidades desde la clase `Estimacion`, ampliando su aplicabilidad.


# Clase `GeneradoraDeDatos`

La clase `GeneradoraDeDatos` está diseñada para generar datos aleatorios de distintas distribuciones estadísticas y calcular sus funciones de densidad de probabilidad (PDF).

## 1. Inicialización

- Recibe un parámetro `n` que indica la cantidad de datos a generar.

## 2. Generación de datos

Ofrece métodos para generar vectores de datos de tamaño `n` según varias distribuciones:

- `generar_datos_norm(mu, sigma)`: Datos de una distribución normal con media `mu` y desviación estándar `sigma`.
- `generar_datos_uniformes(a, b)`: Datos de una distribución uniforme en el intervalo `[a, b]`.
- `generar_datos_t(df, loc=0, scale=1)`: Datos de una distribución t de Student con `df` grados de libertad, ubicación `loc` y escala `scale`.
- `generar_datos_BS()`: Datos de una distribución combinada no estándar: 50% normales estándar y 50% mezcla de normales con medias entre -1 y 1 y desviaciones pequeñas (0.1), generados según un esquema basado en valores uniformes.

## 3. Cálculo de densidades de probabilidad (PDF)

Métodos para calcular la densidad teórica de las distribuciones respectivas:

- `calcular_pdf_norm(x, mu, sigma)`: PDF de la normal.
- `calcular_pdf_uniform(x, a, b)`: PDF de la uniforme.
- `calcular_pdf_t(x, df, loc=0, scale=1)`: PDF de la t de Student.
- `calcular_BS(x)`: PDF de la distribución combinada definida en `generar_datos_BS`, que es una mezcla ponderada de normales estándar y cinco normales con medias espaciadas y desviación 0.1.

---

## Conclusión

La clase permite tanto simular datos bajo diferentes supuestos distribucionales como obtener las funciones de densidad teórica asociadas. Esto es útil para análisis estadísticos, simulaciones o validación de métodos de estimación de densidad.


# Análisis y Resumen de la Clase `Regresion` y Clases Derivadas

## Descripción General

La clase `Regresion` es una base para modelar regresiones estadísticas, incluyendo regresión lineal y logística. Define la estructura general para ajustar modelos, predecir valores, y almacenar parámetros y resultados.

---

## Clase `Regresion`

- **Inicialización:**  
  Se especifica el tipo de modelo (`lineal` o `logistica`).  
  Variables internas para datos `x`, `y`, el modelo ajustado y los coeficientes `betas`.

- **Método `ajustar_modelo(x, y)`:**  
  Ajusta el modelo con los datos proporcionados.  
  - Usa `statsmodels` para regresión lineal (OLS) o logística (Logit).  
  - Almacena parámetros ajustados y errores estándar.

- **Método `predecir_valores(nuevos_x)`:**  
  Realiza predicciones para nuevos datos, requiere que el modelo ya haya sido ajustado.

---

## Clase `RegresionLineal` (hereda de `Regresion`)

- Inicializa con tipo modelo `"lineal"`.
- Métodos específicos para regresión lineal:
  - `graficar_dispersion_y_ajuste()`: Muestra gráficos de dispersión y la línea ajustada.
  - `calcular_correlacion()`: Calcula la correlación entre `X` y `Y`.
  - `analicis_supuestos()`: Análisis gráfico de residuos para verificar supuestos (normalidad, homocedasticidad).
  - `estadisticas()`: Imprime coeficientes, errores estándar, valores t y p-valores.
  - `intervalos_confianza_y_prediccion()`: Calcula intervalos para nuevas predicciones.
  - `calcular_r2()`: Imprime R² y R² ajustado.

---

## Clases Derivadas de `RegresionLineal`

- **`RegresionLinealSimple`:**  
  Ajusta modelo lineal simple con `x` y `y`.
  
- **`RegresionLinealMultiple`:**  
  Ajusta modelo lineal múltiple con matrices de variables predictoras.

---

## Clase `RegresionLogistica` (hereda de `Regresion`)

- Inicializa con tipo `"logistica"`.
- Métodos para:
  - Predicción binaria con un umbral.
  - Evaluación del modelo: genera tabla de confusión y calcula error de clasificación.
  - Cálculo de sensibilidad y especificidad para distintos umbrales.
  - Generación de curva ROC y cálculo del área bajo la curva (AUC).

---

## Conclusión

Esta implementación modular permite trabajar con modelos de regresión lineal y logística, facilitando:

- Ajuste de modelos.
- Visualización y diagnóstico de supuestos (en regresión lineal).
- Evaluación de desempeño en regresión logística con métricas estándar.
- Predicción e inferencia con intervalos de confianza.

Es una base sólida para análisis estadístico y machine learning clásico en Python, usando la biblioteca `statsmodels` para cálculos y ajustes.
