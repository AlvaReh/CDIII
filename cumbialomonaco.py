"""
Módulo de análisis estadístico y modelado predictivo

Autor: Alvaro Graf Reh
Fecha: 10/06/2025

Este módulo contiene clases para:
- Estimación de densidades (histogramas y kernels)
- Análisis descriptivo de datos
- Generación de datos simulados
- Modelos de regresión lineal y logística
"""

from sklearn.metrics import auc
from scipy.stats import expon
from scipy.stats import t
from scipy.stats import chi2, uniform
from numpy.random import randint
import matplotlib.pyplot as plt
import numpy as np
import statsmodels.api as sm
from scipy.stats import norm
import pandas as pd
import random
from sklearn.metrics import auc
from math import pi

class Estimacion:
    """Clase para estimación de densidades."""

    def __init__(self, datos):
        self.datos = datos

    def generar_histograma(self, h):
        """Genera un histograma de los datos con ancho de bin h."""
        bins = np.arange(np.min(self.datos) - h, np.max(self.datos) + h, h)
        fr_abs = np.zeros(len(bins) - 1)
        for ind in range(len(self.datos)):
            for ind_bin in range(len(bins) - 1):
                if self.datos[ind] < bins[ind_bin + 1] and self.datos[ind] >= bins[ind_bin]:
                    fr_abs[ind_bin] += 1
                    break
        estimacion_hist = fr_abs / (len(self.datos) * h)
        return bins, estimacion_hist

    def evaluar_histograma(self, x, h):
        """Evalúa densidad del histograma en puntos x."""
        bins, estimacion_hist = self.genera_histograma(h)
        estimaciones_x = np.zeros(len(x))
        for i in range(len(x)):
            for ind_bin in range(len(bins) - 1):
                if x[i] >= bins[ind_bin] and x[i] < bins[ind_bin + 1]:
                    estimaciones_x[i] = estimacion_hist[ind_bin]
                    break
        return estimaciones_x

    def kernel_gauss(self, u):
        """Calcula el kernel gaussiano."""
        return (1 / np.sqrt(2 * np.pi)) * np.exp(-0.5 * u**2)

    def kernel_uni(self, u):
        """Calcula el kernel uniforme."""
        valores_k = np.zeros_like(u)
        for i in range(len(u)):
            if abs(u[i]) <= 1:
                valores_k[i] = 1 / 2
        if len(valores_k) == 0:
            raise ValueError("El conjunto de valores_k está vacío. No se puede calcular la densidad.")
        return valores_k

    def kernel_epanechnikov(self, u):
        """Calcula el kernel Epanechnikov."""
        valores_k = np.zeros_like(u)
        for i in range(len(u)):
            if abs(u[i]) <= 1:
                valores_k[i] = 3 / 4 * (1 - u[i] ** 2)
        return valores_k

    def kernel_tri(self, u):
        """Calcula el kernel triangular."""
        valores_k = np.zeros_like(u)
        for i in range(len(u)):
            if abs(u[i]) <= 1:
                valores_k[i] = 1 - abs(u[i])
        return valores_k

    def densidad_nucleo(self, x, h, kernel='uniforme'):
        """Estimación de densidad con kernel especificado."""
        densidad = []
        if len(self.datos) == 0:
            raise ValueError("El conjunto de datos está vacío. No se puede calcular la densidad.")
        for xi in x:
            u = (self.datos - xi) / (h + 1e-10)
            if kernel == 'uniforme':
                valores_k = self.kernel_uniforme(u)
            elif kernel == 'gaussiano':
                valores_k = self.kernel_gaussiano(u)
            elif kernel == 'epanechnikov':
                valores_k = self.kernel_epanechnikov(u)
            elif kernel == 'triangular':
                valores_k = self.kernel_triangular(u)
            else:
                raise ValueError(f"Kernel '{kernel}' no reconocido.")
            densidad.append(np.sum(valores_k) / (len(self.datos) * (h + 1e-10)))
        return np.array(densidad)


class AnalisisDescriptivo(Estimacion):
    def __init__(self, datos):
        super().__init__(datos)
        self.datos = np.array(datos)
        self.n = len(self.datos)
        self.resumen_numerico = self.resumen_numerico()

    def calcular_media(self):
        return np.mean(self.datos)

    def calcular_mediana(self):
        return np.median(self.datos)

    def calcular_desviacion_estandar(self):
        return np.std(self.datos, ddof=1)

    def calcular_varianza(self):
        return np.var(self.datos, ddof=1)

    def calcular_cuartiles(self):
        return np.percentile(self.datos, [25, 50, 75])

    def resumen_numerico(self):
        """Genera un resumen numérico de las estadísticas descriptivas."""
        return {
            'La media calculada es': self.media(),
            'La mediana calculada es': self.mediana(),
            'La desviacion estandar calculada es': self.desviacion_estandar(),
            'La varianza calculada es': self.varianza(),
            'Los cuartiles calculados es': self.cuartiles(),
            'El mínimo calculado es': np.min(self.datos),
            'El máximo calculado es': np.max(self.datos)
        }

    def mi_qqplot(self):
        """Genera un QQ plot para comparar los cuantiles muestrales con los cuantiles teóricos de una distribución normal."""
        media = self.media()
        desviacion_estandar = self.desviacion_estandar()
        data_s = (self.datos - media) / desviacion_estandar
        cuantiles_muestrales = np.sort(data_s)
        pp = np.arange(1, (self.n + 1)) / (self.n + 1)
        cuantiles_teoricos = norm.ppf(pp)

        plt.scatter(cuantiles_teoricos, cuantiles_muestrales, color='blue', marker='o')
        plt.xlabel('Cuantiles teóricos')
        plt.ylabel('Cuantiles muestrales')
        plt.plot(cuantiles_teoricos, cuantiles_teoricos, linestyle='-', color='red')
        plt.show()

    def graficar_histograma(self, h):
        """Genera un histograma de los datos con una estimación de densidad."""
        x = np.linspace(np.min(self.datos), np.max(self.datos), 200)  
        y = self.evalua_histograma(x, h)  

        plt.plot(x, y, color='steelblue', linewidth=2, label='Histograma')
        plt.title('Estimación de densidad con histograma')
        plt.xlabel('Datos')
        plt.ylabel('Densidad estimada')
        plt.grid(True)
        plt.legend()
        plt.show()


    def graficar_densidad_nucleo(self, h, kernel='uniforme'):
        """Genera un gráfico de la estimación de densidad usando el método del núcleo."""
        x = np.linspace(np.min(self.datos) - 1, np.max(self.datos) + 1, 200)
        y = self.densidad_nucleo(x, h, kernel)

        plt.plot(x, y, color='purple', label=f'Kernel: {kernel}, h={h}')
        plt.title('Estimación de densidad con método del núcleo')
        plt.xlabel('Datos')
        plt.ylabel('Densidad estimada')
        plt.grid(True)
        plt.legend()
        plt.show()


class GeneradoraDeDatos:  
    def __init__(self, n):
        self.n = n

    def generar_datos_norm(self, mu, sigma):
        """Genera n datos de una distribución normal con media mu y desviación estándar sigma."""
        datos_normales = np.random.normal(mu, sigma, self.n)
        return datos_normales

    def generar_datos_uniformes(self, a, b):
        """Genera n datos de una distribución uniforme en el intervalo [a, b]."""
        datos_uniformes = np.random.uniform(a, b, self.n)
        return datos_uniformes

    def generar_datos_t(self, df, loc=0, scale=1):
        """Genera n datos de una distribución t de Student con df grados de libertad, loc y scale."""
        datos_t = np.random.standard_t(df, size=self.n) * scale + loc
        return datos_t

    def generar_datos_BS(self):
        """Genera n datos de una distribución combinada: 50% normal estándar y 50% uniforme en [-1, 1]."""
        u = np.random.uniform(size=self.n)
        y = np.zeros(self.n)
        for i in range(self.n):
            if u[i] > 0.5:
                y[i] = np.random.normal(0, 1)
                continue
            for j in range(5):
                if (j * 0.1) < u[i] <= ((j + 1) * 0.1):
                    y[i] = np.random.normal(j / 2 - 1, 0.1)
                    break
        return y

    def calcular_pdf_norm(self, x, mu, sigma):
        """Calcula la densidad de probabilidad de una distribución normal."""
        return norm.pdf(x, mu, sigma)

    def calcular_pdf_uniform(self, x, a, b):
        """Calcula la densidad de probabilidad de una distribución uniforme."""
        return uniform.pdf(x, a, b)

    def calcular_pdf_t(self, x, df, loc=0, scale=1):
        """Calcula la densidad de probabilidad de una distribución t de Student."""
        return t.pdf((x - loc) / scale, df) / scale

    def calcular_BS(self, x):
        """Calcula la densidad teórica de la distribución combinada."""
        resultado = 1/2 * norm.pdf(x, 0, 1) + 1/10 * (
            norm.pdf(x, -1, 0.1) + norm.pdf(x, -0.5, 0.1) +
            norm.pdf(x, 0, 0.1) + norm.pdf(x, 0.5, 0.1) +
            norm.pdf(x, 1, 0.1)
        )
        return resultado

class Regresion:
    def __init__(self, tipo_modelo="lineal"):
        self.tipo_modelo = tipo_modelo
        self.x = None
        self.y = None
        self.modelo_ajustado = None
        self.betas = None

    def ajustar_modelo(self, x, y):
        """Ajusta el modelo de regresión según el tipo especificado."""
        self.x = x
        self.y = y
        x_const = sm.add_constant(x)
        if self.tipo_modelo == "lineal":
            modelo = sm.OLS(y, x_const)
        elif self.tipo_modelo == "logistica":
            modelo = sm.Logit(y, x_const)
        else:
            raise ValueError("El tipo_modelo debe ser 'lineal' o 'logistica.")

        self.modelo_ajustado = modelo.fit()
        self.betas = self.modelo_ajustado.params
        self.errores_estandar = self.modelo_ajustado.bse

    def predecir_valores(self, nuevos_x):
        """Realiza predicciones con el modelo ajustado."""
        if self.modelo_ajustado is None:
            raise RuntimeError("Modelo no entrenado, debes ajustar primero.")
            return None
        nuevos_x_const = sm.add_constant(nuevos_x)
        return self.modelo_ajustado.predict(nuevos_x_const)

class RegresionLineal(Regresion):
    def __init__(self):
        """Inicializa el modelo de regresión lineal."""
        """Llama al constructor de la clase base Regresion con tipo_modelo 'lineal'."""
        super().__init__(tipo_modelo="lineal")

    def graficar_dispersion_y_ajuste(self):
        """Genera un gráfico de dispersión de los datos y la recta ajustada."""
        if len(self.x.shape) == 1 or self.x.shape[1] == 1:
            x = self.x if len(self.x.shape) == 1 else self.x[:, 0]
            plt.scatter(x, self.y)
            plt.plot(x, self.modelo_ajustado.fittedvalues, color='red')
            plt.xlabel("X")
            plt.ylabel("Y")
            plt.title("Dispersión y recta ajustada")
            plt.grid(True)
            plt.show()
        else:
          for i in range(self.x.shape[1]):
            plt.scatter(self.x[:, i], self.y)
            plt.xlabel(f"X{i+1}")
            plt.ylabel("Y")
            plt.title(f"Dispersión: X{i+1} vs Y")
            plt.grid(True)
            plt.show()


    def calcular_correlacion(self):
        """Calcula y muestra la correlación entre X e Y."""
        if len(self.x.shape) == 1 or self.x.shape[1] == 1:
            x = self.x if len(self.x.shape) == 1 else self.x[:, 0]
            corr = np.corrcoef(x, self.y)[0, 1]
            print("Correlación entre las variables X e Y:", corr)
        else:
            for i in range(self.x.shape[1]):
                corr = np.corrcoef(self.x[:, i], self.y)[0, 1]
                print(f"Correlación entre las variables X{i+1} e Y:", corr)

    def analicis_supuestos(self):
        """Genera gráficos de residuos y QQ plot."""

        """Calcula los residuos y valores predichos del modelo ajustado."""
        residuos = self.modelo_ajustado.resid
        predichos = self.modelo_ajustado.fittedvalues

        """Gráfico de residuos para verificar la homocedasticidad y normalidad."""
        sm.qqplot(residuos, line='45')
        plt.title("QQ plot de residuos")
        plt.grid(True)
        plt.show()

        plt.scatter(predichos, residuos)
        plt.axhline(0, color='red', linestyle='--')
        plt.xlabel("Valores predichos")
        plt.ylabel("Residuos")
        plt.title("Residuos vs Valores predichos")
        plt.grid(True)
        plt.show()

    def estadisticas(self):
        """Imprime las estadísticas del modelo ajustado."""
        print("Los parametros son:")
        print(self.betas)
        print("\nLos errores estándar son:")
        print(self.errores_estandar)
        print("\nEstadistico t-observado es:")
        print(self.modelo_ajustado.tvalues)
        print("\nLos p-valores calculados son:")
        print(self.modelo_ajustado.pvalues)

    def intervalos_confianza_y_prediccion(self, nuevos_x, alpha=0.05):
        """Calcula intervalos de confianza y predicción para nuevas observaciones."""
        nuevos_x_const = sm.add_constant(nuevos_x)
        pred = self.modelo_ajustado.get_prediction(nuevos_x_const)

        ic_media = pred.conf_int(alpha=alpha, obs=False)
        ic_pred = pred.conf_int(alpha=alpha, obs=True)

        return {
            "Intervalo de confianza de la media inferior": ic_media[:, 0],
            "Intervalo de confianza de la media superior": ic_media[:, 1],
            "Intervalo de confianza de la predicción inferior": ic_pred[:, 0],
            "Intervalo de confianza de la predicción superior": ic_pred[:, 1]
        }

    def calcular_r2(self):
        print("R cuadrado:", self.modelo_ajustado.rsquared)
        print("R cuadrado ajustado:", self.modelo_ajustado.rsquared_adj)


class RegresionLinealSimple(RegresionLineal):
    
    def __init__(self, x, y):
        """Inicializa el modelo de regresión lineal simple."""
        super().__init__()
        self.ajustar_modelo(x, y)

class RegresionLinealMultiple(RegresionLineal):
    def __init__(self, x, y):
        """Inicializa el modelo de regresión lineal múltiple."""
        super().__init__()
        self.ajustar_modelo(x, y)

class RegresionLogistica(Regresion):
    def __init__(self):
        """Inicializa el modelo de regresión logística."""
        super().__init__(tipo_modelo="logistica")
        """Lista para almacenar la sensibilidad."""
        self.sensibilidad = []
        """Lista para almacenar la especificidad."""
        self.especificidad = []

    def prediccion(self, x_nuevo, umbral=0.5):
        """Realiza una predicción binaria basada en un umbral."""
        prob = self.predecir(x_nuevo)
        return (prob >= umbral).astype(int)

    def evaluar_modelo(self, x_test, y_test):
        """Evalúa el modelo de regresión logística y muestra una tabla de confusión."""
        y_pred = self.predigo(x_test)
        FN = np.sum((y_pred == 0) & (y_test == 1))
        FP = np.sum((y_pred == 1) & (y_test == 0))
        TP = np.sum((y_pred == 1) & (y_test == 1))
        TN = np.sum((y_pred == 0) & (y_test == 0))
        error = (FN + FP) / len(y_test)

        """Tabla de confusión con los resultados del modelo."""
        tabla = pd.DataFrame({
            'y_test=1': [TP, FN],
            'y_test=0': [FP, TN]
        }, index=['y_pred=1', 'y_pred=0'])

        print(tabla)
        print(f"Error de mala clasificación: {error:.2f}")

    def sensibilidad_y_especificidad(self, x_test, y_test, umbral):
        """Calcula la sensibilidad y especificidad del modelo para un umbral dado."""
        y_pred = (self.predecir(x_test) >= umbral).astype(int)
        TP = np.sum((y_pred == 1) & (y_test == 1))
        FN = np.sum((y_pred == 0) & (y_test == 1))
        FP = np.sum((y_pred == 1) & (y_test == 0))
        TN = np.sum((y_pred == 0) & (y_test == 0))

        sensibilidad = TP / (TP + FN) if (TP + FN) > 0 else 0
        especificidad = TN / (TN + FP) if (TN + FP) > 0 else 0
        return sensibilidad, especificidad

    def calcular_cortes_roc(self, x_test, y_test):
        """Calcula los puntos de corte para la curva ROC."""
        probs = self.predecir(x_test)
        cortes = np.linspace(0, 1, 100)
        self.sensibilidad = []
        self.especificidad = []

        for umbral in cortes:
            sens, esp = self.sensibilidad_y_especificidad(x_test, y_test, umbral)
            self.sensibilidad.append(sens)
            self.especificidad.append(esp)

        return np.array(self.sensibilidad), np.array(self.especificidad)

    def calcular_curva_roc(self, x_test, y_test):
        """Genera la curva ROC y calcula el AUC."""
        self.puntos_de_corte_roc(x_test, y_test)

        plt.plot(1 - np.array(self.especificidad), self.sensibilidad)
        plt.xlabel("1 - Especificidad")
        plt.ylabel("Sensibilidad")
        plt.title("Curva ROC")
        plt.grid(True)
        plt.show()

    def calcular_AUC(self, x_test, y_test):
        """Calcula el área bajo la curva ROC (AUC)."""
        especificidad, sensibilidad = self.puntos_de_corte_roc(x_test, y_test)
        return auc(1 - np.array(especificidad), np.array(sensibilidad))