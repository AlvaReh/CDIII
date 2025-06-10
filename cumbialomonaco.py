# -*- coding: utf-8 -*-

"""
Autor: Alvaro Graf Reh
Fecha: 10/06/2025
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import statsmodels.api as sm
from scipy import stats
from statsmodels.stats.anova import anova_lm
from scipy.stats import shapiro, norm, expon, t, uniform
from statsmodels.stats.outliers_influence import variance_inflation_factor
from sklearn.metrics import confusion_matrix, roc_curve, auc, classification_report

class CumbiaLomonaco:
    """
    Librería integral para análisis estadístico que incluye:
    - Generación de datos
    - Análisis descriptivo
    - Estimación de densidades
    - Regresión lineal múltiple
    - Regresión logística
    """

    class GeneradoraDatos:
        def __init__(self, datos):
            self.datos = np.array(datos)

        def generar_datos_normal(self, media, desvio, N=1000):
            max_val = self.maximo()
            min_val = self.minimo()
            grilla = np.linspace(min_val, max_val, N)
            self.datos_normal = norm.pdf(grilla, media, desvio)
            return grilla, self.datos_normal

        def generar_datos_BS(self, N=1000):
            u = np.random.uniform(size=(N,))
            y = u.copy()
            ind = np.where(u > 0.5)[0]
            y[ind] = np.random.normal(0, 1, size=len(ind))
            for j in range(5):
                ind = np.where((u > j * 0.1) & (u <= (j+1) * 0.1))[0]
                y[ind] = np.random.normal(j/2 - 1, 1/10, size=len(ind))
            return y

        def maximo(self):
            return np.max(self.datos)

        def minimo(self):
            return np.min(self.datos)

    class AnalisisDescriptivo:
        def __init__(self, datos):
            self.datos = np.array(datos)

        def calculo_de_media(self):
            return np.mean(self.datos)

        def calculo_de_mediana(self):
            return np.median(self.datos)

        def calculo_varianza(self):
            return np.var(self.datos)

        def calculo_desvio_estandar(self):
            return np.std(self.datos)

        def calcular_cuartiles(self):
            return np.percentile(self.datos, [25, 50, 75])

        def correlacion_de_pearson(self, x, y):
            return np.corrcoef(x, y)

        def calcular_histograma(self, h):
            puntos = np.arange(self.minimo(), self.maximo()+h, h)
            histograma = np.zeros(len(puntos)-1)
            for i in range(len(puntos)-1):
                for j in range(len(self.datos)):
                    if self.datos[j] >= puntos[i] and self.datos[j] < puntos[i+1]:
                        histograma[i] += 1
                histograma[i] = histograma[i]/len(self.datos)
                histograma[i] = histograma[i]/h
            return histograma, puntos

        def evalua_histograma(self, h, x):
            histograma, puntos = self.calcular_histograma(h)
            estim_hist = np.zeros(len(x))
            for i in range(len(x)):
                for j in range(len(puntos)-1):
                    if x[i] >= puntos[j] and x[i] < puntos[j+1]:
                        estim_hist[i] = histograma[j]
            return estim_hist

        def kernel_uniforme(self, x):
            return 1 if (x >= -1/2) and (x <= 1/2) else 0

        def kernel_gaussiano(self, x):
            return 1/(np.sqrt(2*np.pi)*np.exp(-x**2/2))

        def kernel_cuadratico(self, x):
            return (3/4)*(1-x**2) if (x >= -1/2) and (x <= 1/2) else 0

        def kernel_triangular(self, x):
            return (1+x)*(x >= 0 and x <= 1)+(1-x)*(x < 0 and x > -1)

        def mi_densidad(self, x, h, kernel):
            density = np.zeros_like(x, dtype=float)
            for i in range(len(x)):
                for j in range(len(self.datos)):
                    if kernel == "uniforme":
                        density[i] += self.kernel_uniforme((self.datos[j]-x[i])/h)
                    elif kernel == "gaussiano":
                        density[i] += self.kernel_gaussiano((self.datos[j]-x[i])/h)
                    elif kernel == "cuadratico":
                        density[i] += self.kernel_cuadratico((self.datos[j]-x[i])/h)
                    elif kernel == "triangular":
                        density[i] += self.kernel_triangular((self.datos[j]-x[i])/h)
            return density/(len(self.datos)*h)

        def qqplot(self, data=None):
            if data is None:
                data = self.datos
            x_ord = np.sort(data)
            n = len(x_ord)
            cuantiles_muestrales = [(i + 1 - 0.5)/n for i in range(n)]
            cuantiles_teoricos = norm.ppf(cuantiles_muestrales)
            plt.scatter(cuantiles_teoricos, x_ord, color='blue', marker='o')
            plt.xlabel('Cuantiles teóricos')
            plt.ylabel('Cuantiles muestrales')
            plt.plot(cuantiles_teoricos, cuantiles_teoricos, linestyle='-', color='red')
            plt.show()

        def maximo(self):
            return np.max(self.datos)

        def minimo(self):
            return np.min(self.datos)

    class Estimacion:
        def __init__(self, datos):
            self.datos = np.array(datos)

        def kernel_gaussiano(self, u):
            return (1/np.sqrt(2*np.pi))*np.exp(-0.5*u**2)

        def kernel_uniforme(self, u):
            return 1 if -0.5 <= u <= 0.5 else 0

        def kernel_cuadratico(self, u):
            return (3/4)*(1-u**2) if -1 <= u <= 1 else 0

        def densidad_nucleo(self, h, kernel, x):
            n = len(self.datos)
            density = np.zeros_like(x)
            for j, val in enumerate(x):
                total = 0
                for dato in self.datos:
                    u = (val-dato)/h
                    if kernel == 'uniforme':
                        total += self.kernel_uniforme(u)
                    elif kernel == 'gaussiano':
                        total += self.kernel_gaussiano(u)
                    elif kernel == 'cuadratico':
                        total += self.kernel_cuadratico(u)
                density[j] = total/(n*h)
            return density

        def evalua_histograma(self, h, x):
            bins = np.arange(min(self.datos)-h/2, max(self.datos)+h, h)
            histograma = np.zeros(len(bins)-1)
            for dato in self.datos:
                for j in range(len(bins)-1):
                    if bins[j] <= dato < bins[j+1]:
                        histograma[j] += 1
                        break
            freq_rel = histograma/len(self.datos)
            densidad = freq_rel/h
            estim = np.zeros(len(x))
            for idx, val in enumerate(x):
                for j in range(len(bins)-1):
                    if bins[j] <= val < bins[j+1]:
                        estim[idx] = densidad[j]
                        break
            return estim

        def miqqplot(self, datos=None):
            if datos is not None:
                self.datos = np.array(datos)
            data = self.datos
            data_ordenada = np.sort(data)
            media = np.mean(data)
            desv = np.std(data)
            data_ord_s = [(i-media)/desv for i in data_ordenada]
            cuantiles_teoricos = [norm.ppf((i+1)/(len(data)+1)) for i in range(len(data))]
            plt.scatter(cuantiles_teoricos, data_ord_s)
            plt.plot(cuantiles_teoricos, cuantiles_teoricos, color='red')
            plt.xlabel('Cuantiles teóricos')
            plt.ylabel('Cuantiles muestrales')
            plt.show()

    class RegresionLinealMultiple:
        def __init__(self, dataframe, variable_respuesta, variables_predictoras, categoria_base=None):
            self.df = dataframe.copy()
            self.y_name = variable_respuesta
            self.x_names = variables_predictoras.copy() if variables_predictoras else []
            self.categoria_base = categoria_base
            self.modelo = None
            self.resultados = None
            self._preparar_datos()

        def _preparar_datos(self):
            for var in [col for col in self.x_names if col in self.df.select_dtypes(['object', 'category']).columns]:
                if self.categoria_base is not None:
                    categories = [c for c in self.df[var].unique() if c != self.categoria_base]
                    categories = [self.categoria_base] + categories
                else:
                    categories = self.df[var].unique()
                self.df[var] = pd.Categorical(self.df[var], categories=categories)
                dummies = pd.get_dummies(self.df[var], prefix=var, drop_first=True)
                self.df = pd.concat([self.df.drop(var, axis=1), dummies], axis=1)
                self.x_names.remove(var)
                self.x_names.extend(dummies.columns.tolist())
            self.y = self.df[self.y_name].astype(float)
            self.X = self.df[self.x_names].astype(float)
            self.X = sm.add_constant(self.X)

        def agregar_interaccion(self, var1, var2, prefijo=None):
            if var1 not in self.df.columns or var2 not in self.df.columns:
                raise ValueError("Ambas variables deben existir en el DataFrame")
            nombre_interaccion = f"{prefijo}_" if prefijo else ""
            nombre_interaccion += f"{var1}_x_{var2}"
            if nombre_interaccion not in self.df.columns:
                self.df[nombre_interaccion] = self.df[var1]*self.df[var2]
                self.x_names.append(nombre_interaccion)
                self._preparar_datos()

        def ajustar_modelo(self):
            self.modelo = sm.OLS(self.y, self.X)
            self.resultados = self.modelo.fit()
            return self.resultados

        def resumen_modelo(self):
            if self.resultados is None:
                self.ajustar_modelo()
            return self.resultados.summary()

        def graficos_diagnostico(self):
            if self.resultados is None:
                self.ajustar_modelo()
            residuos = self.resultados.resid
            predichos = self.resultados.fittedvalues
            plt.figure(figsize=(15, 10))
            plt.subplot(2, 2, 1)
            plt.scatter(predichos, residuos)
            plt.axhline(y=0, color='r', linestyle='--')
            plt.xlabel('Valores predichos')
            plt.ylabel('Residuos')
            plt.title('Residuos vs. Valores predichos')
            plt.subplot(2, 2, 2)
            self._qqplot(residuos)
            plt.title('QQ Plot de residuos')
            plt.subplot(2, 2, 3)
            plt.hist(residuos, bins=20, density=True, alpha=0.6)
            x = np.linspace(min(residuos), max(residuos), 100)
            plt.plot(x, norm.pdf(x, np.mean(residuos), np.std(residuos)), 'r-')
            plt.title('Distribución de residuos')
            plt.subplot(2, 2, 4)
            residuos_estandarizados = np.sqrt(np.abs(residuos-residuos.mean())/residuos.std())
            plt.scatter(predichos, residuos_estandarizados)
            plt.xlabel('Valores predichos')
            plt.ylabel('Raíz cuadrada de residuos estandarizados')
            plt.title('Gráfico de escala-localización')
            plt.tight_layout()
            plt.show()

        def _qqplot(self, data):
            media = np.mean(data)
            desviacion = np.std(data)
            data_s = (data-media)/desviacion
            cuantiles_muestrales = np.sort(data_s)
            n = len(data)
            pp = np.arange(1, (n+1))/(n+1)
            cuantiles_teoricos = norm.ppf(pp)
            plt.scatter(cuantiles_teoricos, cuantiles_muestrales, color='blue', marker='o')
            plt.xlabel('Cuantiles teóricos')
            plt.ylabel('Cuantiles muestrales')
            plt.plot(cuantiles_teoricos, cuantiles_teoricos, linestyle='-', color='red')

        def test_normalidad(self):
            if self.resultados is None:
                self.ajustar_modelo()
            stat, p_valor = shapiro(self.resultados.resid)
            print(f"Test de Shapiro-Wilk para normalidad:")
            print(f"Estadístico = {stat:.4f}, p-valor = {p_valor:.4f}")
            if p_valor < 0.05:
                print("Conclusión: Rechazamos la hipótesis de normalidad (α=0.05)")
            else:
                print("Conclusión: No hay evidencia para rechazar la normalidad (α=0.05)")
            return p_valor

        def test_hipotesis(self, alpha=0.05):
            if self.resultados is None:
                self.ajustar_modelo()
            print(f"Tests de hipótesis para coeficientes (α={alpha}):")
            print("H0: βi = 0 vs H1: βi ≠ 0\n")
            for i, var in enumerate(['Intercept'] + self.x_names):
                p_valor = self.resultados.pvalues[i]
                coef = self.resultados.params[i]
                print(f"Variable: {var}")
                print(f"Coeficiente estimado: {coef:.4f}")
                print(f"p-valor: {p_valor:.4f}")
                if p_valor < alpha:
                    print(f"Conclusión: Rechazamos H0 (α={alpha}). Hay evidencia de que {var} es significativa.")
                else:
                    print(f"Conclusión: No rechazamos H0 (α={alpha}). No hay evidencia suficiente para afirmar que {var} es significativa.")
                print("-"*50)

        def comparar_modelos(self, modelo_reducido):
            if self.resultados is None:
                self.ajustar_modelo()
            if modelo_reducido.resultados is None:
                modelo_reducido.ajustar_modelo()
            anova_res = anova_lm(modelo_reducido.resultados, self.resultados)
            print("Comparación de modelos usando ANOVA:")
            print(anova_res)
            return anova_res

        def predecir(self, nuevos_datos, intervalo_confianza=False, intervalo_prediccion=False, confianza=0.95):
            if self.resultados is None:
                self.ajustar_modelo()
            if isinstance(nuevos_datos, dict):
                nuevos_datos = pd.DataFrame([nuevos_datos])
            nuevos_datos = nuevos_datos.copy()
            for var in self.x_names:
                if var not in nuevos_datos.columns:
                    nuevos_datos[var] = 0
            nuevos_datos = sm.add_constant(nuevos_datos, has_constant='add')
            columnas_necesarias = ['const'] + self.x_names
            nuevos_datos = nuevos_datos[columnas_necesarias]
            pred = self.resultados.get_prediction(nuevos_datos)
            pred_df = pd.DataFrame({'prediccion': pred.predicted_mean})
            if intervalo_confianza:
                ic = pred.conf_int(alpha=1-confianza)
                pred_df['ic_inf'] = ic[:, 0]
                pred_df['ic_sup'] = ic[:, 1]
            if intervalo_prediccion:
                ic_pred = pred.conf_int(obs=True, alpha=1-confianza)
                pred_df['pred_ic_inf'] = ic_pred[:, 0]
                pred_df['pred_ic_sup'] = ic_pred[:, 1]
            return pred_df

        def graficos_exploratorios(self):
            num_vars = len(self.x_names)
            cols = min(3, num_vars)
            rows = (num_vars + cols - 1)//cols
            fig, axs = plt.subplots(rows, cols, figsize=(5*cols, 4*rows))
            if rows == 1 and cols == 1:
                axs = np.array([[axs]])
            elif rows == 1 or cols == 1:
                axs = axs.reshape(-1, 1) if rows > 1 else axs.reshape(1, -1)
            for i, var in enumerate(self.x_names):
                row_idx = i//cols
                col_idx = i % cols
                axs[row_idx, col_idx].scatter(self.df[var], self.y, alpha=0.5)
                axs[row_idx, col_idx].set_title(f'{self.y_name} vs {var}')
                axs[row_idx, col_idx].set_xlabel(var)
                if col_idx == 0:
                    axs[row_idx, col_idx].set_ylabel(self.y_name)
            for i in range(num_vars, rows*cols):
                row_idx = i//cols
                col_idx = i % cols
                axs[row_idx, col_idx].axis('off')
            plt.tight_layout()
            plt.show()

    class RegresionLogistica:
        def __init__(self, dataframe, variable_respuesta, variables_predictoras, categoria_base=None):
            self.df = dataframe.copy()
            self.y_name = variable_respuesta
            self.x_names = variables_predictoras.copy() if variables_predictoras else []
            self.categoria_base = categoria_base
            self.modelo = None
            self.resultados = None
            self._preparar_datos()

        def _preparar_datos(self):
            if self.df[self.y_name].dtype in ['object', 'category']:
                self.df[self.y_name] = self.df[self.y_name].astype('category').cat.codes
            for var in [col for col in self.x_names if col in self.df.select_dtypes(['object', 'category']).columns]:
                if self.categoria_base is not None:
                    categories = [c for c in self.df[var].unique() if c != self.categoria_base]
                    categories = [self.categoria_base] + categories
                else:
                    categories = self.df[var].unique()
                self.df[var] = pd.Categorical(self.df[var], categories=categories)
                dummies = pd.get_dummies(self.df[var], prefix=var, drop_first=True)
                self.df = pd.concat([self.df.drop(var, axis=1), dummies], axis=1)
                self.x_names.remove(var)
                self.x_names.extend(dummies.columns.tolist())
            self.y = self.df[self.y_name].astype(int)
            self.X = self.df[self.x_names].astype(float)
            self.X = sm.add_constant(self.X)

        def agregar_interaccion(self, var1, var2, prefijo=None):
            if var1 not in self.df.columns or var2 not in self.df.columns:
                raise ValueError("Ambas variables deben existir en el DataFrame")
            nombre_interaccion = f"{prefijo}_" if prefijo else ""
            nombre_interaccion += f"{var1}_x_{var2}"
            if nombre_interaccion not in self.df.columns:
                self.df[nombre_interaccion] = self.df[var1]*self.df[var2]
                self.x_names.append(nombre_interaccion)
                self._preparar_datos()

        def ajustar_modelo(self):
            self.modelo = sm.Logit(self.y, self.X)
            self.resultados = self.modelo.fit(disp=0)
            return self.resultados

        def resumen_modelo(self):
            if self.resultados is None:
                self.ajustar_modelo()
            return self.resultados.summary()

        def predecir(self, nuevos_datos, umbral=0.5):
            if isinstance(nuevos_datos, dict):
                nuevos_datos = pd.DataFrame([nuevos_datos])
            nuevos_datos = nuevos_datos.copy()
            for var in self.x_names:
                if var not in nuevos_datos.columns:
                    nuevos_datos[var] = 0
            nuevos_datos = sm.add_constant(nuevos_datos, has_constant='add')
            columnas_necesarias = ['const'] + self.x_names
            nuevos_datos = nuevos_datos[columnas_necesarias]
            probabilidades = self.resultados.predict(nuevos_datos)
            predicciones = (probabilidades >= umbral).astype(int)
            return pd.DataFrame({'Probabilidad': probabilidades, 'Prediccion': predicciones})

        def evaluar_modelo(self, X_test=None, y_test=None, umbral=0.5):
            if X_test is None or y_test is None:
                X_test, y_test = self.X, self.y
            predicciones = self.predecir(X_test.drop(columns=['const'], errors='ignore'), umbral)['Prediccion']
            matriz_confusion = confusion_matrix(y_test, predicciones)
            reporte = classification_report(y_test, predicciones)
            print("Matriz de Confusión:")
            print(matriz_confusion)
            print("\nReporte de Clasificación:")
            print(reporte)
            return matriz_confusion, reporte

        def curva_roc(self, X_test=None, y_test=None):
            if X_test is None or y_test is None:
                X_test, y_test = self.X, self.y
            probabilidades = self.resultados.predict(X_test)
            fpr, tpr, _ = roc_curve(y_test, probabilidades)
            roc_auc = auc(fpr, tpr)
            plt.figure(figsize=(8, 6))
            plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'Curva ROC (AUC = {roc_auc:.2f})')
            plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
            plt.xlabel('Tasa de Falsos Positivos (1 - Especificidad)')
            plt.ylabel('Tasa de Verdaderos Positivos (Sensibilidad)')
            plt.title('Curva ROC')
            plt.legend(loc='lower right')
            plt.grid(True)
            plt.show()
            return roc_auc

        def graficos_diagnostico(self):
            if self.resultados is None:
                self.ajustar_modelo()
            residuos = self.resultados.resid_dev
            predichos = self.resultados.fittedvalues
            plt.figure(figsize=(15, 10))
            plt.subplot(2, 2, 1)
            plt.scatter(predichos, residuos)
            plt.axhline(y=0, color='r', linestyle='--')
            plt.xlabel('Probabilidades Predichas')
            plt.ylabel('Residuos de Devianza')
            plt.title('Residuos vs. Valores Predichos')
            plt.subplot(2, 2, 2)
            stats.probplot(residuos, dist="norm", plot=plt)
            plt.title('QQ Plot de Residuos')
            plt.subplot(2, 2, 3)
            plt.hist(residuos, bins=20, density=True, alpha=0.6)
            x = np.linspace(min(residuos), max(residuos), 100)
            plt.plot(x, norm.pdf(x, np.mean(residuos), np.std(residuos)), 'r-')
            plt.title('Distribución de Residuos')
            plt.tight_layout()
            plt.show()

        def test_hipotesis(self, alpha=0.05):
            if self.resultados is None:
                self.ajustar_modelo()
            print(f"Tests de hipótesis para coeficientes (α={alpha}):")
            print("H0: βi = 0 vs H1: βi ≠ 0\n")
            for i, var in enumerate(['Intercept'] + self.x_names):
                p_valor = self.resultados.pvalues[i]
                coef = self.resultados.params[i]
                print(f"Variable: {var}")
                print(f"Coeficiente estimado: {coef:.4f}")
                print(f"p-valor: {p_valor:.4f}")
                if p_valor < alpha:
                    print(f"Conclusión: Rechazamos H0 (α={alpha}). {var} es significativa.")
                else:
                    print(f"Conclusión: No rechazamos H0 (α={alpha}). {var} no es significativa.")
                print("-"*50)

        def vif(self):
            vif_data = pd.DataFrame()
            vif_data["Variable"] = self.x_names
            vif_data["VIF"] = [variance_inflation_factor(self.X.values, i+1) for i in range(len(self.x_names))]
            return vif_data