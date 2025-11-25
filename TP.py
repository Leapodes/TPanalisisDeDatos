import pandas as pd
import matplotlib.pyplot as plt
import geopandas as gpd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
import sys
import warnings

warnings.filterwarnings("ignore", message=".*GeoJSON does not support open option DRIVER.*")
warnings.filterwarnings("ignore", category=pd.errors.DtypeWarning)
warnings.filterwarnings("ignore", category=pd.errors.SettingWithCopyWarning)

try:
    sys.stdout.reconfigure(encoding='utf-8')
except:
    pass

#  CARGA DE DATOS


def cargar_datos():
    datos_dir = "Datos/"
    df_total = pd.DataFrame()

    for anio in range(16, 26):  # carga 2016-2025
        for trimestre in range(1, 5):

            if anio == 16 and trimestre == 1:
                continue
            elif anio == 25 and trimestre >= 3:
                continue

            archivo = datos_dir + f"usu_individual_T{trimestre}{anio}.txt"

            try:
                df_datos = pd.read_csv(archivo, sep=";", encoding="latin1")
                df_total = pd.concat([df_total, df_datos])
                print(f"{trimestre} Trimestre del año 20{anio} cargado.")
            except:
                pass

    return df_total


#  AJUSTE POR INFLACIÓN  (P47T_real)

def ajustar_por_inflacion(df_total):

    try:
        ipc = pd.read_csv("Datos/ipc_trimestral.csv", encoding="utf-8")
    except:
        print("No se encontró ipc_trimestral.csv. Se usa ingreso nominal.")
        #Crear P47T_real aunque no haya archivo IPC
        df_total["P47T_real"] = pd.to_numeric(df_total["P47T"], errors="coerce")
        return df_total

    ipc["ANO4"] = ipc["ANO4"].astype(int)
    ipc["TRIMESTRE"] = ipc["TRIMESTRE"].astype(int)

    base = ipc[(ipc["ANO4"] == 2024) & (ipc["TRIMESTRE"] == 4)]["IPC"].values
    if len(base) == 0:
        base = ipc["IPC"].iloc[-1]
    else:
        base = base[0]

    df_total["ANO4"] = pd.to_numeric(df_total["ANO4"], errors="coerce")
    df_total["TRIMESTRE"] = pd.to_numeric(df_total["TRIMESTRE"], errors="coerce")

    df_total = df_total.merge(ipc, on=["ANO4", "TRIMESTRE"], how="left")

    df_total["P47T"] = pd.to_numeric(df_total["P47T"], errors="coerce")

    df_total["P47T_real"] = (df_total["P47T"] * (base / df_total["IPC"]))

    return df_total


#  CÁLCULO DE TASAS (NUEVO)

def calcular_tasas(df_total):
    """
    Calcula las tasas de actividad, empleo y desocupación por período y aglomerado
    """
    df = df_total.copy()
    
    # Filtrar aglomerados de interés
    df = df[df["AGLOMERADO"].isin([18, 27])]
    
    # Crear período
    df["PERIODO"] = df["ANO4"].astype(int).astype(str) + "-T" + df["TRIMESTRE"].astype(int).astype(str)
    
    # Convertir ESTADO a numérico
    df["ESTADO"] = pd.to_numeric(df["ESTADO"], errors="coerce")
    df = df.dropna(subset=["ESTADO"])
    
    # Agrupar por período y aglomerado
    resultado = []
    
    for periodo in df["PERIODO"].unique():
        for aglom in [18, 27]:
            df_temp = df[(df["PERIODO"] == periodo) & (df["AGLOMERADO"] == aglom)]
            
            if len(df_temp) == 0:
                continue
            
            # Contar población según estado
            poblacion_total = len(df_temp)
            ocupados = len(df_temp[df_temp["ESTADO"] == 1])
            desocupados = len(df_temp[df_temp["ESTADO"] == 2])
            inactivos = len(df_temp[df_temp["ESTADO"] == 3])
            
            # PEA = Ocupados + Desocupados
            pea = ocupados + desocupados
            
            # Calcular tasas
            if pea > 0:
                tasa_desocupacion = (desocupados / pea) * 100
            else:
                tasa_desocupacion = 0
            
            if poblacion_total > 0:
                tasa_actividad = (pea / poblacion_total) * 100
                tasa_empleo = (ocupados / poblacion_total) * 100
            else:
                tasa_actividad = 0
                tasa_empleo = 0
            
            nombres_aglomerados = {18: "Gran Mendoza", 27: "Comodoro Rivadavia"}
            
            resultado.append({
                "PERIODO": periodo,
                "AGLOMERADO": nombres_aglomerados[aglom],
                "Tasa_Actividad": tasa_actividad,
                "Tasa_Empleo": tasa_empleo,
                "Tasa_Desocupacion": tasa_desocupacion,
                "Ocupados": ocupados,
                "Desocupados": desocupados,
                "PEA": pea,
                "Poblacion_Total": poblacion_total
            })
    
    df_tasas = pd.DataFrame(resultado)
    return df_tasas


# FUNCIONES INDIVIDUALES PARA CADA GRÁFICO DE TASAS

def mostrar_tabla_tasas(df_total):
    """Muestra solo la tabla de tasas"""
    df_tasas = calcular_tasas(df_total)
    
    if len(df_tasas) == 0:
        print("No hay datos disponibles para calcular tasas")
        return
    
    print("\n" + "="*80)
    print(" TASAS LABORALES POR PERÍODO Y AGLOMERADO")
    print("="*80)
    print(df_tasas.to_string(index=False))
    print("="*80)


def grafico_tasa_actividad(df_total):
    """Gráfico solo de Tasa de Actividad"""
    df_tasas = calcular_tasas(df_total)
    
    if len(df_tasas) == 0:
        print("No hay datos disponibles")
        return
    
    pivot = df_tasas.pivot(index="PERIODO", columns="AGLOMERADO", values="Tasa_Actividad")
    pivot.plot(kind="bar", figsize=(10, 6))
    plt.title("Tasa de Actividad (%)", fontsize=14, weight="bold")
    plt.xlabel("Período")
    plt.ylabel("Porcentaje (%)")
    plt.legend(loc="best")
    plt.grid(True, alpha=0.3, axis='y')
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()


def grafico_tasa_empleo(df_total):
    """Gráfico solo de Tasa de Empleo"""
    df_tasas = calcular_tasas(df_total)
    
    if len(df_tasas) == 0:
        print("No hay datos disponibles")
        return
    
    pivot = df_tasas.pivot(index="PERIODO", columns="AGLOMERADO", values="Tasa_Empleo")
    pivot.plot(kind="bar", figsize=(10, 6))
    plt.title("Tasa de Empleo (%)", fontsize=14, weight="bold")
    plt.xlabel("Período")
    plt.ylabel("Porcentaje (%)")
    plt.legend(loc="best")
    plt.grid(True, alpha=0.3, axis='y')
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()


def grafico_tasa_desocupacion(df_total):
    """Gráfico solo de Tasa de Desocupación"""
    df_tasas = calcular_tasas(df_total)
    
    if len(df_tasas) == 0:
        print("No hay datos disponibles")
        return
    
    pivot = df_tasas.pivot(index="PERIODO", columns="AGLOMERADO", values="Tasa_Desocupacion")
    pivot.plot(kind="bar", figsize=(10, 6))
    plt.title("Tasa de Desocupación (%)", fontsize=14, weight="bold")
    plt.xlabel("Período")
    plt.ylabel("Porcentaje (%)")
    plt.legend(loc="best")
    plt.grid(True, alpha=0.3, axis='y')
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()


#  FUNCIONES INDIVIDUALES PARA CADA GRÁFICO DE INGRESOS

def mostrar_tabla_ingresos(df_total):
    """Muestra solo la tabla de ingresos"""
    df = df_total.copy()
    
    df = df[df["AGLOMERADO"].isin([18, 27])]
    df["P47T_real"] = pd.to_numeric(df["P47T_real"], errors="coerce")
    df = df[df["P47T_real"] > 0].dropna(subset=["P47T_real"])
    
    if len(df) == 0:
        print("No hay datos de ingresos disponibles")
        return
    
    df["PERIODO"] = df["ANO4"].astype(int).astype(str) + "-T" + df["TRIMESTRE"].astype(int).astype(str)
    
    nombres_aglomerados = {18: "Gran Mendoza", 27: "Comodoro Rivadavia"}
    df["AGLOMERADO_NOMBRE"] = df["AGLOMERADO"].map(nombres_aglomerados)
    
    df_grouped = df.groupby(["PERIODO", "AGLOMERADO_NOMBRE"])["P47T_real"].agg([
        ("Media", "mean"),
        ("Mediana", "median"),
    ]).reset_index()
    
    print("\n" + "="*80)
    print(" EVOLUCIÓN DE INGRESOS REALES")
    print("="*80)
    print(df_grouped.to_string(index=False))
    print("="*80)


def grafico_ingreso_promedio(df_total):
    """Gráfico solo de Ingreso Promedio"""
    df = df_total.copy()
    
    df = df[df["AGLOMERADO"].isin([18, 27])]
    df["P47T_real"] = pd.to_numeric(df["P47T_real"], errors="coerce")
    df = df[df["P47T_real"] > 0].dropna(subset=["P47T_real"])
    
    if len(df) == 0:
        print("No hay datos de ingresos disponibles")
        return
    
    df["PERIODO"] = df["ANO4"].astype(int).astype(str) + "-T" + df["TRIMESTRE"].astype(int).astype(str)
    nombres_aglomerados = {18: "Gran Mendoza", 27: "Comodoro Rivadavia"}
    df["AGLOMERADO_NOMBRE"] = df["AGLOMERADO"].map(nombres_aglomerados)
    
    df_grouped = df.groupby(["PERIODO", "AGLOMERADO_NOMBRE"])["P47T_real"].mean().reset_index()
    df_grouped.columns = ["PERIODO", "AGLOMERADO_NOMBRE", "Media"]
    
    pivot = df_grouped.pivot(index="PERIODO", columns="AGLOMERADO_NOMBRE", values="Media")
    pivot.plot(kind="bar", figsize=(10, 6))
    plt.title("Ingreso Promedio Real", fontsize=14, weight="bold")
    plt.xlabel("Período")
    plt.ylabel("Ingreso Real ($)")
    plt.legend(loc="best")
    plt.grid(True, alpha=0.3, axis='y')
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()


def grafico_ingreso_mediano(df_total):
    """Gráfico solo de Ingreso Mediano"""
    df = df_total.copy()
    
    df = df[df["AGLOMERADO"].isin([18, 27])]
    df["P47T_real"] = pd.to_numeric(df["P47T_real"], errors="coerce")
    df = df[df["P47T_real"] > 0].dropna(subset=["P47T_real"])
    
    if len(df) == 0:
        print("No hay datos de ingresos disponibles")
        return
    
    df["PERIODO"] = df["ANO4"].astype(int).astype(str) + "-T" + df["TRIMESTRE"].astype(int).astype(str)
    nombres_aglomerados = {18: "Gran Mendoza", 27: "Comodoro Rivadavia"}
    df["AGLOMERADO_NOMBRE"] = df["AGLOMERADO"].map(nombres_aglomerados)
    
    df_grouped = df.groupby(["PERIODO", "AGLOMERADO_NOMBRE"])["P47T_real"].median().reset_index()
    df_grouped.columns = ["PERIODO", "AGLOMERADO_NOMBRE", "Mediana"]
    
    pivot = df_grouped.pivot(index="PERIODO", columns="AGLOMERADO_NOMBRE", values="Mediana")
    pivot.plot(kind="bar", figsize=(10, 6))
    plt.title("Ingreso Mediano Real", fontsize=14, weight="bold")
    plt.xlabel("Período")
    plt.ylabel("Ingreso Real ($)")
    plt.legend(loc="best")
    plt.grid(True, alpha=0.3, axis='y')
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()


#  UNIVARIADO

def analisar_univariado(df_total, variable):

    tasa = {
        "Ocupados": 1,
        "Desocupados": 2,
        "Inactivo": 3,
        "Menor de 10 años": 4
    }

    df_total["ESTADO"] = pd.to_numeric(df_total["ESTADO"], errors="coerce")
    df_total = df_total.dropna(subset=["ESTADO"])
    df_total["ESTADO"] = df_total["ESTADO"].astype(int)

    df_total["AGLOMERADO"] = pd.to_numeric(df_total["AGLOMERADO"], errors="coerce")
    df_total = df_total.dropna(subset=["AGLOMERADO"])
    df_total["AGLOMERADO"] = df_total["AGLOMERADO"].astype(int)

    # Se ajusta a los aglomerados que pide el trabajo
    df_filtrado = df_total[df_total["AGLOMERADO"].isin([18, 27])]

    df_limpio = df_filtrado[df_filtrado["ESTADO"] == tasa[variable]]
    df_final = df_limpio[["ESTADO", "ANO4", "TRIMESTRE", "AGLOMERADO"]]

    df_estado_elegido = df_final[df_final["ESTADO"] == tasa[variable]]

    # Verificar que hay datos
    if len(df_estado_elegido) == 0:
        print(f"\nNo hay datos de '{variable}' para mostrar.")
        return

    df_estado_elegido["PERIODO"] = (
        df_estado_elegido["ANO4"].astype(str) + "-T" + df_estado_elegido["TRIMESTRE"].astype(str)
    )

    df_grouped = df_estado_elegido.groupby(["PERIODO", "AGLOMERADO"]).size().reset_index(name="TOTAL")

    pivot = df_grouped.pivot(index="PERIODO", columns="AGLOMERADO", values="TOTAL").fillna(0)

    nombres_aglomerados = {
        18: "Gran Mendoza",
        27: "Comodoro Rivadavia"
    }
    
    pivot = pivot.rename(columns=nombres_aglomerados)
    pivot.plot(kind="bar", figsize=(10, 6))# Crear figura explícita
    plt.title(f"Total de {variable} por Año y Trimestre")
    plt.xticks(rotation=45)
    plt.legend(loc="upper left")
    plt.tight_layout()
    plt.show()

# ESTADÍSTICAS RESUMEN (MEDIA, MEDIANA, PERCENTILES)

def estadisticas_resumen(df_total):
    print("="*48)
    print(" MEDIDAS DE TENDENCIA CENTRAL DE INGRESOS")
    print("="*48)

    df = df_total.copy()

    # Verificar que existe P47T_real
    if "P47T_real" not in df.columns:
        print("ERROR: La columna P47T_real no existe.")
        print("Verifique que se ejecutó ajustar_por_inflacion().")
        return

    df["P47T_real"] = pd.to_numeric(df["P47T_real"], errors="coerce")
    df = df.dropna(subset=["P47T_real"])
    
    # Filtrar ingresos positivos
    df = df[df["P47T_real"] > 0]

    if len(df) == 0:
        print("No hay datos de ingresos disponibles.")
        return

    media = df["P47T_real"].mean()
    mediana = df["P47T_real"].median()

    print(f"Media:        ${media:,.2f}")
    print(f"Mediana:      ${mediana:,.2f}")
    print("="*48)

#  MULTIVARIADO

def analizar_multivariado(df_total, variable):

    # Mapas
    mapa_estado = {
        "Ocupados": 1,
        "Desocupados": 2,
        "Inactivo": 3,
        "Menor de 10 años": 4
    }

    mapa_sexo = {1: "Masculino", 2: "Femenino"}

    # Educación simple
    def clasificar_educacion(x):
        try:
            x = int(x)
        except:
            return "NS/NR"

        if x in [1,2,3,4,5,6]:   # Primaria + Secundaria
            return "Básico"
        if x in [7,8,9]:         # Superior
            return "Superior"
        return "NS/NR"

    df = df_total.copy()

    for col in ["AGLOMERADO", "ANO4", "TRIMESTRE"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    df = df.dropna(subset=["AGLOMERADO", "ANO4", "TRIMESTRE"])
    df["AGLOMERADO"] = df["AGLOMERADO"].astype(int)
    df = df[df["AGLOMERADO"].isin([18, 27])]

    df["PERIODO"] = df["ANO4"].astype(int).astype(str) + "-T" + df["TRIMESTRE"].astype(int).astype(str)

    # ESTADO (con SEXO)
    if variable in mapa_estado:

        df = df.copy()
        df["ESTADO"] = pd.to_numeric(df["ESTADO"], errors="coerce")
        df["CH04"] = pd.to_numeric(df["CH04"], errors="coerce")

        df = df.dropna(subset=["ESTADO", "CH04"])
        df["ESTADO"] = df["ESTADO"].astype(int)
        df["CH04"] = df["CH04"].astype(int)

        df = df[df["ESTADO"] == mapa_estado[variable]]
        df["SEXO"] = df["CH04"].map(mapa_sexo)

        if len(df) == 0:
            print(f"No hay datos suficientes para {variable}.")
            return

        df_grouped = df.groupby(["PERIODO", "AGLOMERADO", "SEXO"]).size().reset_index(name="TOTAL")

        nombres_aglomerados = {
            18: "Gran Mendoza",
            27: "Comodoro Rivadavia"
        }
        
        df_grouped["AGLOMERADO_TXT"] = df_grouped["AGLOMERADO"].map(nombres_aglomerados)
        df_grouped["CATEGORIA"] = df_grouped["AGLOMERADO_TXT"] + "-" + df_grouped["SEXO"]

        pivot = df_grouped.pivot(index="PERIODO", columns="CATEGORIA", values="TOTAL").fillna(0)

        pivot.plot(kind="bar", figsize=(10, 6))  # Crear figura explícita
        plt.title(f"{variable} — Comparación por Sexo y Aglomerado")
        plt.xticks(rotation=45)
        plt.legend(loc="upper left")
        plt.tight_layout()
        plt.show()
        return

    # EDUCACIÓN SIMPLE
    if variable.lower() == "educacion":

        nombres_aglomerados = {
            18: "Gran Mendoza",
            27: "Comodoro Rivadavia"
        }

        # Clasificar nivel educativo
        df["NIVEL_SIMPLE"] = df["NIVEL_ED"].apply(clasificar_educacion)

        df_grouped = df.groupby(
            ["PERIODO", "AGLOMERADO", "NIVEL_SIMPLE"]
        ).size().reset_index(name="TOTAL")

        df_grouped["AGLOMERADO_TXT"] = df_grouped["AGLOMERADO"].map(nombres_aglomerados)

        df_grouped["CATEGORIA"] = (
            df_grouped["AGLOMERADO_TXT"] + " - " + df_grouped["NIVEL_SIMPLE"]
        )

        pivot = df_grouped.pivot(
            index="PERIODO", 
            columns="CATEGORIA", 
            values="TOTAL"
        ).fillna(0)

        # Gráfico uniforme
        pivot.plot(kind="bar", figsize=(8, 6))# Crear figura explícita
        plt.title("Nivel Educativo — Comparación entre Aglomerados")
        plt.xticks(rotation=45)
        plt.legend(loc="upper left")
        plt.tight_layout()
        plt.show()
        return


#  MODELO DE REGRESIÓN + IMPUTACIÓN (MEJORADO)

def modelacion_regresion(df_total):
    """
    Modelo de regresión para IMPUTAR ingresos faltantes
    """
    df = df_total.copy()

    # Preparar variables
    df["P47T"] = pd.to_numeric(df["P47T"], errors="coerce")
    df["CH06"] = pd.to_numeric(df["CH06"], errors="coerce")
    df["NIVEL_ED"] = pd.to_numeric(df["NIVEL_ED"], errors="coerce")
    df["CH04"] = pd.to_numeric(df["CH04"], errors="coerce")
    df["PP04B_COD"] = pd.to_numeric(df["PP04B_COD"], errors="coerce")
    df["PP04D_COD"] = pd.to_numeric(df["PP04D_COD"], errors="coerce")

    # Separar datos CON ingreso (para entrenar) y SIN ingreso (para imputar)
    df_con_ingreso = df.dropna(subset=["P47T", "CH06", "NIVEL_ED", "CH04"])
    df_sin_ingreso = df[df["P47T"].isna() & df["CH06"].notna() & df["NIVEL_ED"].notna() & df["CH04"].notna()]

    if len(df_con_ingreso) == 0:
        print("No hay datos suficientes para entrenar el modelo")
        return None

    # Entrenar modelo con datos completos
    X = df_con_ingreso[["CH06", "NIVEL_ED", "CH04", "PP04B_COD", "PP04D_COD"]].fillna(-1)
    y = df_con_ingreso["P47T"]

    x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

    modelo = LinearRegression()
    modelo.fit(x_train, y_train)

    y_pred = modelo.predict(x_test)

    print("\n" + "="*60)
    print(" MODELO DE REGRESIÓN PARA IMPUTACIÓN DE INGRESOS")
    print("="*60)
    print(f"Error MSE: {mean_squared_error(y_test, y_pred):,.2f}")
    print(f"R²: {r2_score(y_test, y_pred):.4f}")

    coef_df = pd.DataFrame({
        "Variable": [
            "Edad",
            "Nivel educativo",
            "Sexo",
            "Categoría laboral",
            "Rama de actividad"
        ],
        "Coeficiente": modelo.coef_
    })

    print("\nInfluencia de las variables:")
    print("-"*60)
    print(f"{'Variable':<40} | {'Coeficiente':>15}")
    print("-"*60)
    for v, c in zip(coef_df["Variable"], coef_df["Coeficiente"]):
        print(f"{v:<40} | {c:>15,.2f}")
    print("-"*60)

    # IMPUTACIÓN
    if len(df_sin_ingreso) > 0:
        X_imputar = df_sin_ingreso[["CH06", "NIVEL_ED", "CH04", "PP04B_COD", "PP04D_COD"]].fillna(-1)
        ingresos_imputados = modelo.predict(X_imputar)
        
        print(f"\n Se imputaron {len(ingresos_imputados)} valores faltantes de ingreso")
        print(f"  Ingreso imputado promedio: ${ingresos_imputados.mean():,.2f}")
        print(f"  Rango: ${ingresos_imputados.min():,.2f} - ${ingresos_imputados.max():,.2f}")
    else:
        print("\nNo hay registros con ingreso faltante para imputar")

    print("="*60 + "\n")

    return modelo


def mapa_aglomerados():
    # Cargar el archivo
    try:
        mapa = gpd.read_file("Datos/aglomerados_eph.json", driver="GeoJSON")
    except Exception as e:
        print("Error al cargar el archivo GeoJSON:", e)
        return

    # Aglomerados - filtrar por nombre directamente
    nombres = ["Gran Mendoza", "Comodoro Rivadavia"]
    
    mapa_filtrado = mapa[mapa["aglomerado"].isin(nombres)].copy()
    
    # Verificar que se encontraron datos
    if len(mapa_filtrado) == 0:
        print("No se encontraron Gran Mendoza y Comodoro Rivadavia en el archivo")
        return
    
    mapa_filtrado["nombre"] = mapa_filtrado["aglomerado"]

    # Asegurar que tiene sistema de coordenadas
    if mapa_filtrado.crs is None:
        mapa_filtrado = mapa_filtrado.set_crs("EPSG:4326")

    # Plot
    fig, axes = plt.subplots(1, 2, figsize=(11, 6))
    
    # Mapa 1: Gran Mendoza
    mendoza = mapa_filtrado[mapa_filtrado["nombre"] == "Gran Mendoza"]
    mendoza.plot(ax=axes[0], color="lightblue", edgecolor="black")
    axes[0].set_title("Gran Mendoza", fontsize=12, weight='bold')
    axes[0].set_xlabel("Longitud")
    axes[0].set_ylabel("Latitud")
    axes[0].grid(True, linestyle="--", alpha=0.4)
    
    # Mapa 2: Comodoro Rivadavia
    comodoro = mapa_filtrado[mapa_filtrado["nombre"] == "Comodoro Rivadavia"]
    comodoro.plot(ax=axes[1], color="lightcoral", edgecolor="black")
    axes[1].set_title("Comodoro Rivadavia", fontsize=12, weight='bold')
    axes[1].set_xlabel("Longitud")
    axes[1].set_ylabel("Latitud")
    axes[1].grid(True, linestyle="--", alpha=0.4)
    
    fig.suptitle("Aglomerados EPH: Gran Mendoza y Comodoro Rivadavia", fontsize=14, weight='bold')
    plt.tight_layout()
    plt.show()


#  MENÚ MEJORADO

def menu(df_total):

    while True:
        print("\n" + "="*70)
        print(" MENÚ DE ANÁLISIS DE DATOS - TRABAJO PRÁCTICO EPH")
        print("="*70 + "\n")
        print("--- APROBACIÓN NO DIRECTA (4-5 puntos) ---\n" \
        "\n1) Análisis univariado\n" \
        "2) Medidas de tendencia central y posición\n" \
        "3) Tasas laborales\n" \
        "4) Evolución de ingresos reales\n" \
        "5) Análisis multivariado\n" \
        "\n--- APROBACIÓN DIRECTA (6-10 puntos) ---\n" \
        "\n6) Modelo de regresión e imputación de ingresos\n" \
        "7) Mapa georreferenciado\n"
        "\n--- UTILIDADES ---\n"
        "\n8) Volver a cargar datos\n"
        "0) Salir")
        print("="*70)

        opcion = input("\nSeleccione una opción: ").strip()

        # OPCIÓN 1: ANÁLISIS UNIVARIADO
        if opcion == "1":
            while True:
                print("\nAnálisis Univariado - Variables:")
                print("1 = Ocupados\n"
                "2 = Desocupados\n"
                "3 = Inactivo\n"
                "4 = Menor de 10 años\n"
                "0 = Volver")

                variable = input("Ingrese la opción: ").strip()
                
                if variable == "0":
                    break
                
                mapa = {
                    "1": "Ocupados",
                    "2": "Desocupados",
                    "3": "Inactivo",
                    "4": "Menor de 10 años"
                }
                
                if variable in mapa:
                    analisar_univariado(df_total, mapa[variable])
                else:
                    print("Opción inválida.")

        # OPCIÓN 2: ESTADÍSTICAS RESUMEN
        elif opcion == "2":
            estadisticas_resumen(df_total)

        # OPCIÓN 3: TASAS LABORALES (SUBMENU)
        elif opcion == "3":
            while True: 
                print("Tasas Laborales - Opciones:\n"
                "1 = Tabla de tasas laborales\n"
                "2 = Gráfico Tasa de Actividad\n"
                "3 = Gráfico Tasa de Empleo\n"
                "4 = Gráfico Tasa de Desocupación\n"
                "0 = Volver\n")
                opcion = input("Ingrese la opción: ").strip()
                if opcion == "1":
                    mostrar_tabla_tasas(df_total)
                elif opcion == "2":
                    grafico_tasa_actividad(df_total)
                elif opcion == "3":
                    grafico_tasa_empleo(df_total)
                elif opcion == "4":
                    grafico_tasa_desocupacion(df_total)
                if opcion == "0":
                    break


        # OPCIÓN 4: EVOLUCIÓN DE INGRESOS (SUBMENU)
        elif opcion == "4":
            while True:
                print("Evolución de Ingresos Reales - Opciones:\n"
                "1 = Tabla de ingresos\n"
                "2 = Gráfico Ingreso Promedio\n"
                "3 = Gráfico Ingreso Mediano\n"
                "0 = Volver\n")
                opcion = input("Ingrese la opción: ").strip()
                if opcion == "1":
                    mostrar_tabla_ingresos(df_total)
                elif opcion == "2":
                    grafico_ingreso_promedio(df_total)
                elif opcion == "3": 
                    grafico_ingreso_mediano(df_total)
                if opcion == "0":
                    break   

        # OPCIÓN 5: ANÁLISIS MULTIVARIADO
        elif opcion == "5":
            while True:
                print("\nAnálisis Multivariado - Variables:")
                print("1 = Ocupados\n" 
                "2 = Desocupados\n" 
                "3 = Inactivo\n" 
                "4 = Menor de 10 años\n" 
                "5 = Educación\n" 
                "0 = Volver")
                variable = input("Ingrese la opción: ").strip()
                
                if variable == "0":
                    break
                
                mapa = {
                    "1": "Ocupados",
                    "2": "Desocupados",
                    "3": "Inactivo",
                    "4": "Menor de 10 años",
                    "5": "Educacion",
                }
                
                if variable in mapa:
                    analizar_multivariado(df_total, mapa[variable])
                else:
                    print("Opción inválida.")

        # OPCIÓN 6: MODELO DE REGRESIÓN
        elif opcion == "6":
            print("\nEjecutando modelo de regresión e imputación...")
            modelacion_regresion(df_total)

        # OPCIÓN 7: MAPA GEORREFERENCIADO
        elif opcion == "7":
            print("\nMostrando mapa georreferenciado de aglomerados...")
            mapa_aglomerados()

        # OPCIÓN 8: RECARGAR DATOS
        elif opcion == "8":
            print("\nRecargando datos...")
            df_total = cargar_datos()
            df_total = ajustar_por_inflacion(df_total)

        # OPCIÓN 0: SALIR
        elif opcion == "0":
            print("Saliendo...")
            break

        else:
            print("Opción inválida.\n")

    return df_total


# EJECUCIÓN PRINCIPAL

if __name__ == "__main__":
    print("="*70)
    print("TRABAJO PRÁCTICO ANALISIS DE DATOS")
    print("="*70)
    
    df = cargar_datos()
    df = ajustar_por_inflacion(df)
    
    print("\n Datos cargados correctamente")
    print(f"  Total de registros: {len(df):,}")
    
    df = menu(df)