import pandas as pd
import matplotlib.pyplot as plt
import geopandas as gpd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
import sys
import warnings


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

    for anio in range(16, 26):
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



    pivot.plot(kind='bar')
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
    p25 = df["P47T_real"].quantile(0.25)
    p75 = df["P47T_real"].quantile(0.75)

    print(f"Media:      ${media:,.2f}")
    print(f"Mediana:    ${mediana:,.2f}")
    print(f"Percentil 25: ${p25:,.2f}")
    print(f"Percentil 75: ${p75:,.2f}")
    print("="*46)

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

        df_grouped["CATEGORIA"] = (
            df_grouped["AGLOMERADO"].astype(str) + " - " + df_grouped["SEXO"]
        )

        pivot = df_grouped.pivot(index="PERIODO", columns="CATEGORIA", values="TOTAL").fillna(0)

        nombres_aglomerados = {
        18: "Gran Mendoza",
        27: "Comodoro Rivadavia"
    }
        
        df_grouped["AGLOMERADO_TXT"] = df_grouped["AGLOMERADO"].map(nombres_aglomerados)


        df_grouped["CATEGORIA"] = df_grouped["AGLOMERADO_TXT"] + "-" + df_grouped["SEXO"]

        pivot = df_grouped.pivot(index="PERIODO", columns="CATEGORIA", values="TOTAL").fillna(0)
    

        pivot.plot(kind="bar", figsize=(8, 4))
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
        pivot.plot(kind="bar", figsize=(8, 4))
        plt.title("Nivel Educativo — Comparación entre Aglomerados")
        plt.xticks(rotation=45)
        plt.legend(loc="upper left")
        plt.tight_layout()
        plt.show()
        return


#  MODELO DE REGRESIÓN + IMPUTACIÓN (punto obligatorio)

def modelacion_regresion(df_total):

    df = df_total.copy()

    df["P47T"] = pd.to_numeric(df["P47T"], errors="coerce")
    df["CH06"] = pd.to_numeric(df["CH06"], errors="coerce")
    df["NIVEL_ED"] = pd.to_numeric(df["NIVEL_ED"], errors="coerce")
    df["CH04"] = pd.to_numeric(df["CH04"], errors="coerce")
    df["PP04B_COD"] = pd.to_numeric(df["PP04B_COD"], errors="coerce")
    df["PP04D_COD"] = pd.to_numeric(df["PP04D_COD"], errors="coerce")

    df = df.dropna(subset=["P47T", "CH06", "NIVEL_ED", "CH04"])

    X = df[["CH06", "NIVEL_ED", "CH04", "PP04B_COD", "PP04D_COD"]].fillna(-1)
    y = df["P47T"]

    x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

    modelo = LinearRegression()
    modelo.fit(x_train, y_train)

    y_pred = modelo.predict(x_test)

    print(f"Error MSE: {mean_squared_error(y_test, y_pred):.2f}")
    print(f"R²: {r2_score(y_test, y_pred):.4f}")

    coef_df = pd.DataFrame({
        "Variable": [
            "Edad",
            "Nivel educativo",
            "Sexo",
            "Categoría laboral (PP04B_COD)",
            "Rama de actividad (PP04D_COD)"
        ],
        "Coeficiente": modelo.coef_
    })

    print("\nInfluencia de las variables:")
    print("-"*55)
    print("Variable                       |  Coeficiente")
    print("-"*55)
    for v, c in zip(coef_df["Variable"], coef_df["Coeficiente"]):
        print(f"{v:<30} | {c:>10.2f}")
    print("-"*55)


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

#  MENÚ

def menu(df_total):

    while True:
        print("="*50)
        print(" MENÚ DE ANALISIS DE DATOS")
        print("="*50)

        print(
            "1) Analisis univariado\n"
            "2) Analisis multivariado\n"
            "3) Modelo de regresión e imputación\n"
            "4) Volver a cargar datos\n"
            "5) Medidas de tendencia central\n"
            "6) Mapa georreferenciado\n"
            "0) Salir\n"
        )

        opcion = input("Seleccione una opción: ")

        while opcion == "1":
            print("\nVariaciones:")
            print("1 = Ocupados\n"
            "2 = Desocupados\n"
            "3 = Inactivo\n"
            "4 = Menor de 10 años\n"
            "0 = Volver")

            variable = input("Ingrese la opción: ")
            while variable not in ["1", "2", "3", "4", "0"]:
                variable = input("Opción inválida. Ingrese la opción: ")
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

        while opcion == "2":
            print("\nVariaciones:")
            print("1 = Ocupados\n"
            "2 = Desocupados\n"
            "3 = Inactivo\n"
            "4 = Menor de 10 años\n"
            "5 = Educacion\n"
            "0 = Volver")

            variable = input("Ingrese la opción: ")
            while variable not in ["1", "2", "3", "4", "5", "0"]:
                variable = input("Opción inválida. Ingrese la opción: ")
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

        if opcion == "3":
            print("\nEjecutando modelo de regresión...")
            modelacion_regresion(df_total)

        elif opcion == "4":
            print("\nRecargando datos...")
            df_total = cargar_datos()
            df_total = ajustar_por_inflacion(df_total)
        
        elif opcion == "5":
            estadisticas_resumen(df_total)

        elif opcion == "6":
            mapa_aglomerados()

        elif opcion == "0":
            print("Saliendo...")
            break

        else:
            print("Opcion inválida.\n")

# EJECUCIÓN PRINCIPAL

df = cargar_datos()
df = ajustar_por_inflacion(df)
menu(df)