import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

def cargar_datos():
    datos_dir = "Datos/"    # Carpeta Datos dentro del proyecto
    df_total = pd.DataFrame()
    for anio in range(16, 25): #recorrer en relacion a la cantidad de años de los datos establecidos (del 2016 al 2025)
        for trimestre in range(1, 5): #recorrer la cantidad de trimestres de los años (del primero al cuarto)
            if anio == 16 and trimestre == 1:
                continue
            elif anio == 25 and (trimestre == 3 or trimestre == 4):
                continue

            archivo = datos_dir + f"usu_individual_T{trimestre}{anio}.txt"

            df_datos = pd.read_csv(archivo, sep=";")
            df_total = pd.concat([df_total, df_datos])
            print(f"{trimestre} Trimestre del año {anio} cargado ")
    return df_total

def analisar_univariado(df_total, variable):
    tasa = {
        "Ocupados": 1,
        "Desocupados": 2,
        "Inactivo": 3,
        "Menor de 10 años": 4
    }
    #LIMPIEZA DE DATOS

    df_total["ESTADO"] = pd.to_numeric(df_total["ESTADO"], errors="coerce") #pone todos nros de 'estado' a tipo numerico y el coerce es que si hay un error que lo deje pasar (si no puede cambiar algo a tipo numerico)
    df_total = df_total.dropna(subset=["ESTADO"]) #los que no puede pasar a tipo numerico borra todas las filas vacias que tengan el elemento 'estado' vacio
    df_total["ESTADO"] = df_total["ESTADO"].astype(int) #que pase todo lo de 'estado' a nro entero (int)

    #recorre toda la columna, encuentra la columna, pone todos sus datos en tipo numerico (cualquier tipo de numero) si no puede, el coerce le pone 'nan' y lo deja pasar
    # cuando termina, todos los tipos numericos, los pasa a un tipo de numero entero (int) para ser mas exacto 

    df_total["AGLOMERADO"] = pd.to_numeric(df_total["AGLOMERADO"], errors="coerce")
    df_total = df_total.dropna(subset=["AGLOMERADO"])
    df_total["AGLOMERADO"] = df_total["AGLOMERADO"].astype(int)

    #exactamente igual que el anterior, recorre, analisa y borra lo que sobre o haya espacio


    df_filtrado = df_total[df_total["AGLOMERADO"].isin([9, 10])]
    df_limpio = df_filtrado[df_filtrado["ESTADO"] > 0]
    df_final = df_limpio[["ESTADO", "ANO4", "TRIMESTRE"]]

    df_estado_elegido = df_final[df_final["ESTADO"] == tasa[variable]]
    
    df_estado_elegido["ANO4-TRIMESTRE"] = df_estado_elegido["ANO4"].astype(str) + "-T" + df_estado_elegido["TRIMESTRE"].astype(str)
    '''
    label_estados = {
        1: "Ocupado",
        2: "Desocupado",
        3: "Inactivo",
        4: "Menor10"
    }
    df_estado_elegido["ESTADOSTR"] = df_estado_elegido["ESTADO"].map(label_estados)
    '''
    df_grouped = df_estado_elegido.groupby(["ANO4-TRIMESTRE", "ESTADO"]).agg(TOTAL=('ESTADO', 'count')).reset_index()
    pivot = df_grouped.pivot(index='ANO4-TRIMESTRE', columns='ESTADO', values='TOTAL')


    pivot.plot(kind='bar')
    plt.title(f"Total de {variable} por Año y Trimestre")
    plt.xlabel("Año")
    plt.xticks(rotation=45)
    plt.ylabel(f"Cantidad de {variable}")
    plt.ylim(0, 1800)
    #plt.legend(title="Estado", loc="upper right", bbox_to_anchor=(1.05,1.10))
    plt.tight_layout()
    plt.show()


def analisar_multivariado(df_total, variable):
    tasa = {
        "Ocupados": 1,
        "Desocupados": 2,
        "Inactivo": 3,
        "Menor de 10 años": 4
    }
    #LIMPIEZA DE DATOS

    df_total["ESTADO"] = pd.to_numeric(df_total["ESTADO"], errors="coerce") #pone todos nros de 'estado' a tipo numerico y el coerce es que si hay un error que lo deje pasar (si no puede cambiar algo a tipo numerico)
    df_total = df_total.dropna(subset=["ESTADO"]) #los que no puede pasar a tipo numerico borra todas las filas vacias que tengan el elemento 'estado' vacio
    df_total["ESTADO"] = df_total["ESTADO"].astype(int) #que pase todo lo de 'estado' a nro entero (int)

    #recorre toda la columna, encuentra la columna, pone todos sus datos en tipo numerico (cualquier tipo de numero) si no puede, el coerce le pone 'nan' y lo deja pasar
    # cuando termina, todos los tipos numericos, los pasa a un tipo de numero entero (int) para ser mas exacto 

    df_total["AGLOMERADO"] = pd.to_numeric(df_total["AGLOMERADO"], errors="coerce")
    df_total = df_total.dropna(subset=["AGLOMERADO"])
    df_total["AGLOMERADO"] = df_total["AGLOMERADO"].astype(int)

    #exactamente igual que el anterior, recorre, analisa y borra lo que sobre o haya espacio

    #CH04 sexo de la persona

    df_total["CH04"] = pd.to_numeric(df_total["CH04"], errors="coerce")
    df_total = df_total.dropna(subset=["CH04"])
    df_total["CH04"] = df_total["CH04"].astype(int)

    df_filtrado = df_total[df_total["AGLOMERADO"].isin([9, 10])]
    df_limpio = df_filtrado[df_filtrado["ESTADO"] > 0]
    df_final = df_limpio[["ESTADO", "ANO4", "TRIMESTRE", "CH04"]]

    df_estado_elegido = df_final[df_final["ESTADO"] == tasa[variable]]
    
    df_estado_elegido["ANO4-TRIMESTRE"] = df_estado_elegido["ANO4"].astype(str) + "-T" + df_estado_elegido["TRIMESTRE"].astype(str)
    
    label_sexo = {
        1: "Masculino",
        2: "Femenino"
    }
    df_estado_elegido["SEXOSTR"] = df_estado_elegido["CH04"].map(label_sexo)
    
    df_grouped = df_estado_elegido.groupby(["ANO4-TRIMESTRE", "SEXOSTR"]).agg(TOTAL=('ESTADO', 'count')).reset_index()
    pivot = df_grouped.pivot(index='ANO4-TRIMESTRE', columns='SEXOSTR', values='TOTAL')


    pivot.plot(kind='bar')
    plt.title(f"Total de {variable} por Año y Trimestre")
    plt.xlabel("Año")
    plt.xticks(rotation=45)
    plt.ylabel(f"Cantidad de {variable}")
    plt.ylim(0, 1800)
    plt.legend(title="Sexo", loc="upper right", bbox_to_anchor=(1.05,1.10))
    plt.tight_layout()
    plt.show()

def modelacion_regresion(df_total):
    
    #P47T monto de ingreso total individual
    #CH06 años de la persona
    #CH04 sexo de la persona

    df_total = df_total.copy()

    df_total["P47T"] = pd.to_numeric(df_total["P47T"], errors="coerce")
    df_total = df_total.dropna(subset=["P47T"])
    df_total["P47T"] = df_total["P47T"].astype(float)

    df_total["CH06"] = pd.to_numeric(df_total["CH06"], errors="coerce")
    df_total = df_total.dropna(subset=["CH06"])
    df_total["CH06"] = df_total["CH06"].astype(int)

    df_total["NIVEL_ED"] = pd.to_numeric(df_total["NIVEL_ED"], errors="coerce")
    df_total = df_total.dropna(subset=["NIVEL_ED"])
    df_total["NIVEL_ED"] = df_total["NIVEL_ED"].astype(int)

    #variables a analiar
    x = df_total[["CH06", "NIVEL_ED", "CH04"]] #variables independientes (lo que puedo manipular - en relacion a su nivel de educacion, los años de la persona o el sexo)
    y = df_total["P47T"] #variables dependiente (lo que quiero mostrar - cuanto dinero gana en relacion a los factores anteriores)

    #en relacion a sus caracteristicas como edad, nivel de educacion o sexo (en algunos casos puede influir )puede variar su ingreso individual hacia la persona
    #por eso las variables independientes son las que dan los parametros y con eso puedo saber y manipular datos en base a las variables dependientes
    #(que son las que DEPENDEN de los datos anteriores [variables independientes])

    #divisiion de porcentaje de entrenamiento y prueba (prueba y error de datos en relacion a los ingresos obtenidos y posible margen de error)
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3, random_state=42) #test size es de 0.3 (30% de prueba de los datos que esten de forma acertada)

    modelo = LinearRegression()
    modelo.fit(x_train, y_train)

    # Predicciones
    y_prediccion = modelo.predict(x_test)

    # Métricas
    metrica = mean_squared_error(y_test, y_prediccion)
    r2 = r2_score(y_test, y_prediccion)

    print(f"Error Cuadrático Medio (MSE): {metrica:.2f}")
    print(f"Coeficiente R²: {r2:.4f}")

    # Importancia de variables
    coeficiente_df_total = pd.DataFrame({
        "Variable": ["Edad", "Nivel educativo", "Sexo"],
        "Coeficiente": modelo.coef_
    })

    print("\nInfluencia de las variables:")
    print(coeficiente_df_total)








df_carga_de_informacion = cargar_datos()
#analisar_univariado(df_carga_de_informacion, "Ocupados")
#analisar_multivariado(df_carga_de_informacion, "Ocupados")
#modelacion_regresion(df_carga_de_informacion)

