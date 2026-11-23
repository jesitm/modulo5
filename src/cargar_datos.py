import os
import pandas as pd
def cargarDatos():
    # Obtiene la ruta del directorio donde se encuentra este script (src/)
    script_dir = os.path.dirname(__file__)
    # Construye la ruta completa al archivo .xlsx
    file_path = os.path.join(script_dir, '../Base_de_datos.xlsx')
    # Lee el archivo de Excel usando la ruta completa
    df = pd.read_excel(file_path)
    return df

# --- BLOQUE DE PRUEBA ---
if __name__ == '__main__':
    try:
        datos = cargarDatos()
        print('¡Éxito! El archivo se cargó correctamente.')
        print(f'Total de filas y columnas: {datos.shape}')
        print('\nPrimeras 5 filas:')
        print(datos.head())
    except FileNotFoundError:
        print(
            "Error: No se encontró 'Base_de_datos.xlsx' en la misma carpeta que el script."
        )
    except Exception as e:
        print(f'Ocurrió un error inesperado: {e}')