import os
import pandas as pd

# ==========================
# Configuración
# ==========================

# Archivo de entrada (.csv o .xlsx)

# mama
# input_file = "./breast/BCNB_final.csv"
# input_file = "./breast/HISTAI_final_BC1.csv"
# input_file = "./breast/HISTAI_final_BC2.csv"
# input_file = "./breast/HSI_BRCA_final.xlsx"

# colorectal
# input_file = "./colorectal/HISTAI_B1_final_CRC1.csv"
# input_file = "./colorectal/HISTAI_B1_final_CRC2.csv"

# input_file = "./colorectal/HISTAI_B2_final_CRC1.csv"
# input_file = "./colorectal/HISTAI_B2_final_CRC2.csv"

# input_file = "./colorectal/surgen368_final_CRC1.csv"
# input_file = "./colorectal/surgen368_final_CRC2.csv"
input_file = "./colorectal/surgen1482_final_CRC1.csv"
# input_file = "./colorectal/surgen1482_final_CRC2.csv"

# Solo se utiliza si el archivo es Excel
sheet_name = "Sheet1"      # También puede ser 0, 1, 2, ...

# Columnas que quieres conservar
columns_to_keep = [
    "case_id",
    "age",
    "sex",
    "site_group_norm",
    # "side_norm",
]

# Carpeta donde guardar el nuevo CSV
output_folder = "./clean/colorectal/site/"

# Nombre del nuevo archivo
output_filename = "SURGEN1482_CRC_site.csv"

# ==========================
# Procesamiento
# ==========================

# Crear carpeta de salida si no existe
os.makedirs(output_folder, exist_ok=True)

# Detectar el tipo de archivo
extension = os.path.splitext(input_file)[1].lower()

if extension == ".csv":
    df = pd.read_csv(input_file)

elif extension in [".xlsx", ".xls"]:
    df = pd.read_excel(input_file, sheet_name=sheet_name)

else:
    raise ValueError(f"Formato '{extension}' no soportado.")

# Verificar que todas las columnas existan
missing_columns = [col for col in columns_to_keep if col not in df.columns]

if missing_columns:
    raise ValueError(
        f"Las siguientes columnas no existen en el archivo:\n{missing_columns}"
    )

# Mantener solo las columnas especificadas
new_df = df[columns_to_keep]

# Ruta final
output_path = os.path.join(output_folder, output_filename)

# Guardar CSV
new_df.to_csv(output_path, index=False)

print(f"CSV generado correctamente:")
print(output_path)
print(f"Filas: {len(new_df)}")
print(f"Columnas: {len(new_df.columns)}")