from dataclasses import dataclass, field
from pathlib import Path
from collections import Counter
from typing import Any


@dataclass(kw_only=True)
class AnalysisConfig:

    # =============================
    # 1. Entrada / salida
    # =============================
    file_path: str
    output_dir: str = "outputs"
    dataset_name: str | None = None

    file_type: str = "csv"          # csv | xlsx
    sheet_name: str | int | None = 0

    # =============================
    # 2. Lectura del archivo
    # =============================
    encoding: str | None = None
    separator: str = ","
    decimal: str = "."
    read_na_default: bool = True

    na_values: list[Any] = field(default_factory=lambda: [
        "NA", "N/A", "na", "n/a", "null", "NULL", "None", "none", "", " "
    ])
    true_values: list[Any] = field(default_factory=lambda: ["True", "true", "TRUE", "Yes", "YES", "1"])
    false_values: list[Any] = field(default_factory=lambda: ["False", "false", "FALSE", "No", "NO", "0"])

    # =============================
    # 3. Selección de columnas
    # =============================
    include_columns: list[str] = field(default_factory=list)
    exclude_columns: list[str] = field(default_factory=list)

    id_columns: list[str] = field(default_factory=list)
    text_columns: list[str] = field(default_factory=list)
    ignore_columns: list[str] = field(default_factory=list)

    # =============================
    # 4. Nombres de columnas
    # =============================
    rename_columns: dict[str, str] = field(default_factory=dict)
    strip_column_names: bool = True
    lowercase_column_names: bool = False
    replace_spaces_in_column_names: bool = False
    column_name_separator: str = "_"

    # =============================
    # 5. Tipos forzados
    # =============================
    force_numeric: list[str] = field(default_factory=list)
    force_categorical: list[str] = field(default_factory=list)
    force_datetime: list[str] = field(default_factory=list)
    force_string: list[str] = field(default_factory=list)
    force_boolean: list[str] = field(default_factory=list)

    # =============================
    # 6. Fechas
    # =============================
    parse_dates: list[str] = field(default_factory=list)
    dayfirst: bool = False
    datetime_format: str | None = None

    # =============================
    # 7. Limpieza de texto
    # =============================
    normalize_text: bool = True
    strip_text: bool = True
    collapse_internal_spaces: bool = True
    lowercase_text: bool = False

    # =============================
    # 8. Reemplazos / recodificación
    # =============================
    global_value_mapping: dict[Any, Any] = field(default_factory=dict)
    column_value_mapping: dict[str, dict[Any, Any]] = field(default_factory=dict)
    custom_missing_by_column: dict[str, list[Any]] = field(default_factory=dict)

    # =============================
    # 9. Variables derivadas genéricas
    # =============================
    derived_columns: dict[str, str] = field(default_factory=dict)

    # =============================
    # 10. Detección automática de tipos
    # =============================
    numeric_ratio_threshold: float = 0.80
    datetime_ratio_threshold: float = 0.80
    categorical_max_unique: int | None = 30

    # =============================
    # 11. Resúmenes y análisis
    # =============================
    analyze_numeric: bool = True
    analyze_categorical: bool = True
    analyze_datetime: bool = True
    analyze_missingness: bool = True

    include_missing_in_frequency: bool = True
    include_missing_by_column: dict[str, bool] = field(default_factory=dict)
    
    missing_label: str = "MISSING"
    missing_label_by_column: dict[str, str] = field(default_factory=dict)

    quantiles: list[float] = field(default_factory=lambda: [0.25, 0.50, 0.75])

    min_frequency: int = 1
    max_categories_table: int | None = None
    max_categories_plot: int = 5

    merge_rare_categories: bool = False
    rare_category_threshold: float | None = None   # porcentaje
    rare_category_label: str = "OTHER"

    detect_outliers_iqr: bool = True
    outlier_iqr_multiplier: float = 1.5

    # =============================
    # 12. Visualización
    # =============================
    save_plots: bool = True
    show_plots: bool = False

    plot_format: str = "png"
    plot_dpi: int = 200
    figure_width: int = 8
    figure_height: int = 4

    style: str = "whitegrid"
    palette: str = "deep"
    rotate_xticks: int = 45

    categorical_plot_type: str = "bar"   # bar | barh | lollipop
    categorical_plot_type_by_column: dict[str, str] = field(default_factory=dict)
    column_title_map: dict[str, str] = field(default_factory=dict)

    # =============================
    # 13. Exportación
    # =============================
    save_cleaned_data: bool = False
    save_reports: bool = True
    export_format: str = "csv"      # csv | xlsx

    # =============================
    # 14. Validación
    # =============================
    def __post_init__(self):
        if self.dataset_name is None:
            self.dataset_name = Path(self.file_path).stem

        if self.file_type not in {"csv", "xlsx"}:
            raise ValueError("file_type debe ser 'csv' o 'xlsx'.")

        if self.export_format not in {"csv", "xlsx"}:
            raise ValueError("export_format debe ser 'csv' o 'xlsx'.")

        if not (0 <= self.numeric_ratio_threshold <= 1):
            raise ValueError("numeric_ratio_threshold debe estar entre 0 y 1.")

        if not (0 <= self.datetime_ratio_threshold <= 1):
            raise ValueError("datetime_ratio_threshold debe estar entre 0 y 1.")

        if self.rare_category_threshold is not None and self.rare_category_threshold < 0:
            raise ValueError("rare_category_threshold no puede ser negativo.")

        overlap = set(self.include_columns) & set(self.exclude_columns)
        if overlap:
            raise ValueError(
                f"Estas columnas están en include_columns y exclude_columns a la vez: {sorted(overlap)}"
            )

        # tipos de graficos
        valid_plot_types = {"bar", "barh", "lollipop"}

        if self.categorical_plot_type not in valid_plot_types:
            raise ValueError("categorical_plot_type debe ser 'bar', 'barh' o 'lollipop'.")

        invalid = {
            col: plot_type
            for col, plot_type in self.categorical_plot_type_by_column.items()
            if plot_type not in valid_plot_types
        }
        if invalid:
            raise ValueError(
                f"Tipos de gráfico categórico no soportados por columna: {invalid}"
            )

        forced_cols = (
            self.force_numeric
            + self.force_categorical
            + self.force_datetime
            + self.force_string
            + self.force_boolean
        )
        counts = Counter(forced_cols)
        duplicated = [col for col, n in counts.items() if n > 1]
        if duplicated:
            raise ValueError(f"Hay columnas forzadas a más de un tipo: {sorted(duplicated)}")
        


"""
    1. Entrada / salida

        file_path: ruta del archivo que vas a analizar.

        output_dir: carpeta donde se guardarán reportes y gráficos.

        dataset_name: nombre lógico del dataset; si no lo das, se infiere desde el nombre del archivo en __post_init__.

        file_type: tipo de archivo de entrada, por ejemplo CSV o XLSX.

        sheet_name: hoja a leer si el archivo es Excel.

    Esto agrupa la información mínima para ubicar el archivo y decidir cómo leerlo y cómo nombrar sus salidas.
    2. Lectura del archivo

        encoding: codificación del archivo, útil cuando hay problemas con tildes o caracteres especiales.

        separator: separador del CSV, por ejemplo , o ;.

        decimal: símbolo decimal, útil si el archivo usa , en vez de ..

        read_na_default: permite usar el comportamiento por defecto de pandas para detectar faltantes.

        na_values: lista de valores que quieres tratar como missing.

        true_values y false_values: textos que deben interpretarse como booleanos.

    Este bloque hace que la lectura sea flexible sin reescribir el código del analizador cada vez.
    3. Selección de columnas

        include_columns: si la llenas, el análisis se restringe a estas columnas.

        exclude_columns: columnas que no quieres analizar.

        id_columns: columnas identificadoras, como IDs o códigos únicos.

        text_columns: columnas de texto libre que no quieres tratar como categóricas normales.

        ignore_columns: columnas que directamente quieres saltarte por completo.

    La idea aquí es separar columnas “analizables” de columnas administrativas o problemáticas.
    4. Nombres de columnas

        rename_columns: diccionario para renombrar columnas.

        strip_column_names: elimina espacios al inicio o final del nombre.

        lowercase_column_names: pasa nombres a minúscula.

        replace_spaces_in_column_names: reemplaza espacios por otro separador.

        column_name_separator: símbolo que usarás, por ejemplo _.

    Esto ayuda a estandarizar datasets distintos para que luego se procesen con la misma lógica.
    5. Tipos forzados

        force_numeric: columnas que quieres interpretar sí o sí como numéricas.

        force_categorical: columnas que quieres tratar como categóricas.

        force_datetime: columnas que deben parsearse como fecha.

        force_string: columnas que quieres mantener como texto.

        force_boolean: columnas que deben tratarse como booleanas.

    Esto es útil porque la detección automática de tipos no siempre acierta, especialmente cuando los datos vienen sucios o mezclados.
    6. Fechas

        parse_dates: columnas que quieres intentar convertir a fecha.

        dayfirst: define si el parser debe interpretar primero el día.

        datetime_format: formato explícito de fecha si quieres controlar el parseo.

    Este bloque evita depender solo de la inferencia automática cuando los formatos son ambiguos.
    7. Limpieza de texto

        normalize_text: activa o desactiva la limpieza general de texto.

        strip_text: elimina espacios sobrantes.

        collapse_internal_spaces: convierte múltiples espacios internos en uno solo.

        lowercase_text: pasa el texto a minúscula.

    Sirve para reducir ruido en variables de texto o categorías antes de calcular frecuencias.
    8. Reemplazos / recodificación

        global_value_mapping: reemplazos aplicados a todo el dataset.

        column_value_mapping: reemplazos específicos por columna.

        custom_missing_by_column: valores que en ciertas columnas deben convertirse en missing.

    Este bloque te da control fino sobre recodificaciones sin volver a tocar el código del analizador.
    9. Variables derivadas genéricas

        derived_columns: espacio para declarar nuevas columnas derivadas.

    Aquí lo dejé general a propósito, sin asumir edad, BMI, estadio ni nada del dominio clínico.
    10. Detección automática de tipos

        numeric_ratio_threshold: proporción mínima de valores convertibles a número para considerar una columna como numérica.

        datetime_ratio_threshold: proporción mínima de valores convertibles a fecha para considerarla datetime.

        categorical_max_unique: máximo de valores únicos para que una columna pueda tratarse como categórica en vez de texto libre.

    Esto permite heurísticas básicas para clasificar columnas cuando no las fuerzas manualmente.
    11. Resúmenes y análisis

        analyze_numeric, analyze_categorical, analyze_datetime, analyze_missingness: activan o desactivan cada tipo de análisis.

        include_missing_in_frequency: indica si los missing deben aparecer en las tablas de frecuencia.

        missing_label: etiqueta a usar cuando incluyes missing en frecuencias.

        quantiles: percentiles para resúmenes numéricos.

        min_frequency: frecuencia mínima para mostrar una categoría.

        max_categories_table: límite de categorías en tablas.

        max_categories_plot: límite de categorías en gráficos.

        merge_rare_categories: activa agrupación de categorías raras.

        rare_category_threshold: umbral porcentual para considerar una categoría como rara.

        rare_category_label: etiqueta para agrupar categorías raras.

        detect_outliers_iqr: activa conteo de outliers por regla IQR.

        outlier_iqr_multiplier: multiplicador del IQR.

    Este es el bloque que controla qué análisis quieres y con qué reglas resumir las variables.
    12. Visualización

        save_plots: guarda gráficos en disco.

        show_plots: muestra gráficos en pantalla.

        plot_format: formato de imagen.

        plot_dpi: resolución.

        figure_width y figure_height: tamaño de figura.

        style: estilo general de seaborn.

        palette: paleta de color.

        rotate_xticks: rotación de etiquetas del eje X.

    Esto separa la lógica analítica de la estética y exportación visual.
    13. Exportación

        save_cleaned_data: guarda el dataset ya procesado.

        save_reports: guarda tablas resumen.

        export_format: formato de salida.

    Aquí defines qué artefactos quieres producir al final del análisis.
    14. Validación

    El método __post_init__ se ejecuta automáticamente justo después de crear la instancia de la dataclass, y es el lugar recomendado para validar argumentos o completar campos derivados como dataset_name.
 """