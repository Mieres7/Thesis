# from __future__ import annotations

# from pathlib import Path
# import re

# import numpy as np
# import pandas as pd
# import matplotlib.pyplot as plt
# import seaborn as sns


# class DatasetAnalyzer:
#     def __init__(self, config):
#         self.config = config
#         self.file_path = Path(config.file_path)
#         self.output_dir = Path(config.output_dir) / config.dataset_name
#         self.output_dir.mkdir(parents=True, exist_ok=True)

#         self.df_raw = None
#         self.df = None

#         self.numeric_cols = []
#         self.categorical_cols = []
#         self.datetime_cols = []
#         self.string_cols = []
#         self.boolean_cols = []

#     # =====================
#     # Flujo principal
#     # =====================
#     def run(self):
#         self.load_data()
#         self.standardize_column_names()
#         self.apply_column_renaming()
#         self.apply_column_selection()
#         self.normalize_missing_values()
#         self.normalize_text_columns()
#         self.apply_value_mapping()
#         self.parse_datetime_columns()
#         self.force_declared_types()
#         self.create_derived_columns()
#         self.detect_column_types()

#         reports = self.generate_reports()

#         if self.config.save_plots or self.config.show_plots:
#             self.generate_plots()

#         if self.config.save_cleaned_data:
#             self.save_cleaned_data()

#         return reports

#     # =====================
#     # Lectura
#     # =====================
#     def load_data(self):
#         if self.config.file_type.lower() == "csv":
#             self.df_raw = pd.read_csv(
#                 self.file_path,
#                 sep=self.config.separator,
#                 decimal=self.config.decimal,
#                 encoding=self.config.encoding,
#                 keep_default_na=self.config.read_na_default,
#                 na_values=self.config.na_values,
#                 true_values=self.config.true_values,
#                 false_values=self.config.false_values
#             )
#         elif self.config.file_type.lower() == "xlsx":
#             self.df_raw = pd.read_excel(
#                 self.file_path,
#                 sheet_name=self.config.sheet_name
#             )
#         else:
#             raise ValueError(f"file_type no soportado: {self.config.file_type}")

#         self.df = self.df_raw.copy()
#         return self.df

#     # =====================
#     # Columnas
#     # =====================
#     def standardize_column_names(self):
#         cols = self.df.columns.astype(str)

#         if self.config.strip_column_names:
#             cols = cols.str.strip()

#         if self.config.lowercase_column_names:
#             cols = cols.str.lower()

#         if self.config.replace_spaces_in_column_names:
#             cols = cols.str.replace(r"\s+", self.config.column_name_separator, regex=True)

#         self.df.columns = cols
#         return self.df

#     def apply_column_renaming(self):
#         if self.config.rename_columns:
#             self.df = self.df.rename(columns=self.config.rename_columns)
#         return self.df

#     def apply_column_selection(self):
#         if self.config.include_columns:
#             keep_cols = [c for c in self.config.include_columns if c in self.df.columns]
#             self.df = self.df[keep_cols].copy()

#         excluded = set(
#             self.config.exclude_columns
#             + self.config.id_columns
#             + self.config.ignore_columns
#         )
#         keep_cols = [c for c in self.df.columns if c not in excluded]
#         self.df = self.df[keep_cols].copy()
#         return self.df

#     # =====================
#     # Missing values
#     # =====================
#     def normalize_missing_values(self):
#         for col, missing_values in self.config.custom_missing_by_column.items():
#             if col in self.df.columns:
#                 self.df[col] = self.df[col].replace(missing_values, np.nan)
#         return self.df

#     # =====================
#     # Limpieza de texto
#     # =====================
#     def _normalize_text_value(self, value):
#         if pd.isna(value):
#             return value

#         value = str(value)

#         if self.config.strip_text:
#             value = value.strip()

#         if self.config.collapse_internal_spaces:
#             value = re.sub(r"\s+", " ", value)

#         if self.config.lowercase_text:
#             value = value.lower()

#         return value

#     def normalize_text_columns(self):
#         if not self.config.normalize_text:
#             return self.df

#         object_cols = self.df.select_dtypes(include=["object", "string"]).columns

#         for col in object_cols:
#             self.df[col] = self.df[col].map(self._normalize_text_value)

#         return self.df

#     # =====================
#     # Reemplazos / recodificación
#     # =====================
#     def apply_value_mapping(self):
#         if self.config.global_value_mapping:
#             self.df = self.df.replace(self.config.global_value_mapping)

#         for col, mapping in self.config.column_value_mapping.items():
#             if col in self.df.columns:
#                 self.df[col] = self.df[col].replace(mapping)

#         return self.df

#     # =====================
#     # Tipos
#     # =====================
#     def parse_datetime_columns(self):
#         date_cols = set(self.config.parse_dates + self.config.force_datetime)

#         for col in date_cols:
#             if col in self.df.columns:
#                 self.df[col] = pd.to_datetime(
#                     self.df[col],
#                     errors="coerce",
#                     dayfirst=self.config.dayfirst,
#                     format=self.config.datetime_format
#                 )
#         return self.df

#     def _coerce_to_boolean(self, series):
#         if pd.api.types.is_bool_dtype(series):
#             return series.astype("boolean")

#         mapping = {}
#         for value in self.config.true_values:
#             mapping[str(value).strip().lower()] = True
#         for value in self.config.false_values:
#             mapping[str(value).strip().lower()] = False

#         normalized = series.astype("string").str.strip().str.lower()
#         return normalized.map(mapping).astype("boolean")

#     def force_declared_types(self):
#         for col in self.config.force_numeric:
#             if col in self.df.columns:
#                 self.df[col] = pd.to_numeric(self.df[col], errors="coerce")

#         for col in self.config.force_string:
#             if col in self.df.columns:
#                 self.df[col] = self.df[col].astype("string")

#         for col in self.config.force_categorical:
#             if col in self.df.columns:
#                 self.df[col] = self.df[col].astype("category")

#         for col in self.config.force_boolean:
#             if col in self.df.columns:
#                 self.df[col] = self._coerce_to_boolean(self.df[col])

#         return self.df

#     def create_derived_columns(self):
#         for new_col, rule in self.config.derived_columns.items():
#             if callable(rule):
#                 self.df[new_col] = rule(self.df)
#             elif isinstance(rule, str):
#                 self.df[new_col] = self.df.eval(rule)
#             else:
#                 self.df[new_col] = rule

#         return self.df

#     def detect_column_types(self):
#         self.numeric_cols = []
#         self.categorical_cols = []
#         self.datetime_cols = []
#         self.string_cols = []
#         self.boolean_cols = []

#         for col in self.df.columns:
#             if col in self.config.text_columns:
#                 self.string_cols.append(col)
#                 continue

#             if col in self.config.force_numeric:
#                 self.numeric_cols.append(col)
#                 continue

#             if col in self.config.force_categorical:
#                 self.categorical_cols.append(col)
#                 continue

#             if col in self.config.force_datetime:
#                 self.datetime_cols.append(col)
#                 continue

#             if col in self.config.force_string:
#                 self.string_cols.append(col)
#                 continue

#             if col in self.config.force_boolean:
#                 self.boolean_cols.append(col)
#                 continue

#             series = self.df[col]

#             if pd.api.types.is_bool_dtype(series):
#                 self.boolean_cols.append(col)
#                 continue

#             if pd.api.types.is_datetime64_any_dtype(series):
#                 self.datetime_cols.append(col)
#                 continue

#             numeric_try = pd.to_numeric(series, errors="coerce")
#             numeric_ratio = numeric_try.notna().mean()

#             datetime_try = pd.to_datetime(
#                 series,
#                 errors="coerce",
#                 dayfirst=self.config.dayfirst,
#                 format=self.config.datetime_format
#             )
#             datetime_ratio = datetime_try.notna().mean()

#             n_unique = series.nunique(dropna=True)

#             if numeric_ratio >= self.config.numeric_ratio_threshold:
#                 self.numeric_cols.append(col)
#             elif datetime_ratio >= self.config.datetime_ratio_threshold:
#                 self.datetime_cols.append(col)
#             else:
#                 if self.config.categorical_max_unique is not None and n_unique > self.config.categorical_max_unique:
#                     self.string_cols.append(col)
#                 else:
#                     self.categorical_cols.append(col)

#         return {
#             "numeric": self.numeric_cols,
#             "categorical": self.categorical_cols,
#             "datetime": self.datetime_cols,
#             "string": self.string_cols,
#             "boolean": self.boolean_cols
#         }

#     # =====================
#     # Helpers
#     # =====================
#     def _is_categorical(self, series):
#         return isinstance(series.dtype, pd.CategoricalDtype)

#     def _should_include_missing(self, col_name: str) -> bool:
#         return self.config.include_missing_by_column.get(
#             col_name,
#             self.config.include_missing_in_frequency
#         )

#     def _get_missing_label(self, col_name: str) -> str:
#         return self.config.missing_label_by_column.get(
#             col_name,
#             self.config.missing_label
#         )

#     def _fill_missing_for_display(self, series):
#         include_missing = self._should_include_missing(series.name)

#         if not include_missing:
#             return series

#         label = self._get_missing_label(series.name)

#         if self._is_categorical(series):
#             if label not in series.cat.categories:
#                 series = series.cat.add_categories([label])
#             return series.fillna(label)

#         return series.fillna(label)


#     def _merge_rare_categories(self, series):
#         if not self.config.merge_rare_categories or self.config.rare_category_threshold is None:
#             return series

#         freq_pct = series.value_counts(normalize=True, dropna=False) * 100
#         rare_values = freq_pct[freq_pct < self.config.rare_category_threshold].index

#         if len(rare_values) == 0:
#             return series

#         label = self.config.rare_category_label

#         if self._is_categorical(series) and label not in series.cat.categories:
#             series = series.cat.add_categories([label])

#         return series.replace(list(rare_values), label)

#     def _detected_type_for_column(self, col):
#         if col in self.numeric_cols:
#             return "numeric"
#         if col in self.categorical_cols:
#             return "categorical"
#         if col in self.datetime_cols:
#             return "datetime"
#         if col in self.string_cols:
#             return "string"
#         if col in self.boolean_cols:
#             return "boolean"
#         return "unknown"

#     # =====================
#     # Reportes
#     # =====================
#     def dataset_summary(self):
#         summary = {
#             "dataset_name": self.config.dataset_name,
#             "file_path": str(self.file_path),
#             "n_rows": self.df.shape[0],
#             "n_columns": self.df.shape[1],
#             "numeric_columns": len(self.numeric_cols),
#             "categorical_columns": len(self.categorical_cols),
#             "datetime_columns": len(self.datetime_cols),
#             "string_columns": len(self.string_cols),
#             "boolean_columns": len(self.boolean_cols),
#             "total_missing_cells": int(self.df.isna().sum().sum()),
#             "missing_cells_pct": round((self.df.isna().sum().sum() / self.df.size) * 100, 2) if self.df.size else 0
#         }
#         return pd.DataFrame([summary])

#     def column_summary(self):
#         rows = []
#         for col in self.df.columns:
#             rows.append({
#                 "variable": col,
#                 "pandas_dtype": str(self.df[col].dtype),
#                 "detected_type": self._detected_type_for_column(col),
#                 "missing_n": int(self.df[col].isna().sum()),
#                 "missing_pct": round(float(self.df[col].isna().mean() * 100), 2),
#                 "n_unique": int(self.df[col].nunique(dropna=True))
#             })
#         return pd.DataFrame(rows)

#     def missing_summary(self):
#         if not self.config.analyze_missingness:
#             return pd.DataFrame()

#         return pd.DataFrame({
#             "variable": self.df.columns,
#             "dtype": self.df.dtypes.astype(str).values,
#             "missing_n": self.df.isna().sum().values,
#             "missing_pct": (self.df.isna().mean() * 100).round(2).values,
#             "n_unique": self.df.nunique(dropna=True).values
#         }).sort_values(["missing_pct", "n_unique"], ascending=[False, True])

#     def numeric_summary(self):
#         if not self.numeric_cols or not self.config.analyze_numeric:
#             return pd.DataFrame()

#         df_num = self.df[self.numeric_cols].apply(pd.to_numeric, errors="coerce")
#         summary = df_num.describe(percentiles=self.config.quantiles).T

#         summary["median"] = df_num.median()
#         summary["missing_n"] = df_num.isna().sum()
#         summary["missing_pct"] = (df_num.isna().mean() * 100).round(2)
#         summary["iqr"] = df_num.quantile(0.75) - df_num.quantile(0.25)

#         if self.config.detect_outliers_iqr:
#             outlier_counts = {}
#             for col in df_num.columns:
#                 q1 = df_num[col].quantile(0.25)
#                 q3 = df_num[col].quantile(0.75)
#                 iqr = q3 - q1
#                 lower = q1 - self.config.outlier_iqr_multiplier * iqr
#                 upper = q3 + self.config.outlier_iqr_multiplier * iqr
#                 outlier_counts[col] = df_num[col].between(lower, upper, inclusive="both").eq(False).sum()

#             summary["outlier_n_iqr"] = pd.Series(outlier_counts)

#         summary = summary.reset_index().rename(columns={"index": "variable"})
#         return summary

#     def categorical_summary(self):
#         cols_to_summarize = self.categorical_cols + self.boolean_cols

#         if not cols_to_summarize or not self.config.analyze_categorical:
#             return pd.DataFrame()

#         rows = []

#         for col in cols_to_summarize:
#             series = self.df[col].copy()
#             include_missing = self._should_include_missing(col)

#             if include_missing:
#                 series = self._fill_missing_for_display(series)

#             series = self._merge_rare_categories(series)

#             freq = series.value_counts(dropna=not include_missing)
#             pct = series.value_counts(dropna=not include_missing, normalize=True) * 100

#             if self.config.max_categories_table is not None:
#                 freq = freq.head(self.config.max_categories_table)

#             for value, count in freq.items():
#                 if count < self.config.min_frequency:
#                     continue

#                 rows.append({
#                     "variable": col,
#                     "value": value,
#                     "count": int(count),
#                     "percent": round(float(pct.loc[value]), 2)
#                 })

#         return pd.DataFrame(rows)

#     def datetime_summary(self):
#         if not self.datetime_cols or not self.config.analyze_datetime:
#             return pd.DataFrame()

#         rows = []
#         for col in self.datetime_cols:
#             s = pd.to_datetime(self.df[col], errors="coerce")
#             rows.append({
#                 "variable": col,
#                 "missing_n": int(s.isna().sum()),
#                 "missing_pct": round(float(s.isna().mean() * 100), 2),
#                 "min_date": s.min(),
#                 "max_date": s.max(),
#                 "n_unique": int(s.nunique(dropna=True))
#             })

#         return pd.DataFrame(rows)

#     def generate_reports(self):
#         reports = {
#             "dataset_summary": self.dataset_summary(),
#             "column_summary": self.column_summary(),
#             "missing_summary": self.missing_summary(),
#             "numeric_summary": self.numeric_summary(),
#             "categorical_summary": self.categorical_summary(),
#             "datetime_summary": self.datetime_summary(),
#         }

#         if self.config.save_reports:
#             self._save_report_dict(reports)

#         return reports

#     def _save_report_dict(self, reports: dict):
#         for report_name, df_report in reports.items():
#             if self.config.export_format == "csv":
#                 df_report.to_csv(self.output_dir / f"{report_name}.csv", index=False)
#             elif self.config.export_format == "xlsx":
#                 df_report.to_excel(self.output_dir / f"{report_name}.xlsx", index=False)
#             else:
#                 raise ValueError(f"Formato de exportación no soportado: {self.config.export_format}")

#     # =====================
#     # Gráficos
#     # =====================
#     def generate_plots(self):
#         sns.set_style(self.config.style)
#         self._plot_numeric_distributions()
#         self._plot_categorical_distributions()

#     def _plot_numeric_distributions(self):
#         if not self.config.analyze_numeric:
#             return

#         plot_dir = self.output_dir / "plots_numeric"
#         plot_dir.mkdir(parents=True, exist_ok=True)

#         palette = sns.color_palette(self.config.palette)

#         for col in self.numeric_cols:
#             s = pd.to_numeric(self.df[col], errors="coerce").dropna()
#             if s.empty:
#                 continue

#             fig, axes = plt.subplots(
#                 1, 2,
#                 figsize=(self.config.figure_width * 2, self.config.figure_height)
#             )

#             sns.histplot(s, kde=True, ax=axes[0], color=palette[0])
#             axes[0].set_title(f"Histograma - {col}")

#             sns.boxplot(x=s, ax=axes[1], color=palette[1 % len(palette)])
#             axes[1].set_title(f"Boxplot - {col}")

#             plt.tight_layout()

#             if self.config.save_plots:
#                 fig.savefig(
#                     plot_dir / f"{col}.{self.config.plot_format}",
#                     dpi=self.config.plot_dpi,
#                     bbox_inches="tight"
#                 )
#             if self.config.show_plots:
#                 plt.show()

#             plt.close(fig)


#     def _plot_categorical_distributions(self):
#         if not self.config.analyze_categorical:
#             return

#         plot_dir = self.output_dir / "plots_categorical"
#         plot_dir.mkdir(parents=True, exist_ok=True)

#         palette = sns.color_palette(self.config.palette)

#         for col in self.categorical_cols + self.boolean_cols:
#             series = self.df[col].copy()
#             include_missing = self._should_include_missing(col)

#             if include_missing:
#                 series = self._fill_missing_for_display(series)

#             series = self._merge_rare_categories(series)

#             freq = series.value_counts(dropna=not include_missing).head(
#                 self.config.max_categories_plot
#             )

#             if freq.empty:
#                 continue

#             plot_type = self._get_categorical_plot_type(col)

#             if plot_type == "bar":
#                 fig, ax = plt.subplots(
#                     figsize=(self.config.figure_width, self.config.figure_height)
#                 )

#                 sns.barplot(
#                     x=freq.index.astype(str),
#                     y=freq.values,
#                     ax=ax,
#                     hue=freq.index.astype(str),
#                     palette=self.config.palette,
#                     legend=False
#                 )
#                 ax.set_title(f"Distribución - {col}")
#                 ax.set_ylabel("Frecuencia")
#                 ax.set_xlabel("")
#                 ax.tick_params(axis="x", rotation=self.config.rotate_xticks)

#             elif plot_type == "barh":
#                 fig_height = max(self.config.figure_height, 0.45 * len(freq))
#                 fig, ax = plt.subplots(
#                     figsize=(self.config.figure_width, fig_height)
#                 )

#                 ordered = freq.sort_values(ascending=True)

#                 sns.barplot(
#                     x=ordered.values,
#                     y=ordered.index.astype(str),
#                     ax=ax,
#                     hue=ordered.index.astype(str),
#                     palette=self.config.palette,
#                     legend=False,
#                     orient="h"
#                 )
#                 ax.set_title(f"Distribución - {col}")
#                 ax.set_xlabel("Frecuencia")
#                 ax.set_ylabel("")

#             elif plot_type == "lollipop":
#                 fig_height = max(self.config.figure_height, 0.45 * len(freq))
#                 fig, ax = plt.subplots(
#                     figsize=(self.config.figure_width, fig_height)
#                 )

#                 ordered = freq.sort_values(ascending=True)
#                 y_pos = np.arange(len(ordered))

#                 ax.hlines(
#                     y=y_pos,
#                     xmin=0,
#                     xmax=ordered.values,
#                     color="gray",
#                     alpha=0.6,
#                     linewidth=1.5
#                 )
#                 ax.plot(
#                     ordered.values,
#                     y_pos,
#                     "o",
#                     color=palette[0],
#                     markersize=7
#                 )

#                 ax.set_yticks(y_pos)
#                 ax.set_yticklabels(ordered.index.astype(str))
#                 ax.set_xlabel("Frecuencia")
#                 ax.set_ylabel("")
#                 ax.set_title(f"Distribución - {col}")
#                 ax.grid(axis="x", alpha=0.3)

#             else:
#                 raise ValueError(f"Tipo de gráfico categórico no soportado: {plot_type}")

#             plt.tight_layout()

#             if self.config.save_plots:
#                 fig.savefig(
#                     plot_dir / f"{col}.{self.config.plot_format}",
#                     dpi=self.config.plot_dpi,
#                     bbox_inches="tight"
#                 )
#             if self.config.show_plots:
#                 plt.show()

#             plt.close(fig)

#     def _get_categorical_plot_type(self, col: str) -> str:
#         return self.config.categorical_plot_type_by_column.get(
#             col,
#             self.config.categorical_plot_type
#         )

#     # =====================
#     # Guardado extra
#     # =====================
#     def save_cleaned_data(self):
#         if self.config.export_format == "csv":
#             self.df.to_csv(self.output_dir / "cleaned_dataset.csv", index=False)
#         elif self.config.export_format == "xlsx":
#             self.df.to_excel(self.output_dir / "cleaned_dataset.xlsx", index=False)

#     # =====================
#     # Utilidades
#     # =====================
#     def get_processed_data(self):
#         return self.df.copy()

#     def get_raw_data(self):
#         return self.df_raw.copy()

#     def print_type_summary(self):
#         print("Numéricas:", self.numeric_cols)
#         print("Categóricas:", self.categorical_cols)
#         print("Fechas:", self.datetime_cols)
#         print("Texto libre:", self.string_cols)
#         print("Booleanas:", self.boolean_cols)

from __future__ import annotations

from pathlib import Path
import re
import os

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

class DatasetAnalyzer:
    def __init__(self, config):
        self.config = config
        self.file_path = Path(config.file_path)

        base_dir = os.environ.get("IMAGES_DRIVE") or config.output_dir
        self.output_dir = Path(base_dir) / config.dataset_name
        self.output_dir.mkdir(parents=True, exist_ok=True)

        self.df_raw = None
        self.df = None

        self.numeric_cols = []
        self.categorical_cols = []
        self.datetime_cols = []
        self.string_cols = []
        self.boolean_cols = []

    # =====================
    # Flujo principal
    # =====================
    def run(self):
        self.load_data()
        self.standardize_column_names()
        self.apply_column_renaming()
        self.apply_column_selection()
        self.normalize_missing_values()
        self.normalize_text_columns()
        self.apply_value_mapping()
        self.parse_datetime_columns()
        self.force_declared_types()
        self.create_derived_columns()
        self.detect_column_types()

        reports = self.generate_reports()

        if self.config.save_plots or self.config.show_plots:
            self.generate_plots()

        if self.config.save_cleaned_data:
            self.save_cleaned_data()

        return reports

    # =====================
    # Lectura
    # =====================
    def load_data(self):
        if self.config.file_type.lower() == "csv":
            self.df_raw = pd.read_csv(
                self.file_path,
                sep=self.config.separator,
                decimal=self.config.decimal,
                encoding=self.config.encoding,
                keep_default_na=self.config.read_na_default,
                na_values=self.config.na_values,
                true_values=self.config.true_values,
                false_values=self.config.false_values
            )
        elif self.config.file_type.lower() == "xlsx":
            self.df_raw = pd.read_excel(
                self.file_path,
                sheet_name=self.config.sheet_name
            )
        else:
            raise ValueError(f"file_type no soportado: {self.config.file_type}")

        self.df = self.df_raw.copy()
        return self.df

    # =====================
    # Columnas
    # =====================
    def standardize_column_names(self):
        cols = self.df.columns.astype(str)

        if self.config.strip_column_names:
            cols = cols.str.strip()

        if self.config.lowercase_column_names:
            cols = cols.str.lower()

        if self.config.replace_spaces_in_column_names:
            cols = cols.str.replace(r"\s+", self.config.column_name_separator, regex=True)

        self.df.columns = cols
        return self.df

    def apply_column_renaming(self):
        if self.config.rename_columns:
            self.df = self.df.rename(columns=self.config.rename_columns)
        return self.df

    def apply_column_selection(self):
        if self.config.include_columns:
            keep_cols = [c for c in self.config.include_columns if c in self.df.columns]
            self.df = self.df[keep_cols].copy()

        excluded = set(
            self.config.exclude_columns
            + self.config.id_columns
            + self.config.ignore_columns
        )
        keep_cols = [c for c in self.df.columns if c not in excluded]
        self.df = self.df[keep_cols].copy()
        return self.df

    # =====================
    # Missing values
    # =====================
    def normalize_missing_values(self):
        for col, missing_values in self.config.custom_missing_by_column.items():
            if col in self.df.columns:
                self.df[col] = self.df[col].replace(missing_values, np.nan)
        return self.df

    # =====================
    # Limpieza de texto
    # =====================
    def _normalize_text_value(self, value):
        if pd.isna(value):
            return value

        value = str(value)

        if self.config.strip_text:
            value = value.strip()

        if self.config.collapse_internal_spaces:
            value = re.sub(r"\s+", " ", value)

        if self.config.lowercase_text:
            value = value.lower()

        return value

    def normalize_text_columns(self):
        if not self.config.normalize_text:
            return self.df

        object_cols = self.df.select_dtypes(include=["object", "string"]).columns

        for col in object_cols:
            self.df[col] = self.df[col].map(self._normalize_text_value)

        return self.df

    # =====================
    # Reemplazos / recodificación
    # =====================
    def apply_value_mapping(self):
        if self.config.global_value_mapping:
            self.df = self.df.replace(self.config.global_value_mapping)

        for col, mapping in self.config.column_value_mapping.items():
            if col in self.df.columns:
                self.df[col] = self.df[col].replace(mapping)

        return self.df

    # =====================
    # Tipos
    # =====================
    def parse_datetime_columns(self):
        date_cols = set(self.config.parse_dates + self.config.force_datetime)

        for col in date_cols:
            if col in self.df.columns:
                self.df[col] = pd.to_datetime(
                    self.df[col],
                    errors="coerce",
                    dayfirst=self.config.dayfirst,
                    format=self.config.datetime_format
                )
        return self.df

    def _coerce_to_boolean(self, series):
        if pd.api.types.is_bool_dtype(series):
            return series.astype("boolean")

        mapping = {}
        for value in self.config.true_values:
            mapping[str(value).strip().lower()] = True
        for value in self.config.false_values:
            mapping[str(value).strip().lower()] = False

        normalized = series.astype("string").str.strip().str.lower()
        return normalized.map(mapping).astype("boolean")

    def force_declared_types(self):
        for col in self.config.force_numeric:
            if col in self.df.columns:
                self.df[col] = pd.to_numeric(self.df[col], errors="coerce")

        for col in self.config.force_string:
            if col in self.df.columns:
                self.df[col] = self.df[col].astype("string")

        for col in self.config.force_categorical:
            if col in self.df.columns:
                self.df[col] = self.df[col].astype("category")

        for col in self.config.force_boolean:
            if col in self.df.columns:
                self.df[col] = self._coerce_to_boolean(self.df[col])

        return self.df

    def create_derived_columns(self):
        for new_col, rule in self.config.derived_columns.items():
            if callable(rule):
                self.df[new_col] = rule(self.df)
            elif isinstance(rule, str):
                self.df[new_col] = self.df.eval(rule)
            else:
                self.df[new_col] = rule

        return self.df

    def detect_column_types(self):
        self.numeric_cols = []
        self.categorical_cols = []
        self.datetime_cols = []
        self.string_cols = []
        self.boolean_cols = []

        for col in self.df.columns:
            if col in self.config.text_columns:
                self.string_cols.append(col)
                continue

            if col in self.config.force_numeric:
                self.numeric_cols.append(col)
                continue

            if col in self.config.force_categorical:
                self.categorical_cols.append(col)
                continue

            if col in self.config.force_datetime:
                self.datetime_cols.append(col)
                continue

            if col in self.config.force_string:
                self.string_cols.append(col)
                continue

            if col in self.config.force_boolean:
                self.boolean_cols.append(col)
                continue

            series = self.df[col]

            if pd.api.types.is_bool_dtype(series):
                self.boolean_cols.append(col)
                continue

            if pd.api.types.is_datetime64_any_dtype(series):
                self.datetime_cols.append(col)
                continue

            numeric_try = pd.to_numeric(series, errors="coerce")
            numeric_ratio = numeric_try.notna().mean()

            datetime_try = pd.to_datetime(
                series,
                errors="coerce",
                dayfirst=self.config.dayfirst,
                format=self.config.datetime_format
            )
            datetime_ratio = datetime_try.notna().mean()

            n_unique = series.nunique(dropna=True)

            if numeric_ratio >= self.config.numeric_ratio_threshold:
                self.numeric_cols.append(col)
            elif datetime_ratio >= self.config.datetime_ratio_threshold:
                self.datetime_cols.append(col)
            else:
                if self.config.categorical_max_unique is not None and n_unique > self.config.categorical_max_unique:
                    self.string_cols.append(col)
                else:
                    self.categorical_cols.append(col)

        return {
            "numeric": self.numeric_cols,
            "categorical": self.categorical_cols,
            "datetime": self.datetime_cols,
            "string": self.string_cols,
            "boolean": self.boolean_cols
        }

    # =====================
    # Helpers
    # =====================
    def _is_categorical(self, series):
        return isinstance(series.dtype, pd.CategoricalDtype)

    def _should_include_missing(self, col_name: str) -> bool:
        return self.config.include_missing_by_column.get(
            col_name,
            self.config.include_missing_in_frequency
        )

    def _get_missing_label(self, col_name: str) -> str:
        return self.config.missing_label_by_column.get(
            col_name,
            self.config.missing_label
        )

    def _fill_missing_for_display(self, series):
        include_missing = self._should_include_missing(series.name)

        if not include_missing:
            return series

        label = self._get_missing_label(series.name)

        if self._is_categorical(series):
            if label not in series.cat.categories:
                series = series.cat.add_categories([label])
            return series.fillna(label)

        return series.fillna(label)


    def _merge_rare_categories(self, series):
        if not self.config.merge_rare_categories or self.config.rare_category_threshold is None:
            return series

        freq_pct = series.value_counts(normalize=True, dropna=False) * 100
        rare_values = freq_pct[freq_pct < self.config.rare_category_threshold].index

        if len(rare_values) == 0:
            return series

        label = self.config.rare_category_label

        if self._is_categorical(series) and label not in series.cat.categories:
            series = series.cat.add_categories([label])

        return series.replace(list(rare_values), label)

    def _detected_type_for_column(self, col):
        if col in self.numeric_cols:
            return "numeric"
        if col in self.categorical_cols:
            return "categorical"
        if col in self.datetime_cols:
            return "datetime"
        if col in self.string_cols:
            return "string"
        if col in self.boolean_cols:
            return "boolean"
        return "unknown"

    # =====================
    # Reportes
    # =====================
    def dataset_summary(self):
        summary = {
            "dataset_name": self.config.dataset_name,
            "file_path": str(self.file_path),
            "n_rows": self.df.shape[0],
            "n_columns": self.df.shape[1],
            "numeric_columns": len(self.numeric_cols),
            "categorical_columns": len(self.categorical_cols),
            "datetime_columns": len(self.datetime_cols),
            "string_columns": len(self.string_cols),
            "boolean_columns": len(self.boolean_cols),
            "total_missing_cells": int(self.df.isna().sum().sum()),
            "missing_cells_pct": round((self.df.isna().sum().sum() / self.df.size) * 100, 2) if self.df.size else 0
        }
        return pd.DataFrame([summary])

    def column_summary(self):
        rows = []
        for col in self.df.columns:
            rows.append({
                "variable": col,
                "pandas_dtype": str(self.df[col].dtype),
                "detected_type": self._detected_type_for_column(col),
                "missing_n": int(self.df[col].isna().sum()),
                "missing_pct": round(float(self.df[col].isna().mean() * 100), 2),
                "n_unique": int(self.df[col].nunique(dropna=True))
            })
        return pd.DataFrame(rows)

    def missing_summary(self):
        if not self.config.analyze_missingness:
            return pd.DataFrame()

        return pd.DataFrame({
            "variable": self.df.columns,
            "dtype": self.df.dtypes.astype(str).values,
            "missing_n": self.df.isna().sum().values,
            "missing_pct": (self.df.isna().mean() * 100).round(2).values,
            "n_unique": self.df.nunique(dropna=True).values
        }).sort_values(["missing_pct", "n_unique"], ascending=[False, True])

    def numeric_summary(self):
        if not self.numeric_cols or not self.config.analyze_numeric:
            return pd.DataFrame()

        df_num = self.df[self.numeric_cols].apply(pd.to_numeric, errors="coerce")
        summary = df_num.describe(percentiles=self.config.quantiles).T

        summary["median"] = df_num.median()
        summary["missing_n"] = df_num.isna().sum()
        summary["missing_pct"] = (df_num.isna().mean() * 100).round(2)
        summary["iqr"] = df_num.quantile(0.75) - df_num.quantile(0.25)

        if self.config.detect_outliers_iqr:
            outlier_counts = {}
            for col in df_num.columns:
                q1 = df_num[col].quantile(0.25)
                q3 = df_num[col].quantile(0.75)
                iqr = q3 - q1
                lower = q1 - self.config.outlier_iqr_multiplier * iqr
                upper = q3 + self.config.outlier_iqr_multiplier * iqr
                outlier_counts[col] = df_num[col].between(lower, upper, inclusive="both").eq(False).sum()

            summary["outlier_n_iqr"] = pd.Series(outlier_counts)

        summary = summary.reset_index().rename(columns={"index": "variable"})
        return summary

    def categorical_summary(self):
        cols_to_summarize = self.categorical_cols + self.boolean_cols

        if not cols_to_summarize or not self.config.analyze_categorical:
            return pd.DataFrame()

        rows = []

        for col in cols_to_summarize:
            series = self.df[col].copy()
            include_missing = self._should_include_missing(col)

            if include_missing:
                series = self._fill_missing_for_display(series)

            series = self._merge_rare_categories(series)

            freq = series.value_counts(dropna=not include_missing)
            pct = series.value_counts(dropna=not include_missing, normalize=True) * 100

            if self.config.max_categories_table is not None:
                freq = freq.head(self.config.max_categories_table)

            for value, count in freq.items():
                if count < self.config.min_frequency:
                    continue

                rows.append({
                    "variable": col,
                    "value": value,
                    "count": int(count),
                    "percent": round(float(pct.loc[value]), 2)
                })

        return pd.DataFrame(rows)

    def datetime_summary(self):
        if not self.datetime_cols or not self.config.analyze_datetime:
            return pd.DataFrame()

        rows = []
        for col in self.datetime_cols:
            s = pd.to_datetime(self.df[col], errors="coerce")
            rows.append({
                "variable": col,
                "missing_n": int(s.isna().sum()),
                "missing_pct": round(float(s.isna().mean() * 100), 2),
                "min_date": s.min(),
                "max_date": s.max(),
                "n_unique": int(s.nunique(dropna=True))
            })

        return pd.DataFrame(rows)

    def generate_reports(self):
        reports = {
            "dataset_summary": self.dataset_summary(),
            "column_summary": self.column_summary(),
            "missing_summary": self.missing_summary(),
            "numeric_summary": self.numeric_summary(),
            "categorical_summary": self.categorical_summary(),
            "datetime_summary": self.datetime_summary(),
        }

        if self.config.save_reports:
            self._save_report_dict(reports)

        return reports

    def _save_report_dict(self, reports: dict):
        for report_name, df_report in reports.items():
            if self.config.export_format == "csv":
                df_report.to_csv(self.output_dir / f"{report_name}.csv", index=False)
            elif self.config.export_format == "xlsx":
                df_report.to_excel(self.output_dir / f"{report_name}.xlsx", index=False)
            else:
                raise ValueError(f"Formato de exportación no soportado: {self.config.export_format}")

    # =====================
    # Gráficos
    # =====================
    def generate_plots(self):
        self._apply_editorial_style()
        self._plot_numeric_distributions()
        self._plot_categorical_distributions()

    def _apply_editorial_style(self):
        """
        Aplica el estilo limpio usado en generar_figuras.py (bias_analysis).
        Sin dependencia de scienceplots.
        """
        plt.rcParams.update({
            "font.family": "sans-serif",
            "font.size": 11,
            "axes.titlesize": 13,
            "axes.labelsize": 12,
            "legend.fontsize": 10,
            "figure.dpi": 200,
            "figure.facecolor": "white",
            "savefig.facecolor": "white",
            "axes.edgecolor": "#333333",
            "axes.linewidth": 0.8,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "grid.alpha": 0.35,
            "grid.linestyle": "--",
            "xtick.labelsize": 10,
            "ytick.labelsize": 10,
        })

    # Paleta profesional adaptada de generar_figuras.py
    _FIXED_PALETTE = ["#4C72B0", "#DD8452", "#66c2a5", "#fc8d62", "#8da0cb"]

    def _editorial_palette(self, n: int):
        if n <= len(self._FIXED_PALETTE):
            return self._FIXED_PALETTE[:n]
        return self._FIXED_PALETTE + sns.color_palette("husl", n - len(self._FIXED_PALETTE)).as_hex()

    @staticmethod
    def _clean_spines(ax):
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

    def _plot_numeric_distributions(self):
        if not self.config.analyze_numeric:
            return

        plot_dir = self.output_dir / "plots_numeric"
        plot_dir.mkdir(parents=True, exist_ok=True)

        colors = self._editorial_palette(2)

        for col in self.numeric_cols:
            s = pd.to_numeric(self.df[col], errors="coerce").dropna()
            if s.empty:
                continue

            mean_val = s.mean()
            median_val = s.median()

            fig, axes = plt.subplots(
                1, 2,
                figsize=(self.config.figure_width * 2, self.config.figure_height)
            )

            sns.histplot(
                s, kde=True, ax=axes[0],
                color=colors[0], edgecolor="white", linewidth=0.6, alpha=0.85
            )
            axes[0].axvline(
                mean_val, color=colors[1 % len(colors)], linestyle="--", linewidth=1.4,
                label=f"Media = {mean_val:.2f}"
            )
            axes[0].axvline(
                median_val, color="#444444", linestyle=":", linewidth=1.4,
                label=f"Mediana = {median_val:.2f}"
            )
            display = self.config.column_title_map.get(col, col)
            axes[0].set_title("Histograma")
            axes[0].set_xlabel(display)
            axes[0].set_ylabel("Frecuencia")
            axes[0].legend(frameon=False, loc="upper right")
            self._clean_spines(axes[0])

            sns.boxplot(
                x=s, ax=axes[1], color=colors[0], width=0.35,
                fliersize=3, linewidth=1.0
            )
            axes[1].set_title("Boxplot")
            axes[1].set_xlabel(display)
            self._clean_spines(axes[1])

            # mathtext ($...$) funciona sin necesidad de una instalación de LaTeX real.
            stats_text = f"n = {s.shape[0]}\n$\\sigma$ = {s.std():.2f}"
            axes[1].annotate(
                stats_text,
                xy=(0.98, 0.95), xycoords="axes fraction",
                ha="right", va="top", fontsize=9, color="#555555",
                bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="#cccccc", alpha=0.85)
            )

            fig.suptitle(display, fontsize=14, fontweight="bold", y=1.03)
            plt.tight_layout()

            if self.config.save_plots:
                fig.savefig(
                    plot_dir / f"{col}.{self.config.plot_format}",
                    dpi=self.config.plot_dpi,
                    bbox_inches="tight"
                )
            if self.config.show_plots:
                plt.show()

            plt.close(fig)


    def _plot_categorical_distributions(self):
        if not self.config.analyze_categorical:
            return

        plot_dir = self.output_dir / "plots_categorical"
        plot_dir.mkdir(parents=True, exist_ok=True)

        colors = self._editorial_palette(2)

        for col in self.categorical_cols + self.boolean_cols:
            series = self.df[col].copy()
            include_missing = self._should_include_missing(col)

            if include_missing:
                series = self._fill_missing_for_display(series)

            series = self._merge_rare_categories(series)

            freq = series.value_counts(dropna=not include_missing).head(
                self.config.max_categories_plot
            )

            if freq.empty:
                continue

            plot_type = self._get_categorical_plot_type(col)

            total = int(freq.sum())
            display = self.config.column_title_map.get(col, col)

            if plot_type == "bar":
                fig, ax = plt.subplots(
                    figsize=(self.config.figure_width, self.config.figure_height)
                )

                ordered = freq.sort_values(ascending=False)

                sns.barplot(
                    x=ordered.index.astype(str),
                    y=ordered.values,
                    ax=ax,
                    color=colors[0],
                    edgecolor="white",
                    linewidth=0.6
                )
                for container in ax.containers:
                    ax.bar_label(
                        container,
                        labels=[f"{v:.0f}\n({v / total:.0%})" for v in ordered.values],
                        padding=3, fontsize=8.5, color="#333333"
                    )
                ax.set_title(f"Distribución de {display}")
                ax.set_ylabel("Frecuencia")
                ax.set_xlabel("")
                ax.tick_params(axis="x", rotation=self.config.rotate_xticks)
                ax.margins(y=0.15)
                self._clean_spines(ax)

            elif plot_type == "barh":
                fig_height = max(self.config.figure_height, 0.45 * len(freq))
                fig, ax = plt.subplots(
                    figsize=(self.config.figure_width, fig_height)
                )

                ordered = freq.sort_values(ascending=True)

                sns.barplot(
                    x=ordered.values,
                    y=ordered.index.astype(str),
                    ax=ax,
                    color=colors[0],
                    edgecolor="white",
                    linewidth=0.6,
                    orient="h"
                )
                for container in ax.containers:
                    ax.bar_label(
                        container,
                        labels=[f"{v:.0f} ({v / total:.0%})" for v in ordered.values],
                        padding=3, fontsize=8.5, color="#333333"
                    )
                ax.set_title(f"Distribución de {display}")
                ax.set_xlabel("Frecuencia")
                ax.set_ylabel("")
                ax.margins(x=0.15)
                self._clean_spines(ax)

            elif plot_type == "lollipop":
                fig_height = max(self.config.figure_height, 0.45 * len(freq))
                fig, ax = plt.subplots(
                    figsize=(self.config.figure_width, fig_height)
                )

                ordered = freq.sort_values(ascending=True)
                y_pos = np.arange(len(ordered))

                ax.hlines(
                    y=y_pos,
                    xmin=0,
                    xmax=ordered.values,
                    color="#999999",
                    alpha=0.7,
                    linewidth=1.4
                )
                ax.plot(
                    ordered.values,
                    y_pos,
                    "o",
                    color=colors[0],
                    markersize=8,
                    markeredgecolor="white",
                    markeredgewidth=0.8
                )
                for x_val, y_val in zip(ordered.values, y_pos):
                    ax.annotate(
                        f"{x_val:.0f} ({x_val / total:.0%})",
                        xy=(x_val, y_val), xytext=(6, 0),
                        textcoords="offset points",
                        va="center", fontsize=8.5, color="#333333"
                    )

                ax.set_yticks(y_pos)
                ax.set_yticklabels(ordered.index.astype(str))
                ax.set_xlabel("Frecuencia")
                ax.set_ylabel("")
                ax.set_title(f"Distribución de {display}")
                ax.margins(x=0.18)
                self._clean_spines(ax)
                ax.spines["left"].set_visible(False)
                ax.tick_params(axis="y", length=0)

            else:
                raise ValueError(f"Tipo de gráfico categórico no soportado: {plot_type}")

            plt.tight_layout()

            if self.config.save_plots:
                fig.savefig(
                    plot_dir / f"{col}.{self.config.plot_format}",
                    dpi=self.config.plot_dpi,
                    bbox_inches="tight"
                )
            if self.config.show_plots:
                plt.show()

            plt.close(fig)

    def _get_categorical_plot_type(self, col: str) -> str:
        return self.config.categorical_plot_type_by_column.get(
            col,
            self.config.categorical_plot_type
        )

    # =====================
    # Guardado extra
    # =====================
    def save_cleaned_data(self):
        if self.config.export_format == "csv":
            self.df.to_csv(self.output_dir / "cleaned_dataset.csv", index=False)
        elif self.config.export_format == "xlsx":
            self.df.to_excel(self.output_dir / "cleaned_dataset.xlsx", index=False)

    # =====================
    # Utilidades
    # =====================
    def get_processed_data(self):
        return self.df.copy()

    def get_raw_data(self):
        return self.df_raw.copy()

    def print_type_summary(self):
        print("Numéricas:", self.numeric_cols)
        print("Categóricas:", self.categorical_cols)
        print("Fechas:", self.datetime_cols)
        print("Texto libre:", self.string_cols)
        print("Booleanas:", self.boolean_cols) 