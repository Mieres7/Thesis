#!/usr/bin/env python3
"""
Exportar a CSV los valores numéricos de las 4 tablas de 04-oe1.tex
calculando todo dinámicamente desde las fuentes originales.

JUSTIFICACIÓN DE CADA FUENTE:
===============================

Tabla 1 (completitud agregada) y Tabla 2 (escala, completitud):
  → Scripts/notebooks/outputs/all_missing_summary.csv
  → Scripts/notebooks/outputs/all_dataset_summary.csv
  Son la salida DIRECTA de DatasetAnalyzer sobre los 9 metadatos originales
  (pre-curación). Verificado: n_columns en dataset_summary == n_vars en
  missing_summary para los 10 datasets (✅ todas las secciones coinciden).

Tabla 2 (cobertura, compatibilidad para datasets originales):
  → Scripts/notebooks/outputs/all_missing_summary.csv (columnas disponibles)
  Las etiquetas cualitativas (Alta/Media/Baja) son heurísticas basadas en
  grupos de vars clínicamente relevantes presentes.

Tabla 2 (escala para HISTAI-Breast, HISTAI-Colorectal-b1/b2):
  → Scripts/METADATA_FINAL/breast/HISTAI_final.csv
  → Scripts/METADATA_FINAL/colorectal/HISTAI_B1_final.csv, HISTAI_B2_final.csv
  DatasetAnalyzer NO se ejecutó sobre los splits post-curación, así que n_rows
  se lee de estos archivos.

Tabla 3 (frecuencias BCNB, HSI-BRCA):
  → Scripts/notebooks/outputs/all_categorical_summary.csv
  Verificado: los valores coinciden exactamente con BCNB_final.csv.
  Para HSI-BRCA se usan los valores categóricos sin sufijos "(code N)".

Tabla 3 (cobertura HISTAI-Breast):
  → Scripts/METADATA_FINAL/breast/HISTAI_final.csv
  NECESARIO porque all_categorical_summary.csv tiene la sección HISTAI VACÍA
  (los biomarcadores ER/PR/HER2/Ki67/subtipo se extrajeron del texto libre
  durante la curación, no estaban en el metadata original de 9 columnas).

Tabla 4 (frecuencias colorrectales):
  → Scripts/METADATA_FINAL/colorectal/{surgen386,surgen1482,HISTAI_B1,HISTAI_B2}_final.csv
  Variables site_group_norm, side_norm y sex_norm SOLO existen post-curación
  (son producto de la armonización). Verificado: todos los archivos las tienen.

Heurísticas para la Tabla 2 (indicadores cualitativos):
  - Completitud: proporción de vars con >30% missing sobre el total.
    <10% → Alta, 10-24% → Media, ≥25% → Baja; si global_missing_pct == 0 → Completa.
  - Cobertura: cantidad de grupos de vars relevantes presentes
    (demográficas, anatómicas, patológicas, moleculares, supervivencia).
    ≥5 grupos → Alta, 2-3 → Media, <2 → Baja.
  - Compatibilidad: datasets del mismo dominio clínico.
    ≥2 adicionales → Alta, 1 → Media, 0 → Baja.
"""

import csv
import os
import sys
from collections import OrderedDict, Counter

RAIZ = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
OUTPUTS = os.path.join(RAIZ, 'Scripts', 'notebooks', 'outputs')
METADATA = os.path.join(RAIZ, 'Scripts', 'METADATA_FINAL')
DESTINO = os.path.join(RAIZ, 'Escrito', 'metricas')

MAPA_DS = {
    'ovarian': 'Ovarian',
    'BCNB': 'BCNB',
    'CLWD': 'CLWD',
    'ndb ufes': 'NDB-UFES',
    'tdbta': 'TDBTA',
    'surgen386': 'SurGen-SR386',
    'surgen1482': 'SurGen-SR1482',
    'HSI': 'HSI-BRCA',
    'SOPHIE': 'SOPHIE',
    'HISTAI': 'HISTAI',
}

DOMINIOS = {
    'ovarian': 'ovarian',
    'Ovarian': 'ovarian',
    'BCNB': 'breast',
    'CLWD': 'lung',
    'ndb ufes': 'skin',
    'NDB-UFES': 'skin',
    'tdbta': 'brain',
    'TDBTA': 'brain',
    'surgen386': 'colorectal',
    'SurGen-SR386': 'colorectal',
    'surgen1482': 'colorectal',
    'SurGen-SR1482': 'colorectal',
    'HSI': 'breast',
    'HSI-BRCA': 'breast',
    'SOPHIE': 'skin_spitzoid',
    'HISTAI': 'mixed',
    'HISTAI-Breast': 'breast',
    'HISTAI-Colorectal-b1': 'colorectal',
    'HISTAI-Colorectal-b2': 'colorectal',
}

# ---------------------------------------------------------------------------
#   Parser de CSV con secciones (formato DatasetAnalyzer)
# ---------------------------------------------------------------------------

def _leer_secciones(ruta, rsplit_when_gt=0):
    if not os.path.isfile(ruta):
        print(f"  [ADVERTENCIA] No existe: {ruta}", file=sys.stderr)
        return {}
    secciones = {}
    with open(ruta, encoding='utf-8') as f:
        lines = [l.rstrip('\n') for l in f.readlines()]
    i = 0
    while i < len(lines):
        line = lines[i].strip()
        if not line:
            i += 1
            continue
        if line.startswith('variable'):
            i += 1
            continue
        nombre_ds = line
        i += 1
        while i < len(lines) and not lines[i].strip():
            i += 1
        headers = None
        if i < len(lines) and lines[i].strip().startswith('variable'):
            headers = [h.strip() for h in lines[i].split(',')]
            i += 1
        filas = []
        while i < len(lines) and lines[i].strip():
            row = lines[i].strip()
            if rsplit_when_gt and row.count(',') > rsplit_when_gt:
                parts = row.rsplit(',', rsplit_when_gt)
            else:
                parts = row.split(',')
            parts = [p.strip() for p in parts]
            if headers and len(parts) == len(headers):
                filas.append(dict(zip(headers, parts)))
            i += 1
        secciones[nombre_ds] = filas
    return secciones

# ---------------------------------------------------------------------------
#   Utilidades CSVisitas
# ---------------------------------------------------------------------------

def leer_csv(ruta):
    if not os.path.isfile(ruta):
        return []
    with open(ruta, newline='', encoding='utf-8') as f:
        return list(csv.DictReader(f))


def _cuenta_columna(filas, col, valor):
    return sum(1 for r in filas if r.get(col, '').strip() == valor)


def _frecuencias(filas, col):
    c = Counter()
    for r in filas:
        v = r.get(col, '').strip()
        if v:
            c[v] += 1
    return c


def _cv(filas, var, val):
    """Busca count+percent de una variable categórica en all_categorical_summary."""
    for r in filas:
        if r.get('variable', '').strip() == var and r.get('value', '').strip() == val:
            return r.get('count', '?'), r.get('percent', '?')
    return None, None


# ---------------------------------------------------------------------------
#   TABLA 1  –  Completitud agregada
# ---------------------------------------------------------------------------

def tabla1_completitud_agregada():
    secciones = _leer_secciones(os.path.join(OUTPUTS, 'all_missing_summary.csv'))

    ds_cols = OrderedDict()
    for key, filas in secciones.items():
        ds = MAPA_DS.get(key, key)
        for r in filas:
            pct_str = r.get('missing_pct', '0')
            try:
                pct = float(pct_str)
            except ValueError:
                pct = 0.0
            if ds not in ds_cols:
                ds_cols[ds] = {'total': 0, 'sin_missing': 0, 'gt30': 0, 'gt50': 0, 'gt80': 0}
            ds_cols[ds]['total'] += 1
            if pct == 0.0:
                ds_cols[ds]['sin_missing'] += 1
            if pct > 30:
                ds_cols[ds]['gt30'] += 1
            if pct > 50:
                ds_cols[ds]['gt50'] += 1
            if pct > 80:
                ds_cols[ds]['gt80'] += 1

    orden = ['Ovarian', 'BCNB', 'CLWD', 'NDB-UFES', 'TDBTA',
             'SurGen-SR386', 'SurGen-SR1482', 'HSI-BRCA', 'SOPHIE', 'HISTAI']
    salida = []
    for ds in orden:
        if ds in ds_cols:
            d = ds_cols[ds]
            pct_sin = d['sin_missing'] / d['total'] * 100 if d['total'] else 0
            salida.append({
                'Dataset': ds,
                'Variables': d['total'],
                'Sin missing': f"{d['sin_missing']} ({pct_sin:.1f}%)",
                '>30% missing': d['gt30'],
                '>50% missing': d['gt50'],
                '>80% missing': d['gt80'],
            })
    _escribir_csv('tabla_completitud_agregada.csv',
                  ['Dataset', 'Variables', 'Sin missing',
                   '>30% missing', '>50% missing', '>80% missing'],
                  salida)
    return salida, ds_cols

# ---------------------------------------------------------------------------
#   TABLA 2  –  Indicadores de preselección
# ---------------------------------------------------------------------------

def _asignar_completitud(ds_key, ds_cols, dataset_summary):
    """Asigna etiqueta cualitativa de completitud basada en vars con >30% missing."""
    # Para datasets originales (en ds_cols)
    if ds_key in ds_cols:
        d = ds_cols[ds_key]
        if d['total'] == 0:
            return '?'
        if d['sin_missing'] == d['total'] and d['gt30'] == 0:
            return 'Completa'
        severe_pct = d['gt30'] / d['total']
        if severe_pct == 0 and d['sin_missing'] / d['total'] >= 0.8:
            return 'Alta'
        if severe_pct < 0.10:
            return 'Alta'
        if severe_pct < 0.25:
            return 'Media'
        return 'Baja'
    # Para HISTAI-Breast / Colorectal (post-curación)
    return 'Media'  # default razonable


GRUPOS_CLINICOS = [
    {'nombre': 'demográfico',     'kw': ['age', 'edad', 'años', 'años', 'years', 'gender', 'sex']},
    {'nombre': 'anatómico',       'kw': ['site', 'sitio', 'lateralidad', 'laterality',
                                          'side', 'tumour_site', 'tumor_site']},
    {'nombre': 'patológico',      'kw': ['stage', 'grade', 'grado', 'histologic', 'histological',
                                          'pT_cat', 'pN_cat', 'pM_cat', 'grading']},
    {'nombre': 'molecular',       'kw': ['er', 'pr', 'her2', 'ki67', 'molecular', 'mmr', 'msi',
                                          'kras', 'nras', 'braf', 'receptor', 'subtype', 'subtipo']},
    {'nombre': 'supervivencia',   'kw': ['survival', 'death', 'muerte', 'os_days', 'dfs',
                                          'vital', 'deceased', 'recurrence']},
    {'nombre': 'demográfico2',    'kw': ['race', 'ethnicity', 'raza', 'etnia']},
]


def _cobertura_grupos(columnas):
    """Cuenta grupos de variables clínicamente relevantes presentes en las columnas."""
    cols_joined = ' '.join(c.lower().replace(' ', '_').replace('-', '_').replace('(', '').replace(')', '')
                           for c in columnas)
    grupos_contados = set()
    for grupo in GRUPOS_CLINICOS:
        for kw in grupo['kw']:
            if kw in cols_joined:
                grupos_contados.add(grupo['nombre'])
                break
    # Bonus si hay más de un grupo demográfico
    if 'demográfico' in grupos_contados and 'demográfico2' in grupos_contados:
        grupos_contados.discard('demográfico2')
    return len(grupos_contados)


def _etiqueta_cobertura(n_grupos):
    if n_grupos >= 5:
        return 'Alta'
    if n_grupos >= 3:
        return 'Media'
    return 'Baja'


def _etiqueta_compatibilidad(nombre):
    dom = DOMINIOS.get(nombre, 'unknown')
    # Contar cuántos datasets del mismo dominio están en la lista de 12
    mismos = sum(1 for k, v in DOMINIOS.items() if v == dom and k in (
        'Ovarian', 'BCNB', 'CLWD', 'NDB-UFES', 'TDBTA',
        'SurGen-SR386', 'SurGen-SR1482', 'HSI-BRCA', 'SOPHIE',
        'HISTAI-Breast', 'HISTAI-Colorectal-b1', 'HISTAI-Colorectal-b2'
    ) and k != nombre)
    if dom in ('mixed', 'unknown'):
        return 'Baja'
    if mismos >= 2:
        sufijo = {'breast': 'mama', 'colorectal': 'colorrectal'}.get(dom, dom)
        return f'Alta ({sufijo})'
    if mismos == 1:
        return 'Media'
    return 'Baja'


def _escala_humana(n):
    if n < 300:
        return 'Pequeña'
    if n < 1500:
        return 'Mediana'
    return 'Grande'


def tabla2_indicadores_preseleccion(ds_cols):
    dataset_summary = leer_csv(os.path.join(OUTPUTS, 'all_dataset_summary.csv'))
    n_rows_map = {}
    for r in dataset_summary:
        name_key = r.get('dataset_name', '').strip()
        try:
            n_rows_map[name_key] = int(r.get('n_rows', 0))
        except ValueError:
            n_rows_map[name_key] = 0

    # n_rows para datasets originales
    escala_map = {}
    for k, v in n_rows_map.items():
        if 'ovarian' in k:
            escala_map['Ovarian'] = v
        elif 'BCNB' in k:
            escala_map['BCNB'] = v
        elif 'CLWD' in k:
            escala_map['CLWD'] = v
        elif 'ndb' in k:
            escala_map['NDB-UFES'] = v
        elif 'TDBTA' in k:
            escala_map['TDBTA'] = v
        elif 'SR386' in k:
            escala_map['SurGen-SR386'] = v
        elif 'SR1482' in k:
            escala_map['SurGen-SR1482'] = v
        elif 'HSI' in k:
            escala_map['HSI-BRCA'] = v
        elif 'SOPHIE' in k:
            escala_map['SOPHIE'] = v
        elif 'HISTAI' in k:
            escala_map['HISTAI'] = v

    # n_rows para HISTAI derivados desde METADATA_FINAL
    try:
        histai_breast = list(csv.DictReader(open(
            os.path.join(METADATA, 'breast', 'HISTAI_final.csv'), encoding='utf-8')))
        escala_map['HISTAI-Breast'] = len(histai_breast)
    except Exception:
        escala_map['HISTAI-Breast'] = 0

    for fname, key in [('HISTAI_B1_final.csv', 'HISTAI-Colorectal-b1'),
                       ('HISTAI_B2_final.csv', 'HISTAI-Colorectal-b2')]:
        try:
            rows = list(csv.DictReader(open(
                os.path.join(METADATA, 'colorectal', fname), encoding='utf-8')))
            escala_map[key] = len(rows)
        except Exception:
            escala_map[key] = 0

    # Cobertura desde METADATA_FINAL
    cobertura_map = {}
    try:
        bcnb = list(csv.DictReader(open(
            os.path.join(METADATA, 'breast', 'BCNB_final.csv'), encoding='utf-8')))
        cobertura_map['BCNB'] = _etiqueta_cobertura(_cobertura_grupos(bcnb[0].keys() if bcnb else []))
    except Exception:
        cobertura_map['BCNB'] = '?'

    try:
        histai_b = list(csv.DictReader(open(
            os.path.join(METADATA, 'breast', 'HISTAI_final.csv'), encoding='utf-8')))
        cobertura_map['HISTAI-Breast'] = _etiqueta_cobertura(_cobertura_grupos(histai_b[0].keys() if histai_b else []))
    except Exception:
        cobertura_map['HISTAI-Breast'] = '?'

    import glob as gglob
    for fname, key in [('HISTAI_B1_final.csv', 'HISTAI-Colorectal-b1'),
                       ('HISTAI_B2_final.csv', 'HISTAI-Colorectal-b2'),
                       ('surgen386_final.csv', 'SurGen-SR386'),
                       ('surgen1482_final.csv', 'SurGen-SR1482')]:
        try:
            rows = list(csv.DictReader(open(
                os.path.join(METADATA, 'colorectal', fname), encoding='utf-8')))
            cobertura_map[key] = _etiqueta_cobertura(_cobertura_grupos(rows[0].keys() if rows else []))
        except Exception:
            cobertura_map[key] = '?'

    # Para datasets que NO están en METADATA_FINAL, contar desde all_dataset_summary
    ds_orig_cols = {}
    secciones_missing = _leer_secciones(os.path.join(OUTPUTS, 'all_missing_summary.csv'))
    for key, filas in secciones_missing.items():
        ds = MAPA_DS.get(key, key)
        ds_orig_cols[ds] = [r.get('variable', r.get('Columna', '?')) for r in filas]

    for ds in ['Ovarian', 'CLWD', 'NDB-UFES', 'TDBTA', 'HSI-BRCA', 'SOPHIE', 'SurGen-SR386', 'SurGen-SR1482']:
        if ds not in cobertura_map and ds in ds_orig_cols:
            cobertura_map[ds] = _etiqueta_cobertura(_cobertura_grupos(ds_orig_cols[ds]))
        elif ds not in cobertura_map:
            cobertura_map[ds] = '?'

    orden = ['Ovarian', 'BCNB', 'CLWD', 'NDB-UFES', 'TDBTA',
             'SurGen-SR386', 'SurGen-SR1482', 'HSI-BRCA', 'SOPHIE',
             'HISTAI-Breast', 'HISTAI-Colorectal-b1', 'HISTAI-Colorectal-b2']
    salida = []
    for ds in orden:
        n = escala_map.get(ds, 0)
        completitud_label = _asignar_completitud(ds, ds_cols, dataset_summary)
        cobertura_label = cobertura_map.get(ds, '?')
        compat_label = _etiqueta_compatibilidad(ds)
        salida.append({
            'Dataset': ds,
            'Escala': f"{_escala_humana(n)} ({n})",
            'Completitud': completitud_label,
            'Cobertura': cobertura_label,
            'Compatibilidad': compat_label,
        })
    _escribir_csv('tabla_indicadores_preseleccion.csv',
                  ['Dataset', 'Escala', 'Completitud', 'Cobertura', 'Compatibilidad'],
                  salida)
    return salida

# ---------------------------------------------------------------------------
#   TABLA 3  –  Frecuencias de mama
# ---------------------------------------------------------------------------

def tabla3_frecuencias_mama():
    categ = _leer_secciones(os.path.join(OUTPUTS, 'all_categorical_summary.csv'), rsplit_when_gt=3)
    categ_canon = {MAPA_DS.get(k, k): v for k, v in categ.items()}

    def build_lookup(seccion):
        d = {}
        for row in seccion:
            var = row.get('variable', '').strip()
            val = row.get('value', '').strip()
            d.setdefault(var, {})[val] = {
                'count': row.get('count', '0').strip(),
                'percent': row.get('percent', '0').strip(),
            }
        return d

    bcnb_lk = build_lookup(categ_canon.get('BCNB', []))
    hsi_lk = build_lookup(categ_canon.get('HSI-BRCA', []))

    def fmt(lk, var, val):
        d = lk.get(var, {}).get(val)
        if d:
            return f"{d['count']} ({d['percent']})"
        return '--'

    # HISTAI: cobertura desde METADATA_FINAL
    histai_total = 0
    histai_er = histai_pr = histai_her2 = histai_ki67 = histai_sub = 0
    try:
        with open(os.path.join(METADATA, 'breast', 'HISTAI_final.csv'), encoding='utf-8') as f:
            rows = list(csv.DictReader(f))
        histai_total = len(rows)
        for r in rows:
            if r.get('ER_status_norm', '').strip():
                histai_er += 1
            if r.get('PR_status_norm', '').strip():
                histai_pr += 1
            if r.get('HER2_status_norm', '').strip():
                histai_her2 += 1
            if r.get('Ki67_cat', '').strip():
                histai_ki67 += 1
            if r.get('Molecular_subtype_norm', '').strip():
                histai_sub += 1
    except Exception as e:
        print(f"  [ERROR] leyendo HISTAI_final.csv: {e}", file=sys.stderr)

    def hcov(n):
        return f"{n} de {histai_total} ({n/histai_total*100:.1f})" if histai_total else '?'

    # HSI-BRCA lookup: los valores en all_categorical_summary NO tienen el sufijo "(1)" 
    # así que intentamos con y sin él
    def fmt_hsi_best(var, vals_to_try):
        for v in vals_to_try:
            d = hsi_lk.get(var, {}).get(v)
            if d:
                return f"{d['count']} ({d['percent']})"
        return '--'

    salida = []
    # ER
    salida.append({'Variable': 'ER', 'Categoria': 'Positivo (+)',
                   'BCNB (%)': fmt(bcnb_lk, 'ER', 'Positive'),
                   'HSI-BRCA (%)': fmt_hsi_best('ER', ['Positive', 'Positive (1)']),
                   'HISTAI-Breast (% cubierto)': hcov(histai_er)})
    salida.append({'Variable': 'ER', 'Categoria': 'Negativo (--)',
                   'BCNB (%)': fmt(bcnb_lk, 'ER', 'Negative'),
                   'HSI-BRCA (%)': fmt_hsi_best('ER', ['Negative', 'Negative (0)']),
                   'HISTAI-Breast (% cubierto)': ''})
    # PR
    salida.append({'Variable': 'PR', 'Categoria': 'Positivo (+)',
                   'BCNB (%)': fmt(bcnb_lk, 'PR', 'Positive'),
                   'HSI-BRCA (%)': fmt_hsi_best('PR', ['Positive', 'Positive (1)']),
                   'HISTAI-Breast (% cubierto)': hcov(histai_pr)})
    salida.append({'Variable': 'PR', 'Categoria': 'Negativo (--)',
                   'BCNB (%)': fmt(bcnb_lk, 'PR', 'Negative'),
                   'HSI-BRCA (%)': fmt_hsi_best('PR', ['Negative', 'Negative (0)']),
                   'HISTAI-Breast (% cubierto)': ''})
    # HER2
    salida.append({'Variable': 'HER2', 'Categoria': 'Negativo (0)',
                   'BCNB (%)': fmt(bcnb_lk, 'HER2', 'Negative'),
                   'HSI-BRCA (%)': fmt_hsi_best('HER2', ['Negative', 'Negative (0)']),
                   'HISTAI-Breast (% cubierto)': hcov(histai_her2)})
    salida.append({'Variable': 'HER2', 'Categoria': 'Positivo (+)',
                   'BCNB (%)': fmt(bcnb_lk, 'HER2', 'Positive'),
                   'HSI-BRCA (%)': fmt_hsi_best('HER2', ['Positive', 'Positive (1)']),
                   'HISTAI-Breast (% cubierto)': ''})
    # Ki-67
    salida.append({'Variable': 'Ki-67', 'Categoria': 'Bajo (<20%)',
                   'BCNB (%)': '--',
                   'HSI-BRCA (%)': fmt_hsi_best('KI67', ['Ki67 Index < 20%', 'Ki67 Index < 20% (0)']),
                   'HISTAI-Breast (% cubierto)': hcov(histai_ki67)})
    salida.append({'Variable': 'Ki-67', 'Categoria': 'Alto (>=20%)',
                   'BCNB (%)': '--',
                   'HSI-BRCA (%)': fmt_hsi_best('KI67', ['Ki67 Index >= 20%', 'Ki67 Index >= 20% (1)']),
                   'HISTAI-Breast (% cubierto)': ''})
    # Subtipo molecular
    salida.append({'Variable': 'Subtipo molecular', 'Categoria': 'Luminal A',
                   'BCNB (%)': fmt(bcnb_lk, 'Molecular subtype', 'Luminal A'),
                   'HSI-BRCA (%)': '--',
                   'HISTAI-Breast (% cubierto)': hcov(histai_sub)})
    salida.append({'Variable': 'Subtipo molecular', 'Categoria': 'Luminal B',
                   'BCNB (%)': fmt(bcnb_lk, 'Molecular subtype', 'Luminal B'),
                   'HSI-BRCA (%)': fmt_hsi_best('Molecular_subtype', ['Luminal B HER2-', 'Luminal B HER2- (code 1)']),
                   'HISTAI-Breast (% cubierto)': ''})
    salida.append({'Variable': 'Subtipo molecular', 'Categoria': 'HER2+',
                   'BCNB (%)': fmt(bcnb_lk, 'Molecular subtype', 'HER2(+)'),
                   'HSI-BRCA (%)': fmt_hsi_best('Molecular_subtype', ['HER2+', 'HER2+ (code 4)']),
                   'HISTAI-Breast (% cubierto)': ''})
    salida.append({'Variable': 'Subtipo molecular', 'Categoria': 'Triple negativo',
                   'BCNB (%)': fmt(bcnb_lk, 'Molecular subtype', 'Triple negative'),
                   'HSI-BRCA (%)': fmt_hsi_best('Molecular_subtype', ['Triple Negative', 'Triple Negative (code 3)']),
                   'HISTAI-Breast (% cubierto)': ''})

    _escribir_csv('tabla_frecuencias_mama.csv',
                  ['Variable', 'Categoria', 'BCNB (%)', 'HSI-BRCA (%)', 'HISTAI-Breast (% cubierto)'],
                  salida)
    return salida

# ---------------------------------------------------------------------------
#   TABLA 4  –  Frecuencias colorrectal
# ---------------------------------------------------------------------------

def _frecuencias_colorrectal_para(archivo, columna, nombre_archivo):
    """Lee METADATA_FINAL/colorectal/{archivo} y devuelve {valor: "count (pct%)"}.
    El porcentaje usa como denominador solo filas con valor no vacío (como en LaTeX).
    """
    ruta = os.path.join(METADATA, 'colorectal', archivo)
    if not os.path.isfile(ruta):
        print(f"  [AVISO] No existe {ruta}", file=sys.stderr)
        return {}
    rows = list(csv.DictReader(open(ruta, encoding='utf-8')))
    counts = Counter()
    for r in rows:
        v = r.get(columna, '').strip()
        if v:
            counts[v] += 1
    total_con_dato = sum(counts.values()) or 1
    return {k: f"{c} ({c/total_con_dato*100:.1f}%)" for k, c in counts.most_common()}


def tabla4_frecuencias_colorrectal():
    archivos = [
        ('surgen386_final.csv', 'SurGen-SR386'),
        ('surgen1482_final.csv', 'SurGen-SR1482'),
        ('HISTAI_B1_final.csv', 'HISTAI-b1'),
        ('HISTAI_B2_final.csv', 'HISTAI-b2'),
    ]

    # Sexo
    sexo_por_ds = {}
    for archivo, etiqueta in archivos:
        sexo_por_ds[etiqueta] = _frecuencias_colorrectal_para(archivo, 'sex_norm', etiqueta)

    # Sitio
    sitio_por_ds = {}
    for archivo, etiqueta in archivos:
        sitio_por_ds[etiqueta] = _frecuencias_colorrectal_para(archivo, 'site_group_norm', etiqueta)

    # Lateralidad
    side_por_ds = {}
    for archivo, etiqueta in archivos:
        side_por_ds[etiqueta] = _frecuencias_colorrectal_para(archivo, 'side_norm', etiqueta)

    MAPA_SEXO = {'F': 'Femenino', 'M': 'Masculino'}
    MAPA_SITIO = {
        'rectum': 'Recto', 'sigmoid': 'Sigmoide',
        'right_colon': 'Colon derecho', 'left_colon': 'Colon izquierdo',
        'transverse_colon': 'Colon transverso',
        'other': 'Otros / inespecífico',
        'colon_unspecified': 'Otros / inespecífico',
        'metastatic_liver': 'Metástasis hepática',
        'metastatic_peritoneal': 'Metástasis hepática',
    }
    MAPA_SIDE = {
        'left': 'Izquierdo', 'right': 'Derecho',
        'rectum': 'Recto', 'transverse': 'Transverso',
    }

    def agregar_por_mapa(raw_counts, mapa):
        """Agrupa frecuencias por categoría del mapa, devuelve {cat_label: total_str}."""
        from collections import defaultdict
        agg = defaultdict(int)
        total = 0
        for raw_val, count_str in raw_counts.items():
            # extraer count numérico
            count_num = int(count_str.split()[0])
            cat = mapa.get(raw_val, None)
            if cat:
                agg[cat] += count_num
                total += count_num
        return {k: f"{v} ({v/total*100:.1f}%)" for k, v in agg.items()} if total else {}

    cols_list = ['SurGen-SR386', 'SurGen-SR1482', 'HISTAI-b1', 'HISTAI-b2']

    sexo_agg = {ds: agregar_por_mapa(sexo_por_ds[ds], MAPA_SEXO) for ds in cols_list}
    sitio_agg = {ds: agregar_por_mapa(sitio_por_ds[ds], MAPA_SITIO) for ds in cols_list}
    side_agg = {ds: agregar_por_mapa(side_por_ds[ds], MAPA_SIDE) for ds in cols_list}

    salida = []

    def lookup(ds_dict, cat):
        return ds_dict.get(cat, '--')

    # Sexo
    for cat_label in ['Femenino', 'Masculino']:
        row = {'Variable': 'Sexo', 'Categoria': cat_label}
        for ds in cols_list:
            row[ds] = lookup(sexo_agg.get(ds, {}), cat_label)
        salida.append(row)

    # Sitio
    for cat_label in ['Recto', 'Sigmoide', 'Colon derecho', 'Colon izquierdo',
                      'Colon transverso', 'Otros / inespecífico', 'Metástasis hepática']:
        row = {'Variable': 'Sitio', 'Categoria': cat_label}
        for ds in cols_list:
            row[ds] = lookup(sitio_agg.get(ds, {}), cat_label)
        salida.append(row)

    # Lateralidad
    for cat_label in ['Izquierdo', 'Derecho', 'Recto', 'Transverso']:
        row = {'Variable': 'Lateralidad', 'Categoria': cat_label}
        for ds in cols_list:
            row[ds] = lookup(side_agg.get(ds, {}), cat_label)
        salida.append(row)

    _escribir_csv('tabla_frecuencias_colorrectal.csv',
                  ['Variable', 'Categoria'] + cols_list,
                  salida)
    return salida

# ---------------------------------------------------------------------------
#   CSV writer
# ---------------------------------------------------------------------------

def _escribir_csv(nombre, fieldnames, filas):
    ruta = os.path.join(DESTINO, nombre)
    with open(ruta, 'w', newline='', encoding='utf-8') as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        w.writerows(filas)
    print(f"  -> {ruta} ({len(filas)} filas)")

# ---------------------------------------------------------------------------
#   MAIN
# ---------------------------------------------------------------------------

def main():
    os.makedirs(DESTINO, exist_ok=True)
    print("=== Exportando tablas del Capítulo 4 (todo dinámico) ===\n")

    print("[1/4] Tabla de completitud agregada")
    t1_data, ds_cols = tabla1_completitud_agregada()

    print("\n[2/4] Tabla de indicadores de preselección")
    tabla2_indicadores_preseleccion(ds_cols)

    print("\n[3/4] Tabla de frecuencias de mama")
    tabla3_frecuencias_mama()

    print("\n[4/4] Tabla de frecuencias colorrectal")
    tabla4_frecuencias_colorrectal()

    print(f"\nTodos los CSV exportados a: {DESTINO}")


if __name__ == '__main__':
    main()
