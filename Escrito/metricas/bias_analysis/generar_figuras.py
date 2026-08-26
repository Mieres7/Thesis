import json, os, sys
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

RUTA = os.path.dirname(os.path.abspath(__file__))

with open(os.path.join(RUTA, 'resultados_por_dataset.json'), encoding='utf-8') as f:
    datos = json.load(f)

DOMAIN_MAP = {'Mama': 'mama', 'Colorrectal': 'colorrectal'}

SEX_COLORS = {'F': '#4C72B0', 'M': '#DD8452', 'FEMALE': '#4C72B0'}
SEX_LABELS = {'F': 'Femenino', 'M': 'Masculino', 'FEMALE': 'Femenino'}
AGE_LABELS = ['<50', '50-70', '>=70']
AGE_COLORS = ['#66c2a5', '#fc8d62', '#8da0cb']

plt.rcParams.update({
    'font.family': 'sans-serif',
    'font.size': 11,
    'axes.titlesize': 13,
    'axes.labelsize': 12,
    'legend.fontsize': 10,
    'figure.dpi': 200,
})

def fig_sexo(ds_list, dominio_label, filename):
    datasets = []
    categorias = ['Femenino', 'Masculino']
    data = {c: [] for c in categorias}

    for d in ds_list:
        datasets.append(d['dataset'])
        sex = d.get('sex', {})
        total = sex.get('total', 0) or 1
        if 'F_n' in sex:
            data['Femenino'].append(sex['F_n'] / total * 100)
            data['Masculino'].append(sex.get('M_n', 0) / total * 100)
        elif 'FEMALE_n' in sex:
            data['Femenino'].append(sex['FEMALE_n'] / total * 100)
            data['Masculino'].append(0)

    fig, ax = plt.subplots(figsize=(6, 4))
    x = np.arange(len(datasets))
    bar_width = 0.5
    bottom = np.zeros(len(datasets))

    colors = ['#4C72B0', '#DD8452']
    for i, cat in enumerate(categorias):
        bars = ax.bar(x, data[cat], bar_width, label=cat, color=colors[i], bottom=bottom)
        for bar, val in zip(bars, data[cat]):
            if val > 3:
                ax.text(bar.get_x() + bar.get_width() / 2, bar.get_y() + val / 2,
                        f'{val:.1f}%', ha='center', va='center', fontsize=9, color='white', fontweight='bold')
        bottom += data[cat]

    ax.set_xticks(x)
    ax.set_xticklabels(datasets, fontsize=10)
    ax.set_ylabel('Porcentaje (%)')
    ax.set_title(f'Distribución de sexo por cohorte – {dominio_label}', fontweight='bold')
    ax.set_ylim(0, 110)
    ax.legend(loc='upper right', frameon=True)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: f'{y:.0f}%'))
    fig.tight_layout()
    fig.savefig(os.path.join(RUTA, filename))
    plt.close(fig)
    print(f'  -> {filename}')


def fig_edad(ds_list, dominio_label, filename):
    datasets = []
    data = {g: [] for g in AGE_LABELS}

    for d in ds_list:
        datasets.append(d['dataset'])
        ag = d.get('age_group', {})
        for g in AGE_LABELS:
            key = f'{g}_pct'
            data[g].append(ag.get(key, 0))

    fig, ax = plt.subplots(figsize=(7, 4.5))
    x = np.arange(len(datasets))
    n_groups = len(AGE_LABELS)
    bar_width = 0.25

    for i, g in enumerate(AGE_LABELS):
        offset = (i - n_groups / 2 + 0.5) * bar_width
        bars = ax.bar(x + offset, data[g], bar_width, label=g, color=AGE_COLORS[i])
        for bar, val in zip(bars, data[g]):
            if val > 3:
                ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 1,
                        f'{val:.1f}%', ha='center', va='bottom', fontsize=8)

    ax.set_xticks(x)
    ax.set_xticklabels(datasets, fontsize=10)
    ax.set_ylabel('Porcentaje (%)')
    ax.set_title(f'Distribución de grupos etarios por cohorte – {dominio_label}', fontweight='bold')
    ax.set_ylim(0, 70)
    ax.legend(loc='upper right', frameon=True, title='Grupo etario')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: f'{y:.0f}%'))
    fig.tight_layout()
    fig.savefig(os.path.join(RUTA, filename))
    plt.close(fig)
    print(f'  -> {filename}')


mama_ds = [d for d in datos if DOMAIN_MAP.get(d.get('domain', '')) == 'mama'
           and 'Subtipo' not in d['dataset'] and 'Lateralidad' not in d['dataset']]
cr_ds = [d for d in datos if DOMAIN_MAP.get(d.get('domain', '')) == 'colorrectal'
         and 'Subtipo' not in d['dataset'] and 'Lateralidad' not in d['dataset']]

print('=== Generando figuras de sexo (barras 100% apiladas) ===')
fig_sexo(mama_ds, 'Mama', 'fig_sexo_mama.png')
fig_sexo(cr_ds, 'Colorrectal', 'fig_sexo_colorrectal.png')

print('\n=== Generando figuras de edad (barras agrupadas) ===')
fig_edad(mama_ds, 'Mama', 'fig_edad_mama.png')
fig_edad(cr_ds, 'Colorrectal', 'fig_edad_colorrectal.png')

print('\n¡Figuras regeneradas exitosamente!')
