# Justificación de métricas de evaluación para análisis de bias y fairness en IA clínica

## Resumen ejecutivo

El uso de métricas estándar en evaluaciones de fairness y bias en IA médica ha sido objeto de revisiones sistemáticas recientes que evidencian una fragmentación importante en la literatura: no existe consenso único sobre qué métricas usar, y la mayoría de los trabajos adoptan criterios ad hoc sin justificación formal. En ese contexto, la elección de un protocolo unificado basado en **balanced accuracy**, **macro-F1**, **gaps por subgrupo** sobre esas mismas métricas, **caída ID→OOD**, **matriz de confusión** e **intervalos de confianza por bootstrap** está respaldada por evidencia reciente en IA médica, patología computacional y benchmarks de fairness para modelos fundacionales. Este documento fundamenta cada una de esas elecciones con literatura de 2023–2026.[^1][^2]

---

## 1. Contexto: el problema del desbalance y los subgrupos en IA clínica

Los conjuntos de datos de imágenes médicas presentan desequilibrios de clase sistemáticos que afectan negativamente tanto la utilidad global como la equidad entre subgrupos. Cuando se trabaja con tareas de clasificación multiclase —como subtipificación molecular, estado de HER2, localización tumoral o lateralidad— el desbalance puede hacer que métricas como la exactitud (accuracy) global sean engañosas: un modelo puede reportar alta exactitud general mientras falla sistemáticamente en clases minoritarias o en subgrupos demográficos subrepresentados.[^3][^4][^5][^6]

Este fenómeno está documentado en patología digital con modelos fundacionales. Trabajos como el de Lin et al. (2025) con FAIR-Path demostraron que modelos de patología de referencia exhiben disparidades de rendimiento detectables en función del sexo, raza y edad del paciente, con brechas presentes en aproximadamente 29% de las comparaciones diagnósticas. En la misma línea, Khan et al. (2023) encontraron que los modelos fundacionales preentrenados en imágenes médicas logran mejor rendimiento global pero son consistentemente menos equitativos que los preentrenados en imágenes naturales. Estos hallazgos justifican la necesidad de un protocolo de evaluación que mida tanto utilidad como fairness de forma explícita y separada.[^7][^8][^9]

---

## 2. Balanced accuracy como métrica principal de utilidad

### Justificación formal

La **balanced accuracy** se define como la media aritmética de la sensibilidad (recall) por clase:[^10]

\[\text{Balanced Accuracy} = \frac{1}{K} \sum\_{k=1}^{K} \frac{TP_k}{TP_k + FN_k}\]

donde \(K\) es el número de clases. Esta formulación asigna igual peso a cada clase independientemente de su frecuencia en el conjunto de datos, lo que la hace directamente robusta al desbalance de clases sin necesidad de modificar la distribución de entrenamiento.

### Evidencia empírica en IA clínica

Tertytchny et al. (2024) adoptaron la balanced accuracy como métrica primaria en clasificación clínica precisamente por su capacidad de gestionar el desbalance de clases promediando el recall por clase. Dribus et al. (2025) desarrollaron un marco analítico específico para optimizar la balanced accuracy en modelos clínicos con desbalance de clases, argumentando que es la métrica de referencia para datos médicos con distribuciones asimétricas. Una revisión de 2024 sobre métodos para datasets médicos desbalanceados concluye que la balanced accuracy y el macro-F1 son los indicadores más adecuados para representar el rendimiento real en contextos clínicos multiclase con desbalance.[^11][^4][^10]

### Por qué es adecuada para este contexto

En las tareas de esta tesis —HER2 de 3 clases, subtipo molecular de 4 clases, _site_ de alta cardinalidad y _side_ multiclase— la distribución de clases no es uniforme. Usar accuracy estándar favorecería implícitamente a las clases mayoritarias, haciendo invisible el rendimiento en subtipos menos frecuentes como el triple negativo o la localización metastásica. La balanced accuracy corrige esto sin requerir supuestos adicionales y es interpretable directamente: un valor de 0.5 equivale a rendimiento aleatorio uniforme entre clases.

---

## 3. Macro-F1 como segunda métrica principal

### Justificación formal

El **macro-F1** calcula el F1 (media armónica de precisión y recall) por clase y luego promedia sin ponderar por frecuencia:

\[\text{Macro-F1} = \frac{1}{K} \sum*{k=1}^{K} F1_k = \frac{1}{K} \sum*{k=1}^{K} \frac{2 \cdot P_k \cdot R_k}{P_k + R_k}\]

A diferencia de la balanced accuracy, el F1 penaliza también la precisión baja, capturando falsos positivos además de falsos negativos.

### Evidencia empírica en IA clínica y fairness

El macro-F1 ha sido adoptado como métrica de fairness por subgrupo en estudios recientes de IA clínica. Trabajos sobre predicción temprana de enfermedades cardíacas usan el gap de F1 entre hombres y mujeres (\(\Delta F1\)) como medida directa de disparidad de equidad. Una revisión sistemática sobre datos médicos desbalanceados en clasificación multiclase concluye que el F1-score es la métrica más informativa en escenarios con clases desequilibradas, mientras que la exactitud es la menos informativa. FairMedFM, el benchmark de fairness para modelos fundacionales en imágenes médicas más completo publicado hasta 2024, utiliza métricas de rendimiento por subgrupo —incluyendo variantes del F1— como base para medir disparidades entre grupos demográficos.[^12][^13][^14][^15]

### Complementariedad con balanced accuracy

La balanced accuracy y el macro-F1 no miden exactamente lo mismo: la primera se basa solo en recall, la segunda incorpora precisión. Usarlas juntas permite detectar patrones distintos: un modelo puede tener buena balanced accuracy pero bajo macro-F1 si tiene muchos falsos positivos en clases minoritarias. Esta complementariedad está respaldada por literatura de evaluación multiclase que recomienda reportar ambas métricas en tareas con desbalance.[^4][^16]

---

## 4. Gap máximo entre subgrupos como medida de fairness

### Justificación conceptual

La medición de brechas de rendimiento entre subgrupos es el enfoque más directo para operacionalizar fairness de grupo (_group fairness_) en clasificación. Chen et al. (2023) definen formalmente el gap de fairness como la diferencia máxima de rendimiento entre el subgrupo con mejor desempeño y el de peor desempeño en una métrica dada:[^17][^6]

\[\Delta M = \max*{g \in G} M(g) - \min*{g \in G} M(g)\]

donde \(M\) es la métrica de rendimiento y \(G\) el conjunto de subgrupos protegidos.

### Evidencia de uso en patología y modelos fundacionales

FLEX (Huang et al., 2025), un framework publicado en _Nature Communications_ para mejorar la generalización y fairness de modelos fundacionales de patología, utiliza explícitamente el gap de rendimiento entre grupos demográficos como indicador principal de equidad, mostrando su reducción como evidencia de mejora. FAIR-Path (Lin et al., 2025) reporta la reducción del 91.1% en brechas de rendimiento entre cohortes externas como métrica central de fairness. Obermeyer et al. (2024) en _Nature Medicine_ demuestran que modelos con alta AUC global pueden enmascarar fallas graves en subgrupos específicos, justificando la necesidad de reportar brechas explícitas.[^18][^5][^19][^9][^20][^7]

### Por qué usar el gap sobre balanced accuracy y macro-F1

Usar el **mismo par de métricas** para medir utilidad global y brechas de fairness garantiza coherencia metodológica. Si balanced accuracy y macro-F1 son las métricas de utilidad, sus gaps entre subgrupos son la extensión natural para fairness, sin necesidad de adoptar métricas adicionales como Equalized Odds (que requieren adaptaciones no triviales en escenarios multiclase). Rountree et al. (2025) señalan que la práctica dominante en IA clínica consiste en comparar métricas de discriminación entre subgrupos, aunque con poca adopción formal de métricas de fairness per se. Esta tesis adopta la comparación sistemática de gaps como solución directa, práctica y justificada.[^2]

---

## 5. Caída ID→OOD como medida de generalización y robustez

### Justificación conceptual

La hipótesis central de esta tesis es que el bias en los datos afecta la capacidad de generalización de modelos fundacionales cuando se cambia de distribución. Evaluar la caída de rendimiento entre escenarios _in-distribution_ (ID) e _out-of-distribution_ (OOD) es el mecanismo estándar para cuantificar este impacto.[^21][^22]

La caída absoluta se define como:

\[\Delta*{OOD} = M*{ID} - M\_{OOD}\]

donde una caída mayor indica menor robustez ante cambios de distribución, que en este contexto corresponden a cambios de cohorte o dataset.

### Evidencia empírica en patología digital

FLEX (Huang et al., 2025) evalúa explícitamente el gap de rendimiento entre sitios conocidos y externos como indicador de generalización, demostrando que los modelos fundacionales de patología muestran caídas sustanciales en datos de nuevos centros debido a firmas específicas del sitio. Benchmarks como OpenMIBOOD (CVPR 2025) evidencian que los métodos de detección OOD calibrados en imágenes naturales no se trasladan directamente a imágenes médicas, subrayando la importancia de evaluaciones OOD específicas del dominio médico. Ju et al. (2025) demuestran que los modelos de visión-lenguaje médicos tienen capacidades OOD sub-exploradas, reforzando la relevancia de incluir evaluación OOD sistemática. Chen et al. (2023) enmarcan los cambios de distribución como una fuente central de inequidad algorítmica en IA médica, porque los sesgos del conjunto de entrenamiento se amplifican cuando el modelo se despliega en poblaciones distintas.[^22][^23][^19][^20][^17]

### Por qué es esencial para esta tesis

El escenario experimental propuesto (entrenar en dataset A y evaluar en dataset B) es precisamente un escenario OOD cruzado entre cohortes. Medir la caída absoluta en balanced accuracy y macro-F1 entre ese escenario y el in-distribution es la operacionalización más directa de la hipótesis de investigación: si el bias de dataset impacta la generalización, esa caída será mayor en subgrupos subrepresentados.

---

## 6. Matriz de confusión como análisis de errores por clase

### Justificación

La matriz de confusión no es una métrica escalar sino una representación bidimensional completa del comportamiento del clasificador por clase. En tareas multiclase con clases heterogéneas —como _site_ que incluye desde colon derecho hasta hígado metastásico— la matriz de confusión revela patrones que ninguna métrica escalar puede capturar: qué clases se confunden entre sí, dónde se concentran los errores sistemáticos y si los errores se distribuyen uniformemente o se concentran en subgrupos.[^24]

### Evidencia de uso en fairness clínica

El sistema ShinAI-Agent (2025), diseñado específicamente para monitoreo de fairness y robustez en IA clínica, adopta matrices de confusión a nivel de caso (_case-based confusion matrices_) como base de su arquitectura de base de datos, precisamente porque las métricas de fairness derivadas de la matriz de confusión son no aditivas y no pueden desagregarse de forma fiable sin acceso a la matriz completa. El análisis de errores por subgrupo está recomendado como complemento necesario a las métricas escalares de fairness para entender la naturaleza clínica de las disparidades, no solo su magnitud numérica.[^6][^24][^1]

---

## 7. Intervalos de confianza por bootstrap

### Justificación estadística

Los intervalos de confianza por bootstrap son el estándar no paramétrico para cuantificar incertidumbre en métricas de clasificación cuando el tamaño de muestra es finito y la distribución de la métrica es desconocida. El procedimiento estándar consiste en remuestrear con reemplazo el conjunto de test un número elevado de veces (típicamente 1000–2000), calcular la métrica en cada muestra y derivar los percentiles 2.5 y 97.5 como límites del IC al 95%.[^25][^26]

### Evidencia de uso en fairness clínica

Riley et al. (2025) en _BMJ_ establecen que el bootstrap es el método de referencia para cuantificar incertidumbre en modelos de predicción clínica porque puede aplicarse a cualquier método de modelado sin requerir supuestos distribucionales. En el campo específico de fairness médica, el paper de LLMs en cuidados críticos (NeurIPS 2024) reporta balanced accuracy e intervalos de bootstrap por modelo: modelos como Mistral-Nemo-Instruct-2407 mostraron F1 de 0.84 ± 0.05 y balanced accuracy de 0.84 ± 0.04, con modelos médicos específicos exhibiendo intervalos más amplios, lo que indica menor consistencia. Matos et al. (2025/2026) identifican la cuantificación de incertidumbre como una de las brechas principales en el campo de fairness en IA clínica: la práctica dominante es reportar estimaciones puntuales, sin IC, lo que limita la capacidad de concluir sobre diferencias reales entre grupos.[^27][^28][^26][^1]

### Por qué son esenciales en esta tesis

En evaluaciones por subgrupo, los tamaños de muestra suelen ser más pequeños que en el conjunto global, lo que aumenta la varianza de las métricas. Sin intervalos de confianza, una brecha de fairness de, por ejemplo, 5 puntos en balanced accuracy entre grupos podría ser estadísticamente indistinguible de cero en cohortes pequeñas. Los IC por bootstrap son la forma más práctica de evitar conclusiones falsas sobre bias a partir de fluctuaciones muestrales.

---

## 8. Coherencia del protocolo unificado

### Justificación global

La literatura reciente de fairness en IA médica converge en recomendar protocolos de evaluación unificados que cubran simultáneamente utilidad global, disparidades por subgrupo y generalización OOD. El protocolo propuesto en esta tesis aplica la misma familia de métricas (balanced accuracy + macro-F1) en tres niveles: rendimiento global, gaps entre subgrupos y caída OOD. Esto garantiza comparabilidad directa entre tareas, entre modelos y entre escenarios de distribución, sin necesidad de adoptar métricas distintas para cada dominio.[^29][^30][^13]

Una revisión sistemática de 2025 sobre literatura de fairness en IA clínica (Liu et al., _NPJ Digital Medicine_) identifica que la fragmentación de métricas —usar AUC en un trabajo, accuracy en otro, F1 en un tercero— dificulta la comparación entre estudios y limita el avance del campo. Adoptar balanced accuracy y macro-F1 como métricas unificadas es una decisión metodológicamente sólida y alineada con las recomendaciones emergentes de estandarización.[^6]

### Tabla resumen de justificaciones

| Métrica                      | Función en el protocolo            | Justificación principal                                  | Referencias clave |
| ---------------------------- | ---------------------------------- | -------------------------------------------------------- | ----------------- |
| Balanced accuracy            | Utilidad global                    | Robustez al desbalance multiclase; igual peso por clase  | [^11][^10][^4]    |
| Macro-F1                     | Utilidad global (complementaria)   | Incorpora precisión y recall; informativa en desbalance  | [^12][^15][^13]   |
| Gap en balanced accuracy     | Fairness de grupo                  | Diferencia máxima entre subgrupos en métrica de utilidad | [^17][^19][^18]   |
| Gap en macro-F1              | Fairness de grupo (complementario) | Misma lógica que gap en balanced accuracy                | [^12][^7][^6]     |
| Caída ID→OOD (balanced acc.) | Generalización                     | Cuantifica impacto de cambio de distribución en utilidad | [^19][^21][^22]   |
| Caída ID→OOD (macro-F1)      | Generalización (complementaria)    | Coherencia con métricas de utilidad elegidas             | [^20][^23]        |
| Matriz de confusión          | Análisis de errores                | Revela patrones de confusión por clase y subgrupo        | [^24][^1]         |
| IC por bootstrap             | Incertidumbre estadística          | Estándar no paramétrico para IC en métricas clínicas     | [^26][^25][^27]   |

---

## 9. Prompt de implementación en Python

El siguiente prompt describe los requerimientos de implementación del protocolo de evaluación unificado. Está redactado de forma neutral a la arquitectura para que pueda adaptarse a cualquier pipeline de clasificación WSI.

---

```
Necesito implementar un módulo de evaluación unificado en Python para tareas de clasificación multiclase (3–10 clases).
El módulo debe recibir como entrada las predicciones de un modelo sobre un conjunto de test y retornar
un reporte estructurado con las siguientes métricas:

[UTILIDAD GLOBAL]
- Balanced accuracy (media aritmética del recall por clase, igual peso a todas las clases)
- Macro-F1 (F1 por clase promediado sin ponderar por frecuencia)
- Macro-AUC one-vs-rest (opcional, solo si todas las clases tienen suficiente soporte)
- Matriz de confusión normalizada por fila y sin normalizar

[FAIRNESS POR SUBGRUPO]
El módulo debe recibir un vector de atributos de subgrupo (ej: sexo o grupo de edad).
Para cada subgrupo definido debe:
- Calcular balanced accuracy y macro-F1 por subgrupo individualmente
- Calcular el gap máximo entre subgrupos para cada métrica: max(M_grupo) - min(M_grupo)
- Calcular el gap promedio entre subgrupos para cada métrica

[GENERALIZACIÓN OOD]
El módulo debe poder recibir dos conjuntos de predicciones: uno in-distribution (ID) y uno out-of-distribution (OOD).
Para cada par ID/OOD debe:
- Calcular balanced accuracy y macro-F1 en ambos escenarios
- Calcular la caída absoluta: M_ID - M_OOD
- Calcular la caída relativa porcentual: (M_ID - M_OOD) / M_ID * 100

[INTERVALOS DE CONFIANZA]
Todas las métricas anteriores (utilidad global, por subgrupo y OOD) deben acompañarse de
intervalos de confianza al 95% calculados por bootstrap no paramétrico (mínimo 1000 iteraciones,
remuestreo con reemplazo a nivel de muestra/paciente).
Reportar como: valor_medio ± error, o [limite_inferior, limite_superior].

[SALIDA]
El módulo debe entregar:
- Un DataFrame estructurado con todas las métricas, sus valores puntuales y sus IC
- Una figura con la matriz de confusión (normalizada y absoluta)
- Un resumen en texto plano con los valores principales para incluir en paper

[REQUISITOS TÉCNICOS]
- Implementar con scikit-learn para las métricas base
- Implementar el bootstrap con numpy (remuestreo de índices)
- El módulo debe funcionar independientemente del modelo que genera las predicciones
  (recibe y_true, y_pred, y_prob como arrays de numpy o listas)
- Debe manejar clases con 0 muestras en un subgrupo (reportar NaN en vez de fallar)
- Debe funcionar tanto para evaluación binaria como multiclase
```

---

## 10. Limitaciones y consideraciones finales

Aunque el protocolo propuesto está respaldado por literatura reciente, existen limitaciones conocidas que deben mencionarse en la tesis. La balanced accuracy y el macro-F1 no satisfacen todas las definiciones formales de fairness simultáneamente; como demuestran Chen et al. (2023), muchas definiciones de fairness son matemáticamente incompatibles entre sí, por lo que elegir métricas basadas en rendimiento por subgrupo implica una decisión informada sobre qué tipo de equidad se prioriza. La cuantificación de incertidumbre por bootstrap asume que el conjunto de test es una muestra representativa de la distribución de interés, supuesto que puede violarse con datasets pequeños o muy sesgados. Finalmente, Matos et al. (2025/2026) señalan que el campo carece de métricas de fairness con validación clínica directa, por lo que las métricas propuestas deben entenderse como indicadores metodológicos, no como garantías de equidad clínica en sentido pleno.[^28][^26][^17][^1]

---

## References

1. [Critical Appraisal of Fairness Metrics in Clinical Predictive AI](http://www.arxiv.org/abs/2506.17035) - Predictive artificial intelligence (AI) offers an opportunity to improve clinical practice and patie...

2. [Reporting of Fairness Metrics in Clinical Risk Prediction ... - PMC](https://pmc.ncbi.nlm.nih.gov/articles/PMC11966066/) - by L Rountree · 2025 · Cited by 12 — Our review shows that the use of fairness metrics for evaluatin...

3. [Imbalanced class distribution and performance evaluation metrics](https://pmc.ncbi.nlm.nih.gov/articles/PMC10688675/) - Focus on predictive algorithm and its performance evaluation is extensively covered in most research...

4. [Handling imbalanced medical datasets: review of a decade ...](https://d-nb.info/1350802794/34)

5. [From AUC to Accountability: Metric Choices, Social Norms, and the...](https://openreview.net/forum?id=NTz5sXVTJn) - Clinical and regulatory discussions about trustworthy therapeutic AI speak in ethical and legal term...

6. [A scoping review and evidence gap analysis of clinical AI ...](https://pmc.ncbi.nlm.nih.gov/articles/PMC12167363/) - by M Liu · 2025 · Cited by 48 — We highlight the scarcity of AI fairness research in medical domains...

7. [Article Contrastive learning enhances fairness in pathology ...](https://www.sciencedirect.com/science/article/pii/S2666379125006007) - by SY Lin · 2025 · Cited by 10 — FAIR-Path effectively mitigates 88.5% of these disparities, with ex...

8. [How Fair are Medical Imaging Foundation Models?](https://proceedings.mlr.press/v225/khan23a.html) - While fine-tuning on balanced datasets partially mitigates fairness issues, it is insufficient to co...

9. [Contrastive learning enhances fairness in pathology ...](https://pubmed.ncbi.nlm.nih.gov/41406939/?fc=20250306132545&ff=20251225090041&v=2.18.0.post22+67771e2) - AI-enhanced pathology evaluation systems hold significant potential to improve cancer diagnosis but ...

10. [arXiv:2412.13439v3 [cs.LG] 31 Jan 2025](https://arxiv.org/pdf/2412.13439.pdf) - by G Tertytchny · 2024 · Cited by 5 — Specifically, balanced accuracy served as our primary metric d...

11. [Optimizing balanced accuracy in medical data threshold ...](https://d-nb.info/137048576X/34)

12. [Table 4. Fairness evaluation.](https://pmc.ncbi.nlm.nih.gov/articles/PMC12596549/table/T4/) - Early and accurate prediction of Ischemic Heart Disease (IHD) is critical to reducing cardiovascular...

13. [FairMedFM: Fairness Benchmarking for Medical Imaging ...](https://neurips.cc/virtual/2024/poster/97809) - We introduce FairMedFM, a fairness benchmark for FM research in medical imaging. FairMedFM integrate...

14. [[PDF] FairMedFM: Fairness Benchmarking for Medical Imaging Foundation Models | Semantic Scholar](https://www.semanticscholar.org/paper/FairMedFM:-Fairness-Benchmarking-for-Medical-Models-Jin-Xu/52dd767d8b13f672bdb9676db4d2a324ac6d1163) - This exhaustive analysis evaluates the fairness performance over different evaluation metrics from m...

15. [2023.Comparative Analysis Using Various Performance Metrics in ...](https://www.scribd.com/document/937684906/2023-Comparative-Analysis-Using-Various-Performance-Metrics-in-Imbalanced-Data-for-Multi-class-Text-Classification) - This study conducts a comparative analysis of various performance metrics, including Precision, Reca...

16. [Classification performance assessment for imbalanced multiclass data - Scientific Reports](https://www.nature.com/articles/s41598-024-61365-z) - The evaluation of diagnostic systems is pivotal for ensuring the deployment of high-quality solution...

17. [Algorithm fairness in artificial intelligence for medicine ... - PMC](https://pmc.ncbi.nlm.nih.gov/articles/PMC10632090/) - by RJ Chen · 2023 · Cited by 904 — In this Perspective, we outline fairness in machine learning thro...

18. [The limits of fair medical imaging AI in real-world generalization](https://www.nature.com/articles/s41591-024-03113-4) - When tested across tasks, diseases and imaging modalities, the performance of AI models depends on e...

19. [Knowledge-guided adaptation of pathology foundation models effectively improves cross-domain generalization and demographic fairness](https://www.nature.com/articles/s41467-025-66300-y) - Pathology foundation models can still struggle with generalizability and fairness across demographic...

20. [Knowledge-guided adaptation of pathology foundation models effectively improves cross-domain generalization and demographic fairness - PubMed](https://pubmed.ncbi.nlm.nih.gov/41387953/) - Foundation models in computational pathology suffer from site-specific and demographic biases, which...

21. [Out-of-distribution Detection in Medical Image Analysis: A survey](https://arxiv.org/abs/2404.18279) - Computer-aided diagnostics has benefited from the development of deep learning-based computer vision...

22. [OpenMIBOOD: Open Medical Imaging Benchmarks for Out- ...](https://cvpr.thecvf.com/virtual/2025/poster/33159)

23. [Delving into Out-of-Distribution Detection with Medical ...](https://arxiv.org/abs/2503.01020) - Recent advances in medical vision-language models (VLMs) demonstrate impressive performance in image...

24. [Towards an Analytical System for Supervising Fairness, Robustness, and Dataset Shifts in Health AI - PubMed](https://pubmed.ncbi.nlm.nih.gov/41041783/) - Ensuring trustworthy use of Artificial Intelligence (AI)-based Clinical Decision Support Systems (CD...

25. [Quantifying Uncertainty: All We Need is the Bootstrap?](https://arxiv.org/html/2403.20182v2) - In this paper, we focus on the quantification of uncertainty. Standard errors, confidence intervals,...

26. [Uncertainty of risk estimates from clinical prediction models](https://www.bmj.com/content/388/bmj-2024-080749) - by RD Riley · 2025 · Cited by 69 — The bootstrap approach has a key advantage of being able to be ap...

27. [Reliability in AI-Assisted Critical Care - NeurIPS 2025](https://neurips.cc/virtual/2024/103872)

28. [Critical appraisal of fairness metrics for artificial intelligence ...](<https://www.thelancet.com/journals/landig/article/PIIS2589-7500(26)00024-5/fulltext>) - by J Matos · Cited by 8 — We conducted a scoping review to identify and critically appraise fairness...

29. [RISED: A Pre-Deployment Safety Evaluation Framework ...](https://arxiv.org/html/2605.12895v1) - We propose the RISED Framework: a five-dimension pre-deployment evaluation covering Reliability, Inc...

30. [[PDF] Advancing Fairness in Clinical AI Decision-Making through a ...](https://www.scitepress.org/Papers/2026/143043/143043.pdf)
