---

license: cc-by-nc-4.0

---
Dear researchers and engineers, you're accessing a dataset that would cost millions of dollars to build and took millions of nerves to negotiate favorable terms for its use. Your support, by liking the repositories and upvoting the collection, costs nothing but gives us valuable motivation to continue our contributions to the community. We reserve the right not to approve the request if you don't support our efforts. Thank you very much for collaboration!

# HISTAI Dataset

HISTAI is a comprehensive whole-slide image (WSI) pathological dataset spanning multiple medical specializations. Slides are anonymized and organized into specialized subsets by organ systems or pathology types.

If you wish to support, sponsor, or obtain a commercial license for HISTAI data, please contact us at [models@hist.ai](mailto:models@hist.ai).

For details refer to our report:

* [HISTAI: An Open-Source, Large-Scale Whole Slide Image Dataset for Computational Pathology](https://arxiv.org/abs/2505.12120)

This repository contains metadata and references to images hosted separately. Individual slide images are accessible from specialized Hugging Face datasets.

---

## Dataset Structure

Slides are stored across specialized datasets hosted on Hugging Face. Each specialized dataset contains anonymized slides organized by cases:

```
histai/<dataset_name>/case_<case_id>/slide_<stain>_<slide_number>.tiff
```
or
```
histai/<dataset_name>/case_<case_id>/slide_<magnification>_<stain>_<slide_number>.tiff
```

Most of the slides are stained with Hematoxylin and Eosin (H&E) and scanned at 20X magnification. If a slide differs in magnification from 20X, this information is embedded in the slide filename, as shown above.

Currently available specialized datasets:

* [HISTAI-hematologic](https://huggingface.co/datasets/histai/HISTAI-hematologic)
* [HISTAI-gastrointestinal](https://huggingface.co/datasets/histai/HISTAI-gastrointestinal)
* [HISTAI-breast](https://huggingface.co/datasets/histai/HISTAI-breast)
* [HISTAI-thorax](https://huggingface.co/datasets/histai/HISTAI-thorax)
* [HISTAI-skin-b2](https://huggingface.co/datasets/histai/HISTAI-skin-b2)
* [HISTAI-skin-b1](https://huggingface.co/datasets/histai/HISTAI-skin-b1)
* [HISTAI-colorectal-b1](https://huggingface.co/datasets/histai/HISTAI-colorectal-b1)
* [HISTAI-colorectal-b2](https://huggingface.co/datasets/histai/HISTAI-colorectal-b2)
* [HISTAI-mixed](https://huggingface.co/datasets/histai/HISTAI-mixed)

## Metadata

The master repository includes comprehensive metadata in JSON format for each slide/case, containing detailed pathological, clinical, and technical information:

| Field             | Description                                  | Example                                                                       |
| ----------------- | -------------------------------------------- | ----------------------------------------------------------------------------- |
| `diagnosis`       | Incoming clinical notes (!NOT THE PATHOLOGIST'S DIAGNOSIS)                      | Benign skin neoplasms.                                                        |
| `conclusion`      | Final pathological conclusion                | Intradermal melanocytic nevus of the skin.                                    |
| `diff_diagnosis`  | Differential diagnostic notes (if available) |                                                                               |
| `micro_protocol`  | Microscopic description                      | Skin: Intradermal melanocytic nevus of the skin. Microscopic description: ... |
| `additional_info` | Any additional clinical/pathological notes   | "A repeat review of the histological specimens was performed, including ...                                                                              |
| `age`             | Patient age (years)                          | 37                                                                            |
| `gender`          | Patient gender                               | f                                                                             |
| `icd10`           | ICD-10 classification                        | D22                                                                           |
| `specialization`  | Medical specialization or organ system       | Skin                                                                          |
| `case_mapping`    | Reference to slide images                    | histai/HISTAI-skin-b2/case\_13384                                             |
| `grossing`        | Gross examination details                    | "Head and neck: One fragment, measuring 2×4 mm, gray, firm, with ...          |

---

## Statistics

| Dataset                                | Total Slides | Total Cases |
|----------------------------------------|-------------:|------------:|
| histai/HISTAI-hematologic              |          214 |         214 |
| histai/HISTAI-gastrointestinal         |          202 |         120 |
| histai/HISTAI-breast                   |        1,925 |       1,692 |
| histai/HISTAI-thorax                   |          829 |         657 |
| histai/HISTAI-skin-b2                  |       43,757 |      20,621 |
| histai/HISTAI-skin-b1                  |        7,710 |       1,778 |
| histai/HISTAI-colorectal-b1            |        5,379 |         998 |
| histai/HISTAI-colorectal-b2            |           94 |          62 |
| histai/HISTAI-mixed                    |       52,691 |      21,137 |

- **Total slides**: 112,801
- **Total cases**: 47,279
- **Slides at x40 magnification**: 2,463
- **Slides at x20 magnifications**: 110,338
- **H&E slides**: 92,536
- **IHC slides**: 16,920
- **Other stains**: 3,345
---

## How to Access Images

You can programmatically download slides using:

### Using Hugging Face Hub

```python
from huggingface_hub import snapshot_download

snapshot_download(repo_id="histai/<dataset_name>", repo_type="dataset", local_dir="/local_path")
```

### Using Git

```bash
# Ensure you have Git LFS installed
git lfs install
git clone https://huggingface.co/datasets/histai/<dataset_name>
```

## License

This dataset is licensed under **CC BY-NC 4.0** and is intended exclusively for **research purposes**.

## Citation

Please cite the following if you use this dataset:

```bibtex
@misc{nechaev2025histaiopensourcelargescaleslide,
      title={HISTAI: An Open-Source, Large-Scale Whole Slide Image Dataset for Computational Pathology}, 
      author={Dmitry Nechaev and Alexey Pchelnikov and Ekaterina Ivanova},
      year={2025},
      eprint={2505.12120},
      archivePrefix={arXiv},
      primaryClass={eess.IV},
      url={https://arxiv.org/abs/2505.12120}, 
}
```

---

## Contacts

* **Authors:** Dmitry Nechaev, Alexey Pchelnikov, Ekaterina Ivanova
* **Emails:** dmitry@hist.ai, alex@hist.ai, kate@hist.ai
