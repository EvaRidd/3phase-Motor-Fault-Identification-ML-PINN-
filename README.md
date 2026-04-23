
#  Electric Drive Fault Identification using ML & PINN

[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/)
[![License](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)

## About the project

This repository contains the research and implementation of **machine learning** and **Physics-Informed Neural Networks (PINNs)** for the identification of electrical drive faults.

**Key aspects:**
- Comparative analysis of **KNN**, **Random Forest**, and **SVM**
- Physics-informed approach with regime separation (`start`, `steady`, `fault`)
- Implementation of a **PINN** for dynamic modeling of induction motors
- Feature importance analysis and multi-class ROC evaluation

The main goal is to accurately detect and classify faults such as:
- Phase-to-ground short circuits
- Phase-to-phase short circuits
- Phase loss (open circuit)

##  Dataset & Files

The original data files (CSV), additional Python scripts, and the complete Jupyter notebook are available on Google Drive:

🔗 **[Download the full project data from Google Drive](https://drive.google.com/drive/folders/16hqy2Okze8RiGP4_0PRv6d6OIlOj0E1q?usp=drive_link)**

>  Please use the link above to access all motor data and auxiliary scripts.

## Project Report

A detailed PDF report describing the methodology, experiments, and results is included **inside this repository**:

📁 [`Идентификация_неисправностей_с_применением_МО_ПРОЕКТ (1).pdf`](./Идентификация_неисправностей_с_применением_МО_ПРОЕКТ%20(1).pdf)

> You can view or download it directly from the repository.

##  How to use

1. Clone this repository:
   ```bash
   git clone https://github.com/EvaRidd/your-repo-name.git
   ```
2. Download the data from the [Google Drive link](https://drive.google.com/drive/folders/16hqy2Okze8RiGP4_0PRv6d6OIlOj0E1q?usp=drive_link) and place the `.csv` files in the root folder.
3. Open and run the Jupyter notebook:
   ```bash
   jupyter notebook "Проект практика лето 2025.ipynb"
   ```

## Repository structure

```
.
├── Проект практика лето 2025.ipynb   # Main Jupyter notebook
├── Идентификация_неисправностей_с_применением_МО_ПРОЕКТ (1).pdf  # Full report
├── ML_code_0.py
├── ML_code_1.py
├── PINN_models.py
├── ТОКИ.py
└── README.md
```

##  Key results

- **Steady & fault regimes**: near-perfect classification (accuracy 1.00) with all models
- **Start-up regime**: limited performance (best F1 ~0.76 for phase faults)
- **PINN**: minor improvement in dynamic regimes, redundant in steady/fault due to already perfect baseline

## References

- Raissi, M., Perdikaris, P., Karniadakis, G.E. (2019). Physics-informed neural networks.
- Mitchell, T. (2020). Machine Learning.

## Contact

Eva Ridd – [GitHub Profile](https://github.com/EvaRidd)

Project for summer practice 2025, ITMO University.
