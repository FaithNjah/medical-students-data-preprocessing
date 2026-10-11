# Medical Students Data Preprocessing

A complete data-cleaning and feature-engineering pipeline in Python that turns a raw, incomplete health dataset into a model-ready table.

![Python](https://img.shields.io/badge/Python-3.10+-3776AB?logo=python&logoColor=white)
![pandas](https://img.shields.io/badge/pandas-150458?logo=pandas&logoColor=white)
![scikit-learn](https://img.shields.io/badge/scikit--learn-F7931E?logo=scikitlearn&logoColor=white)
![Jupyter](https://img.shields.io/badge/Jupyter-F37626?logo=jupyter&logoColor=white)

## Overview

Real-world health data is rarely clean. This project works through the [Medical Students Dataset](https://www.kaggle.com/datasets/slmsshk/medical-students-dataset) from Kaggle, which records age, gender, height, weight, blood type, BMI, temperature, heart rate, blood pressure, cholesterol, diabetes and smoking status for medical students. It comes with missing values in every column, duplicate records and an unreliable BMI column.

The project covers the full preprocessing workflow:

| Step | Technique |
|---|---|
| Missing values | Row removal for missing IDs; median and mode imputation |
| Duplicates | Exact-row and repeated-ID de-duplication |
| Outliers | IQR detection, capping at the fences |
| Data types | Integer conversion for ID and age |
| Feature engineering | BMI recomputed from height and weight; WHO BMI categories |
| Encoding | Binary mapping, one-hot and ordinal encoding chosen per feature type |
| Scaling | Z-score standardisation compared with min-max normalisation |
| Visualisation | Box plots, distributions, correlation heatmap |

It started as a data-preprocessing assignment during my data science training (2024). I later revisited it, fixed the bugs, completed the unfinished sections and organised it into a reusable project.

## Walkthrough

The notebook [`notebooks/medical_students_preprocessing.ipynb`](notebooks/medical_students_preprocessing.ipynb) explains each step and the reasoning behind it.

### 1. Explore and classify features
Load the data, inspect its shape and types, and classify each column as categorical (nominal or ordinal) or numerical (discrete or continuous). This decides how each one is cleaned and encoded later.

### 2. Handle missing values
Every column has missing values, and they fall in different rows. Dropping every incomplete row would discard a large part of the dataset, so the pipeline:
- drops only rows with no Student ID, since those can't be tracked or de-duplicated;
- fills numeric columns with the **median**, which is robust to outliers;
- fills categorical columns with the **mode**.

### 3. Remove duplicates
Each student should appear once. The pipeline removes exact duplicate rows and repeated Student IDs, keeping the first record.

### 4. Detect and handle outliers
Box plots and the **IQR rule** (below Q1 − 1.5×IQR or above Q3 + 1.5×IQR) flag extreme values, and these are checked against realistic ranges. Outliers are **capped** rather than deleted, so the rest of that student's record is kept.

### 5. Feature engineering
- **BMI_Computed** = weight (kg) / height (m)². The result is compared with the dataset's own BMI column to check whether that column is consistent.
- **BMI_Category** groups the result into the WHO categories (Underweight, Normal, Overweight, Obese).
- Mean age, height, weight and BMI are compared by gender.

### 6. Encode categorical variables
| Feature | Method | Reason |
|---|---|---|
| Diabetes, Smoking | Binary (No → 0, Yes → 1) | Two values only |
| Gender, Blood Type | One-hot | No natural order |
| BMI_Category | Ordinal | Has a real order |

### 7. Scale features
Both **z-score** and **min-max** scaling are applied and compared. Z-score is kept for the final output.

## Project structure

```
medical-students-data-preprocessing/
├── data/
│   └── README.md            # how to download the dataset
├── figures/                 # charts saved by the notebook
├── notebooks/
│   └── medical_students_preprocessing.ipynb   # step-by-step walkthrough
├── src/
│   └── preprocessing.py     # the same pipeline as a reusable script
├── requirements.txt
└── README.md
```

## Getting started

```bash
git clone https://github.com/FaithNjah/medical-students-data-preprocessing.git
cd medical-students-data-preprocessing
pip install -r requirements.txt
```

1. Download `medical_students_dataset.csv` from [Kaggle](https://www.kaggle.com/datasets/slmsshk/medical-students-dataset) and place it in `data/` (see [`data/README.md`](data/README.md)).
2. Explore the walkthrough:
   ```bash
   jupyter notebook notebooks/medical_students_preprocessing.ipynb
   ```
3. Or run the whole pipeline in one command:
   ```bash
   python src/preprocessing.py --input data/medical_students_dataset.csv --output data/medical_students_clean.csv
   # use --scale minmax for min-max normalisation instead
   ```

## Skills demonstrated

- Data cleaning and validation with **pandas**
- Statistical outlier detection (IQR)
- Feature engineering informed by domain knowledge (BMI, WHO categories)
- Choosing an encoding strategy for each feature type
- Feature scaling with **scikit-learn**
- Exploratory visualisation with **matplotlib** and **seaborn**
- Turning notebook code into a reusable, documented Python module

## Author

**Faith Njah**: software developer with a background in Medical Physiology.
