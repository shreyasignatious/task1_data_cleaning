#  Titanic Data Cleaning & Preprocessing

This project is part of an AI & ML Internship Task — focused on data cleaning and preprocessing of the Titanic dataset for machine learning.



##  Objective

To clean and preprocess raw Titanic dataset by:
- Handling missing values
- Encoding categorical features
- Feature scaling
- Removing outliers



##  Tools Used

- Python 
- Pandas
- NumPy
- Scikit-learn
- Seaborn & Matplotlib (for visualization)



##  Dataset

**Source:** [Kaggle Titanic Dataset](https://www.kaggle.com/datasets/yasserh/titanic-dataset)

- Original file: `train.csv` (inside `/data`)
- Cleaned file: `cleaned_titanic.csv` (auto-generated after script runs)


##  Project Workflow

1. Load dataset
2. Explore data: missing values, data types, basic stats
3. Handle missing values (`Age`, `Embarked`, `Cabin`)
4. Encode categorical columns: `Sex`, `Embarked`
5. Normalize `Age` and `Fare`
6. Remove outliers using quantile threshold
7. Save cleaned dataset to `/data/cleaned_titanic.csv`


##  How to Run

```bash
cd TITANIC_TASK-1_Project
python titanic_cleaning.py
