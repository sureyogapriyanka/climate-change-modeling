# 🌍 Climate Change Social Media Engagement Analysis  
### *Analyzing Public Interaction with Climate Change Content on Social Media Using Machine Learning*

[![Python](https://img.shields.io/badge/python-3.8%2B-blue)](https://www.python.org/downloads/)
[![License](https://img.shields.io/badge/license-MIT-green)](LICENSE)
[![Jupyter](https://img.shields.io/badge/Jupyter-Lab-orange)](https://jupyter.org/)

---

## 📘 Project Overview

This project analyzes social media engagement (likes, comments, and posting behavior) on posts related to climate change. Using a dataset containing text content, user profiles, and engagement metrics, we explore posting patterns, extract insights from text data, and model engagement levels using machine learning techniques.

The analysis aims to understand public sentiment and engagement patterns around climate change discussions on social media platforms, which can inform policy decisions, advocacy efforts, and communication strategies.

---

## 🔍 Key Findings

Based on our exploratory data analysis:

- **Engagement Patterns**: Climate change posts show varying engagement levels throughout the year, with peaks during environmental awareness events
- **Content Analysis**: Posts containing specific keywords related to solutions and actions tend to receive higher engagement
- **User Behavior**: Certain user profiles consistently generate higher engagement than others
- **Temporal Trends**: Engagement levels have increased over time, indicating growing public interest in climate issues

---

## 🎯 Objectives

1. **Explore** trends in climate-related social media discussions
2. **Preprocess & clean** data (handle missing values, encode profiles, extract time-based features)
3. **Transform text** into numerical features using TF-IDF vectorization
4. **Train & evaluate** machine learning models to predict engagement metrics
5. **Visualize results** and export a final analytical report

---

## 📁 Project Structure

```
climate_change_modeling/
├── data/
│   └── raw/
│       └── climate_nasa.csv         # Raw dataset
├── notebooks/
│   ├── 01-setup.ipynb               # Data loading and initial exploration
│   ├── 02-eda.ipynb                 # Exploratory data analysis
│   ├── 03-preprocessing.ipynb       # Data cleaning and feature engineering
│   ├── 04-modeling.ipynb            # Model training and evaluation
│   └── 05-report.ipynb              # Final analysis and reporting
├── src/
│   ├── data.py                      # Data loading and cleaning functions
│   ├── features.py                  # Feature engineering functions
│   └── models.py                    # Model training and evaluation functions
├── reports/
│   ├── figures/                     # Generated plots and visualizations
│   ├── tables/                      # Model results and metrics
│   └── pdf/                         # Exported reports
├── docs/
│   ├── usage.md                     # Module usage guide
│   └── project_summary.md           # Project overview
├── tests/                           # Unit tests (currently empty)
├── run_analysis.py                  # Script to run complete analysis
├── requirements.txt                 # Python dependencies
├── setup.py                         # Package setup file
├── .gitignore                       # Git ignore file
├── LICENSE                          # License information
├── CONTRIBUTING.md                  # Contribution guidelines
├── CODE_OF_CONDUCT.md               # Code of conduct
└── README.md                        # This file
```

---

## 📊 Dataset

The project uses a dataset of social media posts related to climate change, specifically focusing on interactions with NASA's climate content. The dataset includes:

- **date**: Timestamp of the post
- **likesCount**: Number of likes received
- **profileName**: User profile identifier
- **commentsCount**: Number of comments received
- **text**: Text content of the post/comment

The raw dataset is stored in [data/raw/climate_nasa.csv](data/raw/climate_nasa.csv).

---

## ⚙️ Installation

1. Clone this repository:
   ```bash
   git clone https://github.com/yourusername/climate-change-modeling.git
   cd climate-change-modeling
   ```

2. Create a virtual environment (recommended):
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. Install required packages:
   ```bash
   pip install -r requirements.txt
   ```

---

## ▶️ Usage

### Running the Complete Analysis

You can run the entire analysis pipeline with a single command:

```bash
python run_analysis.py
```

### Step-by-Step Analysis with Jupyter Notebooks

The project is organized as a series of Jupyter notebooks that should be run in sequence:

1. [01-setup.ipynb](notebooks/01-setup.ipynb) - Load and inspect the dataset
2. [02-eda.ipynb](notebooks/02-eda.ipynb) - Perform exploratory data analysis
3. [03-preprocessing.ipynb](notebooks/03-preprocessing.ipynb) - Clean and preprocess the data
4. [04-modeling.ipynb](notebooks/04-modeling.ipynb) - Train and evaluate machine learning models
5. [05-report.ipynb](notebooks/05-report.ipynb) - Generate final visualizations and report

To run the notebooks:
```bash
jupyter lab
```

Or run them individually:
```bash
jupyter notebook notebooks/01-setup.ipynb
```

### Using the Python Modules

The project includes reusable Python modules in the [src/](src/) directory:

```python
from src.data import load_data, clean_data
from src.features import extract_date_features, build_feature_matrix
from src.models import train_and_evaluate
```

See [docs/usage.md](docs/usage.md) for detailed usage instructions.

---

## 🧠 Methodology

1. **Data Preprocessing**:
   - Handle missing values in text and numerical columns
   - Extract temporal features (year, month, day, weekday)
   - Encode categorical variables (user profiles)

2. **Feature Engineering**:
   - Apply TF-IDF vectorization to text data (500 most important terms)
   - Combine text features with metadata features

3. **Modeling**:
   - Linear Regression (baseline)
   - Random Forest Regressor
   - Gradient Boosting Regressor

4. **Evaluation**:
   - Mean Absolute Error (MAE)
   - Root Mean Square Error (RMSE)
   - R-squared (R²) coefficient of determination

---

## 📈 Results

The models are evaluated on their ability to predict engagement (likes count) based on post content and metadata. Performance metrics are saved to [reports/tables/model_results.csv](reports/tables/model_results.csv) and visualized in [reports/figures/model_r2.png](reports/figures/model_r2.png).

Typically, the Gradient Boosting model performs best, followed by Random Forest and Linear Regression.

---

## 🛠️ Requirements

- Python 3.8+
- Jupyter Notebook/Lab
- pandas
- numpy
- scikit-learn
- matplotlib
- seaborn
- scipy

Install all dependencies with:
```bash
pip install -r requirements.txt
```

---

## 📝 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

Please read [CONTRIBUTING.md](CONTRIBUTING.md) and [CODE_OF_CONDUCT.md](CODE_OF_CONDUCT.md) for details on our code of conduct and the process for submitting pull requests.

---

## 📧 Contact

For questions or feedback, please open an issue on GitHub.

---

## 🙏 Acknowledgements

- NASA for providing the climate change content that inspired this analysis
- The social media community for engaging with climate change discussions