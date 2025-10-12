🌍 Earthquake Damage Prediction Pipeline
----------------------------------------

This project implements a comprehensive machine learning pipeline to predict the extent of damage (Low, Medium, or High) sustained by buildings after an earthquake, utilizing structural and seismic data. The pipeline features **advanced feature engineering**, **data balancing with SMOTE**, a **standardized scaling step**, and an **optimized weighted soft-voting ensemble** for high-performance prediction.

🚀 Project Overview
-------------------

The goal of this project is to accurately classify the damage\_grade of buildings into three categories (1: Low, 2: Medium, 3: High, mapped internally to 0, 1, 2) based on various structural characteristics, geographical features, and seismic indicators.

The core of the solution is the model.py script, which automates the entire process: data loading, cleaning, feature engineering, training multiple baseline models, training a powerful weighted ensemble classifier, evaluation, and model persistence.

### Key Features

*   **Advanced Feature Engineering:** Creation of interaction features (e.g., soil\_foundation\_interaction, age\_quake\_ratio) and **regional mean-damage** features (Target Encoding applied to geographical levels).
    
*   **Data Preprocessing:** Handling of missing values, standardization of numerical features, and **Target Encoding** for categorical variables.
    
*   **Imbalanced Data Handling:** Application of **SMOTE** (Synthetic Minority Over-sampling Technique) to the training data to improve performance on minority classes.
    
*   **Weighted Ensemble:** Utilizes a **Weighted Soft-Voting Classifier** combining high-performing models like XGBoost, LightGBM, and Random Forest for robust predictions.
    
*   **Model Persistence:** The final ensemble model, along with the StandardScaler and TargetEncoder, is saved using joblib for easy deployment.
    

🛠️ Requirements
----------------

The project requires Python 3.x and the following libraries. The list below is sourced from the included requirements.txt file and represents the necessary packages.

*   pandas
    
*   numpy
    
*   matplotlib
    
*   seaborn
    
*   scikit-learn
    
*   xgboost
    
*   lightgbm
    
*   imbalanced-learn
    
*   category-encoders
    

### Project Files

*   model.py: The complete Python script containing the machine learning pipeline.
    
*   02\_final\_ml\_ready\_dataset.csv: The **input dataset file**. (Must be available in the same directory or the path updated in model.py).
    
*   requirements.txt: List of Python dependencies.
    
*   results.txt: Contains the printed output from a previous run (included for reference).
    

⚙️ How to Run
-------------

This process assumes you have the project files (model.py, requirements.txt, and the data) in a single working directory.

### 1\. 💾 Clone the Repository

Plain textANTLR4BashCC#CSSCoffeeScriptCMakeDartDjangoDockerEJSErlangGitGoGraphQLGroovyHTMLJavaJavaScriptJSONJSXKotlinLaTeXLessLuaMakefileMarkdownMATLABMarkupObjective-CPerlPHPPowerShell.propertiesProtocol BuffersPythonRRubySass (Sass)Sass (Scss)SchemeSQLShellSwiftSVGTSXTypeScriptWebAssemblyYAMLXML`   git clone https://github.com/keerthanab2201/Earthquake-Damage-Prediction.git   `

### 2\. 🐍 Create Virtual Environment

Creating a **virtual environment** is a best practice to isolate the project's dependencies and avoid conflicts with other Python projects on your system.

Plain textANTLR4BashCC#CSSCoffeeScriptCMakeDartDjangoDockerEJSErlangGitGoGraphQLGroovyHTMLJavaJavaScriptJSONJSXKotlinLaTeXLessLuaMakefileMarkdownMATLABMarkupObjective-CPerlPHPPowerShell.propertiesProtocol BuffersPythonRRubySass (Sass)Sass (Scss)SchemeSQLShellSwiftSVGTSXTypeScriptWebAssemblyYAMLXML`   cd /path/to/your/project/  python -m venv venv  venv\Scripts\activate   `

### 3\. 📦 Install Requirements

Use the provided requirements.txt file to install all necessary Python libraries within your newly activated virtual environment.

Plain textANTLR4BashCC#CSSCoffeeScriptCMakeDartDjangoDockerEJSErlangGitGoGraphQLGroovyHTMLJavaJavaScriptJSONJSXKotlinLaTeXLessLuaMakefileMarkdownMATLABMarkupObjective-CPerlPHPPowerShell.propertiesProtocol BuffersPythonRRubySass (Sass)Sass (Scss)SchemeSQLShellSwiftSVGTSXTypeScriptWebAssemblyYAMLXML`   pip install -r requirements.txt   `

### 4\. ▶️ Run the model

Execute the main machine learning pipeline script.

2.  **Monitor Output:** The script will print the results of the data exploration, feature preparation, baseline model performance, and the final ensemble evaluation directly to the console (or to results.txt, if redirected).
    
3.  **Check for Artifacts:** Upon completion, the following files will be generated in your project directory:
    
    *   **earthquake\_ensemble.pkl**: The saved machine learning object.
        
    *   **Weighted\_Ensemble\_confusion\_matrix.png**: The visual evaluation of the final ensemble model.
        
    *   **Pruned\_XGBoost\_confusion\_matrix.png**: The visual evaluation of the optional XGBoost model trained on pruned (selected) features.
