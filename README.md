# 🤖 Java Machine Learning with Weka – Zoo Classification Project

This project demonstrates how to use the Weka library in Java to build, evaluate, and interact with a machine learning model. The dataset used is based on animal classification from the UCI Zoo dataset.

> 🎓 Academic project built in Java using Weka & RandomForest

---

## 🎯 Project overview

- Load and preprocess `.arff` dataset files
- Train a **RandomForest** model on the zoo dataset
- Save and reuse the trained model for predictions
- Make interactive predictions via console input
- Display statistics and model performance

---

## 🛠️ Tech stack

- **Java 21** (Adoptium)
- **Weka** library (v3.8+)
- **IDE:** VS Code (but compatible with IntelliJ or Eclipse)
- **Build tool:** Maven (via `pom.xml`)

---

## 📁 Project structure

/JAVA_ML
├── src/main/java/com/example/
│ ├── WekaMonZoo.java # Trains and saves the model
│ ├── WekaPredictInteractive.java # Predicts based on user input
│ └── WekaStats.java # Outputs model statistics
├── src/main/resources/
│ ├── zoo.arff # Dataset for training
│ └── zoo_test.arff # Dataset for testing (optional)
├── zoo-model.model # Serialized model
├── pom.xml # Maven config
└── README.md # You are here!

---

## 🚀 How to run the project

1. Make sure you have **Java 17+** and **Maven** installed.
2. Compile the project:
```bash
mvn compile
training: mvn exec:java -Dexec.mainClass="com.example.WekaMonZoo"
predictor: mvn exec:java -Dexec.mainClass="com.example.WekaPredictInteractive"
 
 ## ✨ Features

- [x] Load `.arff` datasets using Weka's DataSource
- [x] Remove unused or string attributes
- [x] Set target class index dynamically
- [x] Train RandomForest classifier
- [x] Save trained model to disk (`.model`)
- [x] Interactive console prediction (user input via CLI)
- [x] Print stats per attribute or prediction analysis
