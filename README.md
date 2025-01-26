# MLOps-Project

Files and Codes related to the complete MLOps Project.

---

## Overview
This project demonstrates the end-to-end implementation of MLOps practices to develop, deploy, and manage machine learning models. It includes workflows for REST API development, containerization, scalability, automation, CI/CD, model tracking, and an interactive recommendation system.

---

## Features
1. **Travel Price Prediction**:
   - Predict travel prices using a Random Forest regression model.
   - REST API developed with Flask.

2. **Hotel Recommendation System**:
   - Suggests hotels based on user preferences (price range, location, and duration).
   - Displays insights and enables interactive data exploration using Streamlit.

3. **Scalable Deployment**:
   - Containerized with Docker.
   - Deployed using Kubernetes for scalability.

4. **Automated Workflows**:
   - Apache Airflow orchestrates data preprocessing, model training, and testing.

5. **CI/CD Pipeline**:
   - Jenkins automates building, testing, and deploying the application.

6. **Model Tracking**:
   - MLFlow tracks experiments, metrics, and models systematically.

---

## Project Structure
```
.
├── .ipynb_checkpoints              # Checkpoints for Jupyter notebooks
├── Gender Classification Model     # Gender prediction model and related files
├── Hotel_Recommendation_System     # Files for hotel recommendation system
│   ├── streamlit_app.py            # Streamlit application code
│   ├── hotel_features.csv          # Hotel dataset
│   ├── users_hotel_history.csv     # User booking history dataset
├── apache-airflow                  # Apache Airflow workflows
├── jenkins_project                 # Files for Jenkins CI/CD pipeline
│   ├── data_preprocessing.py       # Data preprocessing script
│   ├── train_model.py              # Model training script
│   ├── test_model.py               # Model testing script
│   ├── Jenkinsfile                 # CI/CD pipeline definition
├── mlruns                          # MLFlow experiment logs
├── templates                       # HTML templates for Flask app
├── .dockerignore                   # Docker ignore file
├── .gitignore                      # Git ignore file
├── Dockerfile                      # Docker configuration file
├── Flight_Price_Prediction.ipynb   # Jupyter notebook for price prediction
├── ML Flow (Python file).ipynb     # Jupyter notebook for MLFlow integration
├── README.md                       # Project README file
├── app.py                          # Flask application for travel price prediction
├── flights.csv                     # Flight dataset
├── hotels.csv                      # Hotel dataset
├── ml_flight_dataset.csv           # Preprocessed flight dataset
├── price-prediction-deployment.yaml # Kubernetes deployment configuration
├── price-prediction-service.yaml   # Kubernetes service configuration
├── random_forest_price_prediction.pkl # Trained model file
├── requirements.txt                # Python dependencies
└── users.csv                       # User dataset
```

---

## Workflow

### 1. REST API Development
- Built a Flask-based REST API for travel price prediction.
- Integrated HTML templates for user interaction.
- Prediction results displayed dynamically.

### 2. Docker Deployment
- Containerized the Flask application using Docker.
- Commands used:
  ```bash
  docker build -t travel-price-prediction .
  docker run -p 5000:5000 travel-price-prediction
  ```

### 3. Kubernetes for Scalability
- Deployed the Docker container with Kubernetes for scalability.
- Configured YAML files to manage pods and services.
- Commands used:
  ```bash
  kubectl apply -f price-prediction-deployment.yaml
  kubectl apply -f price-prediction-service.yaml
  ```

### 4. Automated Workflows with Apache Airflow
- Orchestrated workflows for data preprocessing, training, and testing.
- Configured Directed Acyclic Graphs (DAGs) to manage workflows.

### 5. CI/CD Pipeline with Jenkins
- Automated pipeline using Jenkins with the following stages:
  - Clone repository.
  - Install dependencies.
  - Preprocess data.
  - Train and test model.
  - Build and deploy Docker container.
- Webhook integrated with GitHub to trigger the pipeline on code changes.

### 6. Model Tracking with MLFlow
- Tracked experiments for various regression models (e.g., Random Forest, XGBoost).
- Logged metrics, parameters, and artifacts.
- Registered and versioned models for deployment readiness.

### 7. Hotel Recommendation System
- Developed a Streamlit app for:
  - Hotel recommendations based on filters.
  - Insights like top destinations and frequently booked hotels.
  - Interactive data exploration.

---

## Installation and Usage

### Prerequisites
- Python 3.8 or higher
- Docker
- Kubernetes (Minikube or similar setup)
- Apache Airflow
- Jenkins
- MLFlow

### Steps to Run
1. Clone the repository:
   ```bash
   git clone https://github.com/Jay-mishra04/MLOps-Project.git
   ```

2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

3. Run the Flask application:
   ```bash
   python app.py
   ```

4. Build and run the Docker container:
   ```bash
   docker build -t travel-price-prediction .
   docker run -p 5000:5000 travel-price-prediction
   ```

5. Deploy on Kubernetes:
   ```bash
   kubectl apply -f price-prediction-deployment.yaml
   kubectl apply -f price-prediction-service.yaml
   ```

6. Launch the Streamlit web application:
   ```bash
   streamlit run Hotel_Recommendation_System/streamlit_app.py
   ```

7. Start MLFlow UI:
   ```bash
   mlflow ui
   ```

8. Set up and run the Jenkins pipeline.

---

## Screenshots

### Travel Price Prediction REST API
![Travel Price Prediction](image1.png)

### Docker Deployment
![Docker Images](image2.png)

### Kubernetes Deployment
![Kubernetes Deployment](image3.png)

### Apache Airflow DAGs
![Apache Airflow](image4.png)

### MLFlow Tracking
![MLFlow](image5.png)

### Hotel Recommendation System
![Hotel Recommendation System](image6.png)

---

## Conclusion
This project integrates various MLOps tools and practices to streamline the lifecycle of machine learning models. The implementation covers data preprocessing, automation, containerization, scalability, and interactive user interfaces, demonstrating a robust and scalable MLOps pipeline.

---

## Author
Mritunjay Mishra

