"""
Author: Chao Li

Description:
This program implements a CLI-based ML application that can:
    - Train and evaluate Regression and Classification models
    - Persist models using pickle
    - Load saved models to make prediction from user input
Datasets:
    - Regression: California Housing dataset (scikit-learn)
    - Classification: Phishing Website Detection dataset (Kaggle CSV)
"""

from regression import train_regression_model, use_regression_model
from classification import train_classification_model, use_classification_model

# ==============================================
# Main menu
# ==============================================

def show_main_menu():
    print("\nWelcome to the ML CLI App")
    print("Choose task:")
    print("1) Regression (estimate a number)")
    print("2) Classification (phishing detection)")

def show_second_menu():
    print("\nChoose option: ")
    print("1) Train model")
    print("2) Use model for prediction")

# ==============================================
# Main program
# ==============================================

def main():
    while True:
        show_main_menu()
        task = input("> ")

        if task == "1":
            show_second_menu()
            option = input("> ")

            if option == "1":
                train_regression_model()
            elif option == "2":
                use_regression_model()
            else:
                print("Invalid option. Please try again.")
        
        elif task == "2":
            show_second_menu()
            option = input("> ")

            if option == "1":
                print("Choose classifier model: ")
                print("1) KNN\n2) SVM\n3) DecisionTree")
                model_choice_map = {"1": "KNN", "2": "SVM", "3": "DecisionTree"}
                model_choice = input("> ")
                train_classification_model(model_choice=model_choice_map.get(model_choice, "KNN"))
            elif option == "2":
                use_classification_model()
            else:
                print("Invalid option. Please try again.")
        else:
            print("Invalid task selection. Please try again.")

if __name__ == "__main__":
    main()