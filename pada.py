import os
import sys
import inquirer
import numpy as np

# If you don't have inquirer --> pip install inquirer
from inquirer import prompt, List

# Functions
from realtime_testing import realtime_testing
from data_collection import collect_data
from preprocessing_and_model import preprocess_and_train_model
from evaluation import evaluation

# Define a list of choices
choices = [
    inquirer.List('action',
                  message="Select an option:",
                  choices=[
                      'Create your own dataset',
                      'Test in real time',
                      'Exit'
                  ])
]

# Prompt the user to select an option
answer = inquirer.prompt(choices)

# Process the user's answer
if answer['action'] == 'Create your own dataset':
    print("You chose to create your own dataset!")
    # Prompt user for dataset folder name
    dataset_name = input("Enter a name for the dataset folder: ")
    DATA_PATH = os.path.join('datasets', dataset_name)
    try:
        os.makedirs(DATA_PATH)
    except:
        pass

    # Prompt user for classes
    classes_str = input(
        "Enter the classes for the model, separated by commas: ")
    actions = np.array(classes_str.split(','))

    # Log the user's selections to the log file
    with open("logs.md", "a") as log_file:
        log_file.write("Dataset folder name: " + dataset_name + "\n")
        log_file.write("Classes: " + classes_str + "\n")

    print("Dataset folder name: ", dataset_name)
    print("Classes: ", actions)

    # Prompt user if they want to start creating their dataset
    train_model_choice = input("Do you want to create the dataset now? (y/n) ")
    if train_model_choice.lower() == 'y':
        collect_data(DATA_PATH, actions)
    else:
        print("That's too bad. See you around!")

    # Prompt user if they want to train the model now
    train_model_choice = input("Do you want to train the model now? (y/n) ")
    if train_model_choice.lower() == 'y':
        # Get name for the model
        model_name = input("Enter a name for your model: ")
        epochs = int(input("Enter a number for epochs: "))

        preprocess_and_train_model(DATA_PATH, actions, model_name, epochs)

        # Log the user's selections to the log file
        with open("logs.md", "a") as log_file:
            log_file.write("Model name: " + model_name + "\n")
            log_file.write("Epochs: " + str(epochs) + "\n")

    else:
        print("Okay, you can train your model later by running the 'preprocessing_and_model.py' script.")

elif answer['action'] == 'Test in real time':
    print("You chose to test in real time!")
    models = [f for f in os.listdir('models') if f.endswith('.h5')]
    if not models:
        print('No models found in root folder!')
    else:
        questions = [
            List('model',
                 message='Choose the model that you want to test out:',
                 choices=models,
                 ),
        ]

        answer = prompt(questions)
        model_path = answer['model']

        print(f"You chose model {answer['model']}")
        classes = input(
            "Enter the classes of the selected model, separated by commas: ")
        classes = classes.split(',')
        classes = [c.strip() for c in classes]
        print("You have chosen the following classes:")
        for i, c in enumerate(classes):
            print(f"{i+1}. {c}")
        realtime_testing(model_path, classes)

        # Prompt user to log the action
        log_choice = input("Do you want to log this action? (y/n) ")
        if log_choice.lower() == 'y':
            log_entry = f"\nTest in real time:\nModel: {model_path}\nClasses: {classes}"
            with open("logs.md", "a") as f:
                f.write(log_entry)
            print("Log entry saved to 'logs.md'!")
        else:
            print("Action not logged.")
