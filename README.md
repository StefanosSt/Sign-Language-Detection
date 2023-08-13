## Sign Language Detection

### Description

This is a Python project for sign language detection using deep learning techniques. The project allows the user to create their own dataset, train a model, and test it in real-time.

### Dependencies

- Python 3.x
- TensorFlow 2.x
- Keras
- OpenCV
- NumPy
- Mediapipe
- Inquirer

To install the dependencies, run the following command in the terminal:

```
pip install numpy inquirer keras tensorflow matplotlib opencv-python

```

### Usage

To run the project, navigate to the project directory and run the following command in the terminal:

```
python pada.py
```

The user will then be prompted to select one of the following options:

<ul>
    <li>Create your own dataset</li>
    <li>Test in real time</li>
    <li>Exit</li>
</ul>

### Create your own dataset

<p>If you choose this option, you will be prompted to enter a name for your dataset folder and the classes for your model, separated by commas. Once you have provided this information, the program will ask if you want to start creating the dataset. If you choose to do so, the program will begin collecting data and storing it in the specified folder. If you choose not to create the dataset at this time, you can do so later by running the 'collect_data.py' script.</p>

<p>If you choose to train the model, the program will ask for a name for your model and the number of epochs you want to run. The program will then preprocess the data, train the model, and save it to a .h5 file. If the user enters an invalid input for the number of epochs, the program will run the fit function with 200 epochs by default.</p>

<p><b>Note:</b> If you have an existing dataset and want to train a new model, make sure to move your dataset to a new folder, otherwise the old dataset will be overwritten.</p>

<h3>Test in real time</h3>

<p>If you choose this option, the program will prompt you to select a model to test and the classes that were used to train the model. The program will then open a camera feed and begin detecting and classifying signs in real-time. If the user chooses to log this action, the program will save a log entry to logs.md that records the model and classes used for the test.</p>

<p>Note: If you do not have a model to test, you can train one using the 'Create your own dataset' option.</p>

### Author

Stefanos Stamoulis mscaidl-0014
