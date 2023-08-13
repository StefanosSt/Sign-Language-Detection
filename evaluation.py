from sklearn.utils import class_weight
import os
import numpy as np
from sklearn.metrics import classification_report, confusion_matrix
from tensorflow.keras.models import load_model
from tensorflow.keras.utils import to_categorical
from sklearn.model_selection import train_test_split


def evaluation(DATA_PATH, actions, model_name):
    no_sequences = 30
    sequence_length = 30

    label_map = {label: num for num, label in enumerate(actions)}

    sequences, labels = [], []

    for action in actions:
        for sequence in range(no_sequences):
            window = []
            for frame_num in range(sequence_length):
                resampled = np.load(os.path.join(
                    DATA_PATH, action, str(sequence), "{}.npy".format(frame_num)))
                window.append(resampled)
            sequences.append(window)
            labels.append(label_map[action])

    X = np.array(sequences)
    y = to_categorical(labels).astype(int)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.1, stratify=y)

    model = load_model(f'models/{model_name}.h5')

    y_pred = model.predict(X_test)
    y_pred_classes = np.argmax(y_pred, axis=1)
    y_true_classes = np.argmax(y_test, axis=1)

    print("Confusion Matrix:")
    print(confusion_matrix(y_true_classes, y_pred_classes))
    print("\nClassification Report:")
    print(classification_report(y_true_classes,
          y_pred_classes, target_names=actions))
