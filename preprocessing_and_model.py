import os
import numpy as np
from sklearn.model_selection import train_test_split
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
from tensorflow.keras.callbacks import TensorBoard


def preprocess_and_train_model(DATA_PATH, actions, model_name, epochs):
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

    model = Sequential()
    model.add(LSTM(64, return_sequences=True, activation='relu',
                   input_shape=(30, 1662)))
    model.add(LSTM(128, return_sequences=True, activation='relu'))
    model.add(LSTM(64, return_sequences=False, activation='relu'))
    model.add(Dense(64, activation='relu'))
    model.add(Dense(32, activation='relu'))
    model.add(Dense(actions.shape[0], activation='softmax'))

    model.compile(optimizer='adam', loss='categorical_crossentropy',
                  metrics=['categorical_accuracy'])

    log_dir = os.path.join('Logs')
    tb_callback = TensorBoard(log_dir=log_dir)

    if isinstance(epochs, int):
        model.fit(X_train, y_train, epochs=epochs, callbacks=[tb_callback])
    else:
        model.fit(X_train, y_train, epochs=200, callbacks=[tb_callback])

    model.save(f'models/{model_name}.h5')

    print("Model trained and saved.")
