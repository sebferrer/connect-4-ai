from keras.layers.core import Dense, Activation, Dropout
from keras.models import load_model
from keras.models import Sequential
from keras.utils import np_utils

import tensorflow as tf
import pandas as pd
import numpy as np
import shutil
import flask
import os
from flask_cors import CORS
import json
from threading import Thread
from pathlib import Path
import copy
from random import randrange

# initialize our Flask application and the Keras model
LOCAL_DIR = '/opt/keras-daemon/server/'
FILES_DIR = '/opt/keras-daemon/files/'
NB_EPOCHS = 100
training_nns = {}
ready_nns = []
app = flask.Flask(__name__)
CORS(app)


def is_number(s):
    try:
        float(s)
        return True
    except ValueError:
        return False


def label_vocabulary(labels):
    labels_dict = {}
    i = 0
    for label in labels:
        if label not in labels_dict:
            labels_dict[label] = i
            i += 1
    return labels_dict


def replace_commas(model_name, train_file):
    train_file = os.path.join(FILES_DIR + train_file)
    # Replace commas with colons
    with open(train_file, 'r') as f:
        # Remove model directory if it already exists
        if os.path.exists(model_name) and os.path.isdir(model_name):
            shutil.rmtree(model_name)
        os.mkdir(LOCAL_DIR + model_name)
        with open(LOCAL_DIR + model_name + '/' + "training.csv", 'w') as t:
            for line in f:
                new_line = line.replace(";", ",")
                t.write(new_line)


def replace_commas_test(model_name, test_file):
    # Replace commas with colons
    with open(test_file, 'r') as f:
        with open(LOCAL_DIR + model_name + '/' + "testing.csv", 'w') as t:
            for line in f:
                new_line = line.replace(";", ",")
                t.write(new_line)


def model_(input_dim, hidden, output):
    #  MLP Model
    layer = 0
    model = Sequential()
    # The first layer
    model.add(Dense(hidden[layer], input_dim=input_dim))
    model.add(Activation('relu'))
    model.add(Dropout(0.15))

    # Hidden layers
    layer += 1
    while layer < len(hidden):
        model.add(Dense(hidden[layer]))
        model.add(Activation('relu'))
        model.add(Dropout(0.15))
        layer += 1

    # The Softmax layer (last layer)
    model.add(Dense(int(output)))
    model.add(Activation('softmax'))

    # we'll use categorical xent for the loss, and RMSprop as the optimizer
    model.compile(loss='categorical_crossentropy', optimizer='rmsprop', metrics=['accuracy'])
    return model


def train_(model_name, train_file, hidden, output, nb_epochs):
    training_nns[model_name] = {'epoch': 0, 'nb_epochs': nb_epochs, 'percentage':  0, 'running': True}

    # Replace commas with colons
    replace_commas(model_name, train_file)

    # Read data
    training = pd.read_csv(LOCAL_DIR + model_name + '/' + "training.csv", header=None)
    labels = training.ix[:, training.shape[1] - 1].values.astype('int32')
    x_training = training.ix[:, :training.shape[1] - 2].values.astype('float32')

    # Create the dictionary of labels
    labels_dict = label_vocabulary(labels)
    labels_classes = []
    for i in labels:
        labels_classes.append(labels_dict[i])
    labels_classes = np.array(labels_classes)

    # convert list of labels to binary class matrix
    y_train = np_utils.to_categorical(labels_classes)

    # pre-processing: divide by max and substract mean
    scale = np.max(x_training)
    x_training /= scale
    mean = np.std(x_training)
    x_training -= mean
    input_dim = x_training.shape[1]

    # Save the data
    np.savez(LOCAL_DIR + model_name + '/' + model_name + '.npz', labels_dict=labels_dict, scale=scale, mean=mean)

    # Building the model
    # Tensorflow graph
    globals()['graph-{}'.format(model_name)] = tf.Graph()
    with globals()['graph-{}'.format(model_name)].as_default():
        # Tensorflow session for this graph
        globals()['session-{}'.format(model_name)] = tf.Session()
        with globals()['session-{}'.format(model_name)].as_default():
            globals()['model-{}'.format(model_name)] = model_(input_dim, hidden, output)

            for i in range(int(nb_epochs)):
                print("Epoch " + str(i+1) + "/ " + str(nb_epochs))
                training_nns[model_name]['epoch'] = i+1
                training_nns[model_name]['percentage'] = (i+1) * 100 / nb_epochs
                globals()['model-{}'.format(model_name)].fit(x_training, y_train, nb_epoch=1, batch_size=16,
                                                             validation_split=0.2, verbose=1)
                if not training_nns[model_name]['running']:
                    print("-- Training stopped --")
                    break

            globals()['model-{}'.format(model_name)].save(str(LOCAL_DIR + model_name + '/' + model_name + ".h5"))
    if model_name in training_nns:
        ready_nns.append(model_name)
        del training_nns[model_name]


def load_(model_name, model_file_path):
    # create tensorflow graph and session, then load the model
    globals()['graph-{}'.format(model_name)] = tf.Graph()
    with globals()['graph-{}'.format(model_name)].as_default():
        globals()['session-{}'.format(model_name)] = tf.Session()
        with globals()['session-{}'.format(model_name)].as_default():
            globals()['model-{}'.format(model_name)] = load_model(model_file_path)


def get_status(model_name):
    nn_status = "nexist"
    if model_name in ready_nns:
        nn_status = "ready"
    elif model_name in [*training_nns]:
        nn_status = "training"

    return nn_status


def check_consistency_(training_file, nb_inputs, nb_outputs):
    nb_inputs = int(nb_inputs)
    nb_outputs = int(nb_outputs)
    training_file = os.path.join(FILES_DIR + training_file)
    if not os.path.exists(training_file):
        return json.dumps({'status': "nexist", 'line': 0, 'column': 0})
    elif os.stat(training_file).st_size == 0:
        return json.dumps({'status': "empty", 'line': 0, 'column': 0})
    f = open(training_file, "r")
    line_index = 1
    col_index = 1
    for line in f:
        if not line:
            return json.dumps({'status': "emptyinput", 'line': line_index, 'column': col_index})
        inputs = line.split(";")
        output = inputs.pop()
        if len(inputs) != nb_inputs:
            return json.dumps({'status': "badinputlength", 'line': line_index, 'column': col_index})
        if not is_number(output):
            return json.dumps({'status': "notnumericoutput", 'line': line_index, 'column': col_index})
        output = int(output)
        if output < 1 or output > nb_outputs:
            return json.dumps({'status': "badnumericoutput", 'line': line_index, 'column': col_index})
        for current_input in inputs:
            if not is_number(current_input):
                return json.dumps({'status': "notnumericinput", 'line': line_index, 'column': col_index})
            col_index += 1
        col_index = 1
        line_index += 1
    f.close()

    return json.dumps({'status': "ready", 'line': 0, 'column': 0})


def check_parameters(expected_args, current_args):
    missing_args = []
    for arg in current_args:
        if arg not in expected_args:
            missing_args.append(arg)

    return missing_args


def write_(output_dir, file_name, file_content):
    if os.path.exists(output_dir + file_name):
        os.remove(output_dir + file_name)
    f = open(output_dir + file_name, "a")
    f.write(file_content)
    f.close()


def to_2d_array(nb_cols, nb_rows, vector):
    array = []

    vector = vector.split(";")

    if nb_cols * nb_rows != len(vector):
        print("Error: vector length is " + str(len(vector)) + " and should be" + str(nb_cols * nb_rows))
        return None

    c = 0
    for i in range(nb_cols):
        new = []
        for j in range(nb_rows):
            new.append(int(vector[c]))
            c += 1
        array.append(new)

    return array

# Upload page
@app.route("/", methods=["GET"])
def home():
    file_name = 'upload.html'
    return flask.render_template(file_name)


# http://localhost:5000/upload
@app.route("/upload", methods=["POST"])
def upload():
    output_dir = FILES_DIR
    file_name = flask.request.form.get('filename')
    file_content = flask.request.form.get('content')

    write_(output_dir, file_name, file_content)

    return flask.jsonify({'upload': 'success'})


# http://localhost:5000/train?owner=person&id=1&train_file=train.csv&input=4&hidden=20:10&output=3
@app.route("/mlp/train", methods=["GET"])
def train():
    owner = flask.request.args.get('owner')
    id_neural_net = flask.request.args.get('id')
    training_file = flask.request.args.get('train_file')
    nn_input = flask.request.args.get('input')
    hidden = flask.request.args.get('hidden')
    output = flask.request.args.get('output')
    nb_epochs = flask.request.args.get('nb_epochs')
    missing_args = check_parameters(flask.request.args, ['owner', 'id', 'train_file', 'input', 'hidden', 'output'])
    if missing_args:
        return flask.jsonify({'train': 'missing_args', 'missing_args': missing_args})

    nb_epochs = NB_EPOCHS if nb_epochs is None else int(nb_epochs)

    hidden = hidden.split(':')
    hidden_layers = []
    for i in hidden:
        hidden_layers.append(int(i))
    model_name = owner + "." + id_neural_net

    consistency = json.loads(check_consistency_(training_file, nn_input, output))
    if str(consistency['status']) == 'ready':
        # train the model
        if model_name in [*training_nns]:
            return flask.jsonify({'train': 'already_training', 'epoch': training_nns[model_name]['epoch'],
                                  'nb_epochs': training_nns[model_name]['nb_epochs'],
                                  'percentage': training_nns[model_name]['percentage']})
        elif model_name in ready_nns:
            return flask.jsonify({'train': 'already_trained'})
        else:
            thread = Thread(target=train_, args=(model_name, training_file, hidden_layers, output, nb_epochs))
            thread.start()
    else:
        return flask.jsonify(consistency)

    # Output
    return flask.jsonify({'train': 'training'})


# http://localhost:5000/status?owner=person&id=1
@app.route("/mlp/status", methods=["GET"])
def status():
    owner = flask.request.args.get('owner')
    id_neural_net = flask.request.args.get('id')
    missing_args = check_parameters(flask.request.args, ['owner', 'id'])
    if missing_args:
        return flask.jsonify({'status': 'missing_args', 'missing_args': missing_args})

    model_name = owner + "." + id_neural_net

    nn_status = get_status(model_name)

    if nn_status == "nexist" or nn_status == "ready":
        return flask.jsonify({'status': nn_status})

    return flask.jsonify({'status': nn_status, 'epoch': training_nns[model_name]['epoch'],
                          'nb_epochs': training_nns[model_name]['nb_epochs'],
                          'percentage': training_nns[model_name]['percentage']})


# http://localhost:5000/stop?owner=person&id=1
@app.route("/mlp/stop", methods=["GET"])
def stop():
    owner = flask.request.args.get('owner')
    id_neural_net = flask.request.args.get('id')
    missing_args = check_parameters(flask.request.args, ['owner', 'id'])
    if missing_args:
        return flask.jsonify({'stop': 'missing_args', 'missing_args': missing_args})

    model_name = owner + "." + id_neural_net

    if model_name not in [*training_nns]:
        return flask.jsonify({'stop': 'error_not_training'})

    training_nns[model_name]['running'] = False

    return flask.jsonify({'stop': 'success'})


# http://localhost:5000/load?owner=person&id=1
@app.route("/mlp/load", methods=["GET"])
def load():
    owner = flask.request.args.get('owner')
    id_neural_net = flask.request.args.get('id')
    missing_args = check_parameters(flask.request.args, ['owner', 'id'])
    if missing_args:
        return flask.jsonify({'load': 'missing_args', 'missing_args': missing_args})

    model_name = owner + "." + id_neural_net

    model_file_path = LOCAL_DIR + model_name + '/' + model_name + '.h5'
    my_file = Path(model_file_path)
    if not my_file.is_file():
        return flask.jsonify({'load': 'error_file_not_exist'})

    # Load the model
    load_(model_name, model_file_path)

    return flask.jsonify({'load': 'success'})


# http://localhost:5000/remove?owner=person&id=1
@app.route("/mlp/remove", methods=["GET"])
def remove():
    owner = flask.request.args.get('owner')
    id_neural_net = flask.request.args.get('id')
    missing_args = check_parameters(flask.request.args, ['owner', 'id'])
    if missing_args:
        return flask.jsonify({'remove': 'missing_args', 'missing_args': missing_args})

    model_name = owner + "." + id_neural_net

    if model_name in [*training_nns]:
        return flask.jsonify({'remove': 'error_still_training'})

    if model_name not in ready_nns:
        return flask.jsonify({'remove': 'error_model_nexist'})

    model_file_path = LOCAL_DIR + model_name + '/' + model_name + '.h5'
    my_file = Path(model_file_path)
    if not my_file.is_file():
        return flask.jsonify({'remove': 'error_file_not_exist'})
    else:
        os.remove(model_file_path)
        shutil.rmtree(LOCAL_DIR + model_name)
        if model_name in ready_nns:
            ready_nns.remove(model_name)

    return flask.jsonify({'remove': 'success'})


# http://127.0.0.1:5000/checkconsistency?trainingfile=admin.admin.shapesapi.csv&nbinputs=25&nboutputs=7
@app.route("/checkconsistency", methods=["GET"])
def check_consistency():
    training_file = flask.request.args.get('trainingfile')
    nb_inputs = int(flask.request.args.get('nbinputs'))
    nb_outputs = int(flask.request.args.get('nboutputs'))
    missing_args = check_parameters(flask.request.args, ['trainingfile', 'nbinputs', 'nboutputs'])
    if missing_args:
        return flask.jsonify({'status': 'missing_args', 'missing_args': missing_args})

    return flask.jsonify(json.loads(check_consistency_(training_file, nb_inputs, nb_outputs)))


# http://localhost:5000/predict?owner=person&id=1&input_file=test.csv
@app.route("/mlp/predict", methods=["POST"])
def predict():
    prediction = {}

    player = int(flask.request.form.get('player'))
    board = str(flask.request.form.get('board'))

    board2d = to_2d_array(7, 6, board)
    board = normalize(board, ";")

    attack = detect_attack(player, board2d)
    if attack > -1:
        print("Attack: " + str(attack+1))
        prediction['status'] = 'success'
        prediction['prediction'] = attack + 1
        prediction['confidence'] = 1
        return flask.jsonify(prediction)
    else:
        defense = detect_defense(player, board2d)
        if defense > -1:
            print("Defense: " + str(defense+1))
            prediction['status'] = 'success'
            prediction['prediction'] = defense + 1
            prediction['confidence'] = 1
            return flask.jsonify(prediction)

    fk_juan = no_juan(player, board2d)
    if no_juan(player, board2d) > -1:
        print("No Juan: " + str(fk_juan + 1))
        prediction['status'] = 'success'
        prediction['prediction'] = fk_juan + 1
        prediction['confidence'] = 1
        return flask.jsonify(prediction)

    owner = "mlp.bot"
    id_neural_net = "player" + str(player)
    file_name = id_neural_net + ".csv"

    input_file = FILES_DIR + file_name

    write_(FILES_DIR, file_name, board)

    if not os.path.exists(input_file):
        return flask.jsonify({'status': 'error_input_nexist', 'prediction': 0, 'confidence': 0})

    model_name = owner + "." + id_neural_net

    if model_name not in ready_nns:
        return flask.jsonify({'status': 'error_nn_not_ready', 'prediction': 0, 'confidence': 0})

    # Load data
    data = np.load(LOCAL_DIR + model_name + '/' + model_name + '.npz')
    labels_dict = data['labels_dict'].item()
    scale = data['scale']
    mean = data['mean']

    # Read test example
    replace_commas_test(model_name, input_file)
    x_test = pd.read_csv(LOCAL_DIR + model_name + '/' + 'testing.csv', header=None).values.astype('float32')

    # pre-processing: divide by max and substract mean
    x_test /= scale
    x_test -= mean

    with globals()['graph-{}'.format(model_name)].as_default():
        with globals()['session-{}'.format(model_name)].as_default():
            predictions = globals()['model-{}'.format(model_name)].predict(x_test)

    predicted_classe = np.argmax(np.round(predictions), axis=1)
    predicted = int(list(labels_dict.keys())[list(labels_dict.values()).index(predicted_classe)]) - 1

    safe_lines = get_safe_lines(player, board2d)
    available_lines = get_available_lines(board) if len(safe_lines) == 0 else safe_lines
    if predicted not in available_lines:
        safe_line = available_lines[randrange(len(available_lines))]
        print("Line " + str(predicted) + " not safe or not available, play " + str(safe_line) + " instead")
        prediction['status'] = 'success'
        prediction['prediction'] = safe_line + 1
        prediction['confidence'] = 1
        return flask.jsonify(prediction)

    # return the predicted label
    print("Predicted: " + str(predicted))
    prediction['status'] = 'success'
    prediction['prediction'] = predicted + 1
    prediction['confidence'] = float(max(predictions[0]))

    # return the results as a JSON response
    return flask.jsonify(prediction)


def pivot_board(board):
    new_board = []
    for i in range(len(board)):
        new_board[i] = []
        for j in range(len(board[i])):
            new_board[j][i] = board[i][j]
    del new_board[-1]

    return new_board


def check_board(board, player):
    player = int(player)
    nb_cols = 7
    nb_rows = 6

    # horizontalCheck
    for j in range(nb_rows-3):
        for i in range(nb_cols):
            if board[i][j] == player and board[i][j+1] == player and \
                    board[i][j+2] == player and board[i][j+3] == player:
                return True

    # verticalCheck
    for i in range(nb_cols-3):
        for j in range(nb_rows):
            if board[i][j] == player and board[i+1][j] == player and \
                    board[i+2][j] == player and board[i+3][j] == player:
                return True

    # ascendingDiagonalCheck
    for i in range(3, nb_cols):
        for j in range(nb_rows-3):
            if board[i][j] == player and board[i-1][j+1] == player and \
                    board[i-2][j+2] == player and board[i-3][j+3] == player:
                return True

    # descendingDiagonalCheck
    for i in range(3, nb_cols):
        for j in range(3, nb_rows):
            if board[i][j] == player and board[i-1][j-1] == player and \
                    board[i-2][j-2] == player and board[i-3][j-3] == player:
                return True

    return False


def normalize(vec, delimiter):
    vec = vec.split(delimiter)
    for i in range(len(vec)):
        vec[i] = str(float(vec[i])/2)
    vec = delimiter.join(vec)

    return vec


def get_other_player(player):
    return 2 if player == 1 else 1


def detect_attack(player, board):
    return detect_attack_or_defense(player, board)


def detect_defense(player, board):
    return detect_attack_or_defense(get_other_player(player), board)


def detect_attack_or_defense(player, board):
    available_lines = get_available_lines(board)
    for i in range(len(available_lines)):
        line = available_lines[i]
        board_temp = copy.deepcopy(board)
        next_row = get_next_row(board, line)
        board_temp[line][next_row] = player
        if check_board(board_temp, player):
            return line
    return -1


def no_juan(player, board):
    for i in range(1, len(board) - 3):
        first_row = len(board[i]) - 1
        if board[i-1][first_row] == board[i+2][first_row] == 0 \
                and board[i][first_row] == board[i+1][first_row] == get_other_player(player):
            return i-1 if randrange(2) == 0 else i+2
    return -1


def get_next_row(board, line):
    for i in range(len(board[line]) - 1, -1, -1):
        if board[line][i] == 0:
            return i
    return -1


def get_available_lines(board):
    available_lines = []
    for i in range(len(board)):
        if board[i][0] == 0:
            available_lines.append(i)
    return available_lines


def get_safe_lines(player, board):
    available_lines = get_available_lines(board)
    safe_lines = []
    for i in range(len(available_lines)):
        line = available_lines[i]
        board_temp = copy.deepcopy(board)
        next_row = get_next_row(board, line)
        board_temp[line][next_row] = player
        if detect_defense(player, board_temp) == -1:
            safe_lines.append(line)
    return safe_lines


# curl -X GET 'http://localhost:5000/predict..'
# starting the server
if __name__ == "__main__":
    app.run(host='0.0.0.0', port=5001)
