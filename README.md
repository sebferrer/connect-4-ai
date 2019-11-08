# Connect 4 AI

Connect 4 deep learning AI

### To use Docker image

The Makefile in the "image" folder does everything for you. To build the image do:

```
cd image/
docker build --tag=connect4ai .
```

To run the Docker image locally:

```
docker run -p 5001:5001 connect4ai
```

| Port   |      Role      |    API Inputs    |
|----------|-------------|--------|
| /mlp/load | Load an existing model | owner, id |
| /mlp/train | Create and train a new model | owner, id, train_file, input, hidden, output, nb_epochs |
| /mlp/predict | Predict via a trained model | player, board |
| /mlp/status | Get the status of a model | owner, id |
| /mlp/stop | Stop a model traning | owner, id |
| /mlp/remove | Remove a model | owner, id |
| /mlp/check_consistency | Check the consistency of a training file with respect to a model | trainingfile, nbinputs, nboutputs |
