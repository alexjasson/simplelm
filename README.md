# simplelm
This is a basic character level RNN with MGU cells (https://arxiv.org/pdf/1603.09420).


## Compilation

Prerequisites:

- A UNIX-like OS (Linux or mac)
- GCC
- make

From the project root, run:

```bash
make
```

## Usage

To train the model:

```
./train <hiddenSize> <numLayers> <seqLength>
```

To chat with the model:

```
./chat <temperature>
```

- `hiddenSize` - size of the hidden (and embedding) layers
- `numLayers` - number of stacked MGU layers
- `seqLength` - number of tokens per training step
- `temperature` - sampling temperature (0 for greedy, higher for more randomness)

## Getting started

```
make
./train 128 2 64
./chat 0.5
```

![](<images/perft(7).png>)