#  Neural Probabilistic Language Model (Character-Level)

Implementation of [**A Neural Probabilistic Language Model**](https://www.jmlr.org/papers/volume3/bengio03a/bengio03a.pdf) by *Yoshua Bengio et al., 2003* adapted to character-level modeling using PyTorch.

Datase used for training:- [**Tiny Shakespeare**](https://github.com/karpathy/char-rnn/blob/master/data/tinyshakespeare/input.txt)

## Repo layout

```

bengio-char-lm/
├── README.md
├── paper_walkthrough.md
├── requirements.txt
├── model.py
├── utils.py
├── train.py
├── generate.py
├── checkpoints/            # saved model weights
├── notebooks/             # experimental notebook
└── samples/               # sample outputs
```

---

## Quickstart

1. Create a virtual environment and install dependencies:

```bash
pip install -r requirements.txt
```

2. Train (defaults use `data/input.txt` — tiny shakespeare):

```bash
python train.py --data_path data/input.txt --epochs 15 --batch_size 128 --device cuda
```

3. Generate text from a checkpoint:

```bash
python generate.py --checkpoint checkpoints/best.pt --seed "KING:\n" --length 500 --temperature 0.8 --top_k 10
```

## Notes

- The `train.py` script saves the best model (validation loss) to `checkpoints/best.pt` by default.
- `generate.py` loads the checkpoint and the vocabulary saved during training, so make sure `vocab.pt` and checkpoint are present in the working directory.

---

## requirements.txt

```
torch
numpy
tqdm
```

---

## Sample text generated

When seeded with `KING:\n`

```

KING:
Clife, Paris, to his bear that a
Will'd great Ay, and
have thou detion. True and the way.

ISABELLA:
The should witching.

SICINIUS:
He'll diese,
Were had arms:
When that but servise?

DERBY:
Why, soal me, on more.

ROMEO:
So the other?

BENVOLIO:
Howsself of his brother's not but that did the like me what should not a foint of all this natullys he old and stain in a pointly the goldins,
And here to slege curded holy, my lord.

KING RICHARD III:
Now now, hon that you. Tranks,
More than water'd i

```

When seeded with `MARCIUS\n`
```

MARCIUS\now that hands
The words: when he wall honess in his armyant to him.

THOMAS MOWBRAY:
Yet; by shough, not it, and grough the cold with but my person, and her my lord, be the world one most to spuded strib,   
When he shall be shall be shiped on him:
And defend, he is thou ary my maid honour,
To do would not captasion all.

CORIOLANUS:
Ay, horry bury well stay,
For the shall death me badich but my did we county they doush are the hold arty.

DUKE VINCENTIO:
That with the countremed;
When he heardiend

```

---