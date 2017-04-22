# pytorch-practice
Some example scripts on pytorch

## CONLL 2000 Chunking task

Uses BiLSTM CRF loss with char CNN embeddings. To run use:

```
cd data/conll2000
bash get_data.sh
cd ..
ipython "PyTorch CONLL 2000 Chunking.py" # Currently takes 9 hours on Tesla K80 GPU
```

89.7% mean F1 on test data. 

