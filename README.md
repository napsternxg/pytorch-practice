# pytorch-practice
Some example scripts on pytorch

## CONLL 2000 Chunking task

Uses BiLSTM CRF loss with char CNN embeddings. To run use:

```
cd data/conll2000
bash get_data.sh
cd ..
python chunking_bilstm_crf_char_concat.py # Takes around # 8 hours on Tesla K80 GPU
```

92.82% mean F1 on test data. 

