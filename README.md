# Sentence-Level Sequence Labeling

This repository contains the corpus and code for **Sentence-Level Sequence Labeling**.

[**Sentence-Level Sequence Labeling for Automatic Compliance Analysis of Privacy Policy Text with GDPR Article 13**](https://github.com/zhangfanTJU/Sentence-Level-Sequence-Labeling)

Fan Zhang, Meishan Zhang, Shuang Liu and Baiyang Zhao

## Reqirements
```
pip install -r requirements.txt
```

## Quick Start
```
# Baseline
## GloVe
python3 train-baseline.py --emb glove

## BERT
python3 train-baseline.py --emb bert

# +Syntax
## GloVe
python3 train.py --emb glove --use_dep

## BERT
python3 train.py --emb bert --use_dep
```

## Citation

```bibtex
@article{zhang2022sequencelabeling,
  title={Sentence-Level Sequence Labeling for Automatic Compliance Analysis of Privacy Policy Text with GDPR Article 13}, 
  author={Fan Zhang, Meishan Zhang, Shuang Liu and Baiyang Zhao}
}
```
