# Topic DSMs

Source code for [Mixture of Topic-Based Distributional Semantic and Affective Models](https://ieeexplore.ieee.org/document/8334459/) conference paper.


## Datasets
You can download the corresponding datasets using the following links:

#### WS 353
```sh
$ cd data/
$ wget www.cs.technion.ac.il/~gabr/resources/data/wordsim353/wordsim353.zip
$ unzip wordsim353.zip
```

#### MEN
```sh
$ cd data/
$ wget https://staff.fnwi.uva.nl/e.bruni/resources/MEN.zip
$ unzip MEN.zip
```

#### SCWS
```sh
$ cd data/
$ wget http://www-nlp.stanford.edu/~ehhuang/SCWS.zip
$ unzip SCWS.zip
```

## Usage

#### Requirements
* gensim
* tqdm
* sklearn
* scipy

Download Google's Word2Vec,
```sh
$ wget https://storage.googleapis.com/google-code-archive-source/v2/code.google.com/word2vec/source-archive.zip
$ unzip source-archive.zip
$ cd source-archive/word2vec/trunk/
$ make
$ cd ../../../
$ mv source-archive/word2vec/trunk google-w2v/
```

#### Run
Toy corpora are provided in the ```corpora/``` directory. 
The corpus is expected in document (1 document per line) and sentence (1 sentence per line) formats with extensions ```.doc``` and ```.sent``` respectively.
A file with stopwords is provided: ```stopwords.txt```

Run the main script,
```sh
$ python3 topic_dsms.py --corpus_dir ../corpora/ \
                        --topics 50 \
                        --docs toy.doc \
                        --sents toy.sent \
                        --save_dir ../toy_corpus_out/ \
                        --dataset ../data/MEN/MEN_dataset_natural_form_full \
                        --name MEN \
                        --construct
```

In order to perform fusion of multiple topic-based DSM groups generated from different topic models, run the following script.
The script expects that the topic models and the respective topic-DSMs are already constructed.
```sh
python3 fusion.py --corpus_dir ../corpora/ \
                  --topics 20 30 40 \
                  --docs toy.doc \
                  --sents toy.sent \
                  --save_dir ../toy_corpus_out/ \
                  --dataset ../data/SCWS/ratings.txt \
                  --context \
                  --name SCWS
```

#### Citation
Please cite the following paper when using this software.
> @inproceedings{christopoulou2018mixture,
>  title={Mixture of topic-based distributional semantic and affective models},
>  author={Christopoulou, Fenia and Briakou, Eleftheria and Iosif, Elias and Potamianos, Alexandros},
>  booktitle={2018 IEEE 12th International Conference on Semantic Computing (ICSC)},
>  pages={203--210},
>  year={2018},
>  organization={IEEE}
> }



