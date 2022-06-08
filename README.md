# en-vi-nli-xlm-roberta-large-4-last-layers

# Dataset:
## With English: I used NLTK to preprocess

### English premise length distribution
![en_premise_length_distribution](https://user-images.githubusercontent.com/101851984/172543158-8978be27-86ad-4990-9e09-fd21371d8cda.png)
### Vietnamese hypothesis length distribution
![hypothesis_length_distribution](https://user-images.githubusercontent.com/101851984/172543638-f1ccc80e-d908-428e-8c79-8256a86f07f0.png)
### English premise length + Vietnamese hypothesis length distribution
![en-vi-length-distribution](https://user-images.githubusercontent.com/101851984/172543744-ff236083-4bca-46d0-895c-b4d9842226c9.png)

## With Vietnamese: I used Underthesea to preprocess
### Vietnamese premise length distribution
![vi_premise_length_distribution](https://user-images.githubusercontent.com/101851984/172543957-3ca3a7ee-92d8-4707-b12f-7a90374d8ea9.png)
### Vietnamese premise length + Vietnamese hypothesis length distribution
![vi-total](https://user-images.githubusercontent.com/101851984/172544045-88e7a7ce-ab21-4225-88a5-21b9ac4ef8c4.png)

However, I did not take all samples. I only took which sample total length is smaller than or equal 100 tokens and greater than 0

The training dataset:
| Num samples | premise language | hypothesis language |
|-------------|------------------|---------------------|
| 390762      | English          | Vietnamese          |
| 387164      | Vietnamese       | Vietnamese          |

The validation dataset:
| Num samples | premise language | hypothesis language |
|-------------|------------------|---------------------|
| 2490        | English          | Vietnamese          |
| 2490        | Vietnamese       | Vietnamese          |

The testing dataset:
| Num samples | premise language | hypothesis language |
|-------------|------------------|---------------------|
| 5010        | English          | Vietnamese          |
| 5010        | Vietnamese       | Vietnamese          |

Because of the limitation of GPU Tesla T4, I only trained random 60k samples during training section.
| Num samples | premise language | hypothesis language |
|-------------|------------------|---------------------|
| 30170       | English          | Vietnamese          |
| 29830       | Vietnamese       | Vietnamese          |

# Parameters

| Parameter               |                        |
|-------------------------|------------------------|
| Max length              | 100                    |
| epochs                  | 4                      |
| warmup steps            | 200                    |
| optimizer               | AdamW                  |
| lr                      | 1e-5                   |
| eps                     | 1e-8                   |
| concat 4 last layers                             |

# Performance
## With en-vi pairs
|              | precision | recall   | f1-score | support |
|--------------|-----------|----------|----------|---------|
| entailment   | 0.88302   | 0.84072  | 0.86135  | 1670    |
| neutral      | 0.81568   | 0.81617  | 0.81592  | 1670    |
| contradiction| 0.85535   | 0.89581  | 0.87511  | 1670    |
| accuracy     |           |          | 0.85090  | 5010    |
| macro avg    | 0.85135   | 0.85090  | 0.85079  | 5010    |
| weighted avg | 0.85135   | 0.85090  | 0.85079  | 5010    |

## With vi-vi pairs
|              | precision | recall   | f1-score | support |
|--------------|-----------|----------|----------|---------|
| entailment   | 0.85434   | 0.75509  | 0.80165  | 1670    |
| neutral      | 0.73233   | 0.81916  | 0.77332  | 1670    |
| contradiction| 0.84334   | 0.84132  | 0.84233  | 1670    |
| accuracy     |           |          | 0.80519  | 5010    |
| macro avg    | 0.81000   | 0.80519  | 0.80577  | 5010    |
| weighted avg | 0.81000   | 0.80519  | 0.80577  | 5010    |

# Future works
- Train with the entire dataset
- Set max length as 256
- Apply RXF at https://arxiv.org/pdf/2008.03156v1.pdf
