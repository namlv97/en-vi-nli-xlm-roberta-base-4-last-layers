# en-vi-nli-xlm-roberta-large-4-last-layers

# Dataset:
With English: I used NLTK to preprocess

With Vietnamese: I used Underthesea to preprocess

However, I did not take all samples. I only took which sample length is smaller than or equal 100 tokens and greater than 0

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
