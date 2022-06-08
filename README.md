# en-vi-nli-xlm-roberta-large-4-last-layers

# Dataset:
With English: I used NLTK to preprocess
With Vietnamese: I used Underthesea to preprocess
However, I will not take all samples. I only take which sample length is smaller than or equal 100 tokens and greater 0

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
