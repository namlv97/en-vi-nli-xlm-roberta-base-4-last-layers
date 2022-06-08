# en-vi-nli-xlm-roberta-base-4-last-layers

# Parameters

| Parameter               |                        |
|-------------------------|------------------------|
| num training samples    | 60K (en-vi + vi-vi)    |
| num validation samples  | 4980 (en-vi) + (vi-vi) |
| num testing samples     | 5010 (en-vi) + (vi-vi) |
| Max length              | 100                    |
| epochs                  | 4                      |
| warmup steps            | 200                    |
| optimizer               | AdamW                  |
| lr                      | 1e-5                   |
| eps                     | 1e-8                   |
| concat 4 last layers                             |
