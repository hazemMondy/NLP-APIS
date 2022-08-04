# NLP Grading and plagiarism - APIS

## 1. Setup

```
pip install -r requirments.txt

python src/models/model_download.py

python -m spacy download en_core_web_sm
```

## 2. Usage

### 2.1 *API side*
```python
!python src/main.py
```
### 2.1.1 *Plagiarsim*

```python

import requests as req


inp = {"essays_dict": 
   {"p7": "carpooled overtone censurers Gorey tyrannises",
   "uv8": "amphimixis Oozlefinch mesmerize attaman untrue",
   "7mgj": "remodelled unitholders toddled uncousinly pug-dog",
   "evjd2": "shortness tupara potato-moth proximates digger",
   "yDim":  "amphimixis mesmerize attaman untrue"
   },
  "cased": false}

out = req.post(url, json=inp)

print(out.json())
```

```python

{
    "plagiarism_results": [
        {
            "uv8": {
                "yDim": 0.7832804918289185
            }
        },
        {
            "yDim": {
                "uv8": 0.7832803726196289
            }
        }
    ]
}

```

### 2.1.2 *Grading*

```python

import requests as req

inp = {"essays_dict": {
        "78vv74": "\"\"@increase@\"\"1.0",
        "9": "name all is good hi gallon increase",
        "10": "all is good dafdr",
        "11": "all is decrease dafdr"
    },
    "grading_model": "manual_keywords_grading",
    "cased": false
    }

out = req.post(url, json=inp)

print(out.json())
```

```python

{
    "grades": {
        "9": 1.0,
        "10": 0.0,
        "11": 0.0
    }
}

```

### 2.2 *Individual classes*

```python

# has the encoder singleton
import transformer_model
# for grading
import keywords_grading

bert = transformer_model()
KGM = keywords_grading(bert)

answers = ["ahmed is eating a pizza",
        "i go to school by bus",
        "i am eating a pizza",
        "ahmed is eating a pizza"
        ]
ids = [i in range(len(answers))]

# call the pipeline
KGM.predict(answers= answers ,
        ids = ids,
        )
```

```python
{
    "grades": {
        "9": 0.177,
        "10": 0.475,
        "11": 0.93
    }
}
```

## 3. Data

### 3.1 Grading

>ASAP dataset

### 3.1 Plagiarism

> quora question pairs

> arXiV
