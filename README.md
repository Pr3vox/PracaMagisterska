# PracaMagisterska
# Klasyfikacja zanieczyszczeń suszu herbacianego przy użyciu głębokich sieci konwolucyjnych

Praca magisterska przedstawia model uczenia maszynowego, który na podstawie zdjęcia suszu rozpoznaje, czy jest on czysty, czy zanieczyszczony krzyżowo lub materiałem organicznym.



# Pliki

| Plik | Opis |
|---|---|
| `main.py` | trening modelu i liczenie wyników |
| `Test_modelu.py` | test modelu na pojedynczym zdjęciu |
| `model_1.h5` |  model 1 |
| `model_2.h5` |  model 2 |
| `pyproject.toml` | lista bibliotek i ich wersji |
| `Podział_Kategorie/` | zdjęcia (foldery `Czyste`, `Krzyżowe`, `Organiczne`) |

## Dane

- Łącznie **126 zdjęć**, 3 klasy

Podział: 
- Czyste - 56
- Krzyżowe - 35
- Organiczne - 35


## Ustawienia treningu

### Model 1

| Parametr | Wartość |
|---|---|
| Rozmiar zdjęcia | 224 x 224 |
| Liczba epok | 10 |
| Optymalizator | Adam |

### Model 2

| Parametr | Wartość |
|---|---|
| Rozmiar zdjęcia | 224 x 224 |
| Liczba epok | 30 |
| Optymalizator | Adam |
| Learning rate | 1e-4 |
| EarlyStopping | monitor: val_loss, patience: 6 |
| ReduceLROnPlateau | monitor: val_loss, factor: 0.5, patience: 3 |

Do treningu użyto 80% zdjęć (101), do walidacji 20% (25).

## Wyniki modelu 1 (zbiór walidacyjny, 25 zdjęć)

**Accuracy: 0.64**

| Klasa | Precision | Recall |
|---|---|---|
| Czyste | 0.6667 | 0.6667 |
| Krzyżowe | 0.5333 | 1.0000 |
| Organiczne | 1.0000 | 0.3636 |

### Macierz pomyłek


| Prawdziwa klasa | Model: Czyste | Model: Krzyżowe | Model: Organiczne | Razem zdjęć | |
|---|:---:|:---:|:---:|:---:|-|
| **Czyste** | **4** | 2 | 0 | 6 | 4 rozpoznane dobrze, 2 model uznał za Krzyżowe |
| **Krzyżowe** | 0 | **8** | 0 | 8 | wszystkie 8 rozpoznane dobrze |
| **Organiczne** | 2 | 5 | **4** | 11 | 4 rozpoznane dobrze, 2 uznane za Czyste, 5 za Krzyżowe |

## Wyniki modelu 2 (zbiór walidacyjny, 25 zdjęć)

**Accuracy: 0.64**

| Klasa | Precision | Recall |
|---|---|---|
| Czyste | 0.5000 | 0.6667 |
| Krzyżowe | 0.6000 | 0.7500 |
| Organiczne | 0.8571 | 0.5455 |

### Macierz pomyłek


| Prawdziwa klasa | Model: Czyste | Model: Krzyżowe | Model: Organiczne | Razem zdjęć | |
|---|:---:|:---:|:---:|:---:|-|
| **Czyste** | **4** | 2 | 0 | 6 | 4 rozpoznane dobrze, 2 model uznał za Krzyżowe |
| **Krzyżowe** | 1 | **6** | 1 | 8 | 6 rozpoznanych dobrze, 1 uznane za Czyste, 1 za Organiczne |
| **Organiczne** | 3 | 2 | **6** | 11 | 6 rozpoznanych dobrze, 3 uznane za Czyste, 2 za Krzyżowe |

## Przebieg trening


| Epoka | Accuracy (trening) | Loss (trening) | Val accuracy | Val loss | Learning rate |
|:---:|:---:|:---:|:---:|:---:|:---:|
| 1 | 0.3663 | 1.2848 | 0.2400 | 1.4724 | 1e-4 |
| 2 | 0.5149 | 1.0648 | 0.3200 | 1.0989 | 1e-4 |
| 3 | 0.5248 | 0.9926 | 0.4000 | 1.0346 | 1e-4 |
| 4 | 0.6139 | 0.8904 | 0.5200 | 0.9951 | 1e-4 |
| 5 | 0.6733 | 0.8294 | 0.4800 | 0.9875 | 1e-4 |
| 6 | 0.6634 | 0.7503 | 0.4800 | 0.9942 | 1e-4 |
| 7 | 0.7030 | 0.7368 | 0.4800 | 0.9182 | 1e-4 |
| 8 | 0.7030 | 0.7527 | 0.4800 | 0.9802 | 1e-4 |
| 9 | 0.6634 | 0.7642 | 0.5600 | 0.9221 | 1e-4 |
| 10 | 0.7921 | 0.6053 | 0.4400 | 0.8791 | 1e-4 |
| 11 | 0.7426 | 0.5952 | 0.4000 | 1.0159 | 1e-4 |
| 12 | 0.7327 | 0.6225 | 0.4800 | 0.8740 | 1e-4 |
| 13 | 0.8119 | 0.6098 | 0.4800 | 0.9042 | 1e-4 |
| 14 | 0.7921 | 0.5489 | 0.4800 | 0.8731 | 1e-4 |
| 15 | 0.7327 | 0.6456 | 0.4000 | 1.0494 | 1e-4 |
| 16 | 0.7921 | 0.5467 | 0.6000 | 0.9009 | 1e-4 |
| 17 | 0.8317 | 0.4704 | 0.5600 | 0.9025 | 1e-4 |
| 18 | 0.8416 | 0.5450 | 0.6000 | 0.8628 | 5e-5 |
| 19 | 0.7921 | 0.5137 | 0.6000 | 0.8982 | 5e-5 |
| 20 | 0.8614 | 0.4773 | 0.5600 | 0.9177 | 5e-5 |
| 21 | 0.8416 | 0.4505 | 0.6000 | 0.7978 | 5e-5 |
| 22 | 0.8317 | 0.5417 | 0.6800 | 0.7936 | 5e-5 |
| 23 | 0.8218 | 0.4288 | 0.6000 | 0.7954 | 5e-5 |
| 24 | 0.8416 | 0.4725 | 0.6800 | 0.7550 | 5e-5 |
| 25 | 0.8515 | 0.4672 | 0.6400 | 0.7467 | 5e-5 |
| 26 | 0.8020 | 0.5178 | 0.5600 | 0.8274 | 5e-5 |
| 27 | 0.8614 | 0.4345 | 0.6000 | 0.8040 | 5e-5 |
| 28 | 0.8614 | 0.4483 | 0.6000 | 0.8052 | 5e-5 |
| 29 | 0.8812 | 0.3844 | 0.6000 | 0.7762 | 2.5e-5 |
| 30 | 0.8812 | 0.4224 | 0.6000 | 0.7643 | 2.5e-5 |