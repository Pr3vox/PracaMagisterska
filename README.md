# PracaMagisterska
# Klasyfikacja zanieczyszczeń suszu herbacianego przy użyciu głębokich sieci konwolucyjnych

Praca magisterska przedstawia model uczenia maszynowego, który na podstawie zdjęcia suszu rozpoznaje, czy jest on czysty, czy zanieczyszczony krzyżowo lub materiałem organicznym.



# Pliki

| Plik | Opis |
|---|---|
| `main.py` | trening modelu i liczenie wyników |
| `Test_modelu.py` | test modelu na pojedynczym zdjęciu |
| `model_lepszy.h5` | zapisany wytrenowany model |
| `pyproject.toml` | lista bibliotek i ich wersji |
| `Podział_Kategorie/` | zdjęcia (foldery `Czyste`, `Krzyżowe`, `Organiczne`) |

## Dane

- Łącznie **126 zdjęć**, 3 klasy

Podział: 
- Czyste - 56
- Krzyżowe - 35
- Organiczne - 35


## Ustawienia treningu

| Parametr | Wartość |
|---|---|
| Rozmiar zdjęcia | 224 x 224 |
| Liczba epok | 10 |
| Optymalizator | Adam |

Do treningu użyto 80% zdjęć (101), do walidacji 20% (25).

## Wyniki (zbiór walidacyjny, 25 zdjęć)

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
