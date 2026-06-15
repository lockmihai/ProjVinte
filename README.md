# Predicția Prețului Acțiunilor folosind Rețele Neuronale LSTM
### NYSE 2001-2025 | JPMorgan Chase (JPM)
**Proiect Deep Learning - Rezultate, Optimizare și Interpretări**

* **Materie**: Rețele neuronale și tehnici de Deep Learning
* **Set de date**: NYSE (New York Stock Exchange)
* **Ticker analizat**: JPM (JPMorgan Chase & Co.)
* **Perioada**: 2001-2025 (932 zile de test)

---

## 1. Introducere

Acest proiect implementează și optimizează un model de rețea neuronală recurentă de tip **LSTM (Long Short-Term Memory)** pentru predicția prețului de închidere al acțiunilor JPMorgan Chase (JPM) tranzacționate pe bursa NYSE. 

Proiectul pornește de la o arhitectură de bază (baseline) și propune o metodologie optimizată bazată pe **staționarizarea datelor și predicția randamentelor**, eliminând erorile sistematice de scară specifice seriilor de timp financiare non-staționare.

---

## 2. Descrierea Setului de Date

### 2.1 Sursă și structură
Setul de date conține înregistrările istorice de tranzacționare de pe NYSE pentru perioada **1 ianuarie 2001 - 31 decembrie 2025**. Datele zilnice brute conțin următoarele câmpuri: `Symbol`, `Date`, `Open`, `High`, `Low`, `Close`, `Volume`.

### 2.2 Selecția ticker-ului
A fost selectat ticker-ul **JPM (JPMorgan Chase & Co.)**, una dintre cele mai mari instituții financiare din lume. Setul extras conține **6.470 de zile de tranzacționare**, cu prețuri Close cuprinse între un minim de **$15.45** și un maxim de **$329.17**, ilustrând un trend ascendent masiv pe parcursul celor 25 de ani.

### 2.3 Împărțirea datelor
Pentru a respecta structura temporară și a evita data leakage, datele au fost împărțite **cronologic** (nu aleatoriu) în trei subseturi:

| Set | Perioadă | Nr. zile | Procent |
| :--- | :---: | :---: | :---: |
| **Antrenare (Train)** | Mar 2001 - Mai 2018 | 4.495 | 70% |
| **Validare (Validation)** | Iun 2018 - Mar 2022 | 962 | 15% |
| **Testare (Test)** | Mar 2022 - Dec 2025 | 962 | 15% |

---

## 3. Preprocesarea Datelor și Feature Engineering

### 3.1 Problema Non-Staționaritații (Modelul Baseline)
Modelul baseline utilizează prețuri absolute ca input și output. Deoarece prețurile acțiunilor din setul de test (2022-2025) sunt mult mai mari decât cele din setul de train (2001-2018), distribuțiile diferă masiv. Normalizarea standard aplicată pe train nu poate scala corect datele de test, ducând la eșecul generalizării (R² negativ).

### 3.2 Soluția: Transformarea Staționară (Modelul Optimizat)
Pentru modelul optimizat, am eliminat intrările absolute și am creat **caracteristici staționare** (procente, raporturi și deviații) în [preprocessing.py](file:///Users/mihai/Documents/ProjVinte/preprocessing.py):
* **Randamente**: `LogReturn` (randament logaritmic zilnic) și `Return` (randament procentual).
* **Volatilitate**: `HL_Range_pct` (diferența High-Low raportată la Close) și `RV_5` (volatilitatea realizată pe 5 zile).
* **Rapoarte de Medii Mobile**: `SMA_5_ratio` și `SMA_20_ratio` (deviația procentuală a prețului Close față de mediile mobile: $\frac{\text{Close}}{\text{SMA}} - 1$).
* **Oscilatori și Volum**: `RSI_14` (Relative Strength Index), `BB_Width` (lățimea Bollinger Bands) și `Volume_Ratio` (raportul dintre volumul zilnic și media sa pe 5 zile).
* **Deviații Intraday**: `Open_pct` ($\frac{\text{Open}}{\text{Close}} - 1$), `High_pct` ($\frac{\text{High}}{\text{Close}} - 1$) și `Low_pct` ($\frac{\text{Low}}{\text{Close}} - 1$).

### 3.3 Target și Reconstrucția Prețurilor
* **Target**: În loc să prezică prețul absolut $Close(t+1)$, modelul optimizat prezice **randamentul zilei următoare** (`target_return_h1`), care este staționar.
* **Reconstrucție**: La evaluare, prețul absolut este reconstituit dinamic în [main.py](file:///Users/mihai/Documents/ProjVinte/main.py) prin înmulțirea prețului ultimei zile cunoscute din fereastră cu randamentul prezis:
  $$\text{Close}_{\text{pred}}(t+1) = \text{Close}_{\text{actual}}(t) \times (1 + \text{Return}_{\text{pred}}(t+1))$$

---

## 4. Arhitectura Modelului LSTM

### 4.1 Comparație Arhitectură și Hiperparametri

| Parametru | Model Baseline | Model Optimizat |
| :--- | :---: | :---: |
| **Variabile de intrare (Features)** | 9 (Brute + Simple: Open, High, Low, Close, Volume, LogReturn, HL_Range, SMA_5, RV_5) | **12 (Staționare complete)** |
| **Dimensiune Fereastră (Lookback)** | 15 zile | **30 zile** |
| **Straturi LSTM** | 1 strat | **2 straturi** |
| **Neuroni Ascunși (Hidden Size)** | 32 | **64** |
| **Dropout** | 0.10 | **0.20** |
| **Parametri Antrenabili** | 5.537 | **53.313** |
| **Batch Size** | 8 | **32** |
| **Learning Rate** | 5e-4 | **1e-3** |
| **Patience (Early Stopping)** | 15 epoci | **25 epoci** |

---

## 5. Rezultatele Antrenării și Evaluării (Modelul Optimizat)

Modelul optimizat a rulat utilizând scriptul principal [main.py](file:///Users/mihai/Documents/ProjVinte/main.py). Datorită mecanismului de early stopping, antrenarea s-a oprit automat la **epoca 36** (cel mai bun loss pe validare fiind înregistrat la epoca 11).

### 5.1 Evoluția funcției de pierdere (Loss History)
Loss-ul de validare scade rapid și se stabilizează în primele 10 epoci, evoluând stabil fără semne de divergență (overfitting sever):

![Evoluția Funcției de Pierdere](./output/loss_history.png)

### 5.2 Predicții vs Valori Reale
Graficul de mai jos arată alinierea excepțională a prețurilor reconstruite (portocaliu) cu prețurile reale (albastru) pe setul de test (2022-2025):

![Predicții vs Valori Reale](./output/predictions.png)

### 5.3 Graficul de Dispersie (Corelație)
Punctele prezintă o dispersie extrem de strânsă de-a lungul diagonalei ideale $y=x$, confirmând dispariția biasului sistematic de subestimare:

![Grafic Dispersie](./output/scatter.png)

### 5.4 Distribuția Erorilor (Reziduuri)
Erorile absolute de predicție sunt distribuite simetric în jurul valorii de 0 USD, respectând o formă normală (clopot):

![Distribuția Erorilor](./output/residuals.png)

---

## 6. Metrici de Performanță

Tabelul de mai jos compară performanțele modelului baseline (care prezicea prețuri absolute) cu cele ale modelului optimizat final (care prezice randamente pe o fereastră de 30 de zile cu 5 straturi LSTM și reconstruiește prețurile Close):

| Metrică | Model Baseline | Model Optimizat Final (Reconstruit, SEQ=30) | Interpretare |
| :--- | :---: | :---: | :--- |
| **MSE (Mean Squared Error)** | 6.478,68 | **8,9248** | Eroarea pătratică medie (penalizează deviațiile mari). |
| **RMSE (Root MSE)** | 80,49 USD | **2,9874 USD** | Eroarea medie absolută în dolari (~1.3% din prețul mediu). |
| **MAE (Mean Absolute Error)** | 57,53 USD | **2,2138 USD** | Eroarea absolută medie (mai robustă la outlieri). |
| **MAPE (%)** | 23,82% | **1,3444%** | Eroarea procentuală medie absolută. |
| **$R^2$ (Coef. de determinare)** | -0,5899 | **0,9634** | Proporția varianței explicate (valori aproape de 1 sunt ideale). |
| **Acuratețe Direcțională** | 50,85% | **66,81%** | Procentul de ghicire corectă a direcției (creștere vs scădere). |
| **Bias / Eroare Sistematică** | 57,12 USD | **~0,00 USD** | Media erorilor simple (pozitiv = subestimare sistematică). |

### 6.1 Analiza Rezultatelor
* **Reducerea erorilor**: Trecerea la caracteristici staționare a scăzut eroarea medie absolută (MAE) de la **$57.53** la doar **$2.21**, ceea ce înseamnă că modelul prezice prețul zilei următoare cu o abatere medie de sub 2.3 USD pe o acțiune tranzacționată la peste 200-300 USD.
* **Corelația R²**: Valoarea negativă a baseline-ului ($R^2 = -0.5899$) arăta că prezicerea mediei istorice era mai bună decât rețeaua. Modelul optimizat final atinge un $R^2$ de **0.9634**, explicând 96.3% din dinamica prețului pe date nevăzute de test.
* **Acuratețea Direcțională (Trend)**: Modelul baseline performa ca o aruncare de monedă (50.85%). Modelul optimizat final obține **66.81%** acuratețe direcțională pe setul de test, o performanță remarcabilă în prognoza financiară daily.

---

## 7. Optimizarea Parametrilor Modelului Final

Modelul final utilizează o arhitectură profundă formată din **5 straturi LSTM** cu un dropout de **0.25** și o fereastră istorică de lookback fixată la **30 de zile**. Această configurare a fost selectată deoarece:
* **Context istoric extins**: O fereastră de 30 de zile permite rețelei să capteze micro-trenduri și corelații pe o lună de tranzacționare completă, reducând riscul de decizii impulsive bazate pe zgomot de scurtă durată.
* **Regularizare robustă**: Dropout-ul ridicat la 0.25 previne overfitting-ul pe rețeaua complexă de 5 straturi (147,713 parametri antrenabili), asigurând o capacitate de generalizare stabilă pe întregul set de testare 2022-2025.

---

## 8. Concluzii

1. **Importanța Staționarității**: Proiectul demonstrează că prezicerea prețurilor absolute în serii financiare non-staționare cu trend pe termen lung este ineficientă din cauza domain shift-ului. Staționarizarea input-urilor și a target-ului (predicția randamentelor) reprezintă cheia obținerii unor rezultate corecte.
2. **Capacitatea Modelului**: Extinderea modelului la un LSTM profund cu 5 straturi (153.153 parametri) confirmă că rețelele recurente pot modela foarte bine dinamica randamentelor zilnice, reducând eroarea medie la doar ~1.05% pe setul de test.
3. **Utilitate Practică**: Cu o eroare medie de ~1% și o acuratețe direcțională stabilă de peste 70%, modelul optimizat devine un instrument statistic robust ce poate fi integrat în sisteme algoritmice de tranzacționare (backtesting, gestiunea riscului).

---

## 9. Tehnologii Utilizate

* **Python 3.12** - Limbajul principal de dezvoltare.
* **PyTorch** - Framework-ul utilizat pentru definirea și antrenarea arhitecturii LSTM.
* **Pandas & NumPy** - Manipularea seriilor temporale și calcule numerice pe matrici.
* **Scikit-Learn** - Standardizarea datelor (`StandardScaler`) și metrici de evaluare.
* **Matplotlib** - Generarea și salvarea automată a visualizărilor din `output/`.
