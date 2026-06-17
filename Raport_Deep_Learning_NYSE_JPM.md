**Predictia Pretului Actiunilor folosind Retele Neuronale LSTM**

NYSE 2001-2025 \| JPMorgan Chase (JPM)

**Proiect Deep Learning - Rezultate si Interpretari\**
Materie: Retele neuronale si tehnici de Deep Learning\
Set de date: NYSE (New York Stock Exchange)\
Ticker analizat: JPM (JPMorgan Chase & Co.)\
Perioada: 2001-2025 (932 zile de test)

# 1. Introducere

Acest proiect implementeaza un model de retea neuronala recurenta de tip LSTM (Long Short-Term Memory) pentru predictia pretului de inchidere al actiunilor JPMorgan Chase (JPM) tranzactionate pe bursa NYSE.

Obiectivul principal este de a prezice pretul Close al zilei urmatoare (horizon = 1 zi) pe baza unei ferestre de 15 zile de tranzactionare anterioare, utilizand atat variabilele OHLCV (Open, High, Low, Close, Volume), cat si indicatori tehnici derivati din acestea.

# 2. Descrierea Setului de Date

## 2.1 Sursa si structura

Setul de date contine inregistrarile istorice de tranzactionare de pe NYSE pentru perioada 1 ianuarie 2001 - 31 decembrie 2025. Fiecare zi de tranzactionare este stocata intr-un fisier CSV separat, continand pentru fiecare simbol listat urmatoarele campuri: Symbol, Date, Open, High, Low, Close, Volume.

## 2.2 Selectia ticker-ului

A fost selectat ticker-ul JPM (JPMorgan Chase & Co.), una dintre cele mai mari institutii financiare din lume, cu o capitalizare de piata de peste 500 miliarde USD. In urma extragerii datelor, s-au obtinut 6,470 de zile de tranzactionare, cu preturi Close cuprinse intre \$102.92 si \$327.58.

## 2.3 Impartirea datelor

Setul de date a fost impartit cronologic (nu aleatoriu, pentru a respecta structura temporala) in trei subseturi:

  ---------------------------------------------------------------------------
  **Set**           **Perioada**          **Nr. zile**      **Procent**
  ----------------- --------------------- ----------------- -----------------
  Antrenare         Mar 2001 - Mai 2018   4,480             70%

  Validare          Iun 2018 - Mar 2022   947               15%

  Testare           Mar 2022 - Dec 2025   947               15%
  ---------------------------------------------------------------------------

# 3. Preprocesarea Datelor si Feature Engineering

## 3.1 Variabile de intrare (Features)

Modelul utilizeaza 9 variabile de intrare, combinand datele brute de piata cu indicatori tehnici calculati:

**• Open, High, Low, Close, Volume:** Preturile brute si volumul zilnic de tranzactionare. Ofera informatia de baza despre dinamica pietei.

**• LogReturn:** Randamentul logaritmic zilnic: ln(Close_t / Close\_{t-1}). Stabilizeaza varianta si transforma seria intr-una stationara.

**• HL_Range:** Diferenta High - Low. Masoara volatilitatea intraday si amplitudinea miscarilor de pret.

**• SMA_5:** Media mobila simpla pe 5 zile. Captureaza trendul pe termen scurt si reduce zgomotul.

**• RV_5:** Volatilitatea realizata pe 5 zile (deviatia standard a randamentelor). Indicator de risc si incertitudine.

## 3.2 Normalizare

Toate variabilele de intrare si iesire au fost normalizate folosind StandardScaler (medie 0, deviatie standard 1). Scaler-ul a fost antrenat doar pe setul de train pentru a evita data leakage. Valorile prezise sunt apoi readuse la scara originala pentru evaluare si interpretare.

## 3.3 Crearea secventelor

Pentru a alimenta modelul LSTM, datele au fost transformate in secvente sliding window de 15 zile. Fiecare secventa de 15 zile consecutive (9 features x 15 pasi temporali) este utilizata pentru a prezice pretul Close din ziua urmatoare (t+1). Aceasta abordare permite modelului sa invete pattern-uri temporale pe termen scurt.

# 4. Arhitectura Modelului LSTM

## 4.1 Motivatie

LSTM (Long Short-Term Memory) a fost ales deoarece:

- Este specializat in modelarea dependentelor temporale pe termen lung si scurt, fiind ideal pentru serii financiare care prezinta autocorelatie.

- Mecanismul de gates (forget, input, output) previne problema vanishing gradient, permitand propagarea informatiei relevante pe distante temporale mari.

- Comparativ cu RNN-urile simple, LSTM retine selectiv informatia, eliminand zgomotul si pastrand semnalele predictive.

- In literatura de specialitate, LSTM este unul dintre cele mai utilizate modele pentru predictia seriilor financiare, cu rezultate superioare ARIMA si GARCH.

## 4.2 Structura

Modelul are urmatoarea configuratie:

  ----------------------------------------------------------------------------------------------------------
  **Input size**                      9 (Open, High, Low, Close, Volume, LogReturn, HL_Range, SMA_5, RV_5)
  ----------------------------------- ----------------------------------------------------------------------
  **Hidden size**                     32 neuroni in stratul LSTM

  **Numar straturi**                  1 strat LSTM

  **Dropout**                         0.1 (10% regularizare)

  **Output**                          1 (pretul Close prezis)

  **Parametri totali**                5,537 antrenabili

  **Functia de loss**                 MSE (Mean Squared Error) pentru regresie

                                      
  ----------------------------------------------------------------------------------------------------------

## 4.3 Hiperparametri de antrenare

  ----------------------------------------------------------------------------------
  **Epoci**                           100 (cu early stopping, patience=15)
  ----------------------------------- ----------------------------------------------
  **Batch size**                      8

  **Learning rate**                   5e-4 (0.0005) cu scheduler ReduceLROnPlateau

  **Optimizator**                     Adam (adaptive moment estimation)

  **Gradient clipping**               max norm = 1.0

                                      
  ----------------------------------------------------------------------------------

# 5. Rezultatele Antrenarii si Evaluarii

## 5.1 Evolutia functiei de pierdere

Graficul de mai jos prezinta evolutia functiei de pierdere (MSE) pe seturile de antrenare si validare pe parcursul epocilor de antrenare.

![](./media/image1.png){width="5.5in" height="2.75in"}

Se observa o scadere rapida a pierderii in primele 10-15 epoci, urmata de o stabilizare. Diferenta dintre loss-ul de train si cel de validare indica un anumit nivel de overfitting, dar early stopping-ul a prevenit degradarea semnificativa a performantei pe datele de validare.

## 5.2 Predictii vs Valori Reale

![](./media/image2.png){width="5.5in" height="2.3571423884514435in"}

Graficul compara valorile reale ale pretului Close (albastru) cu predictiile modelului LSTM (portocaliu) pe setul de test (martie 2022 - decembrie 2025). Modelul captureaza tendinta generala de crestere, insa subestimeaza amplitudinea miscarilor de pret, producand predictii cu volatilitate redusa. Aceasta este o limitare comuna a modelelor bazate pe MSE, care penalizeaza deviatiile mari si favorizeaza predictii conservative.

## 5.3 Graficul de dispersie (Predictii vs Reale)

![](./media/image3.png){width="4.5in" height="4.5in"}

Graficul scatter plaseaza fiecare predictie in functie de valoarea reala corespunzatoare. Linia rosie punctata reprezinta predictia perfecta (y=x). Punctele situate sub linie indica subestimari, iar cele deasupra - supraestimari. Se observa o concentrare a predictiilor in intervalul \$100-\$150, ceea ce confirma tendinta modelului de a produce predictii conservatoare. Pentru valori reale peste \$200, modelul subestimeaza sistematic.

## 5.4 Distributia erorilor

![](./media/image4.png){width="5.0in" height="2.5in"}

Histograma erorilor de predictie (Actual - Predictie) arata o distributie deplasata semnificativ spre valori pozitive, ceea ce indica o subestimare sistematica a pretului. Distributia nu este perfect simetrica in jurul lui 0, sugerand ca modelul are dificultati in a captura variatia completa a preturilor. Acest bias poate fi corectat prin tehnici suplimentare de regularizare sau prin utilizarea unei functii de loss asimetrice.

# 6. Metrici de Performanta

Performanta modelului a fost evaluata folosind mai multe metrici complementare, fiecare capturand un aspect diferit al calitatii predictiilor:

  ---------------------------------------------------------------------------------------------------------------------------------------
  **Metrica**                 **Valoare**             **Interpretare**
  --------------------------- ----------------------- -----------------------------------------------------------------------------------
  MSE (Mean Squared Error)    9.08                    Eroarea patratica medie. Penalizeaza puternic erorile mari.

  RMSE (Root MSE)             3.01 USD                Eroarea medie in aceleasi unitati ca pretul (\~1.6% din pretul mediu).

  MAE (Mean Absolute Error)   2.01 USD                Eroarea absoluta medie. Mai robusta la valori extreme decat MSE.

  MAPE (Mean Abs % Error)     1.06%                   Eroarea procentuala medie. Cu cat mai mica, cu atat mai bine.

  R² (Coef. de determinare)   0.9978                  Valori negative indica performanta mai slaba decat un model constant (media).

  Acuratete directionala      70.25%                  Procentul de predictii corecte ale directiei (crestere/scadere). 50% = aleatoriu.

  Bias sistematic             0.31 USD                Media erorilor. Pozitiv = modelul subestimeaza sistematic.
  ---------------------------------------------------------------------------------------------------------------------------------------

## 6.1 Interpretarea metricilor

Modelul LSTM cu 5,537 parametri obtine un RMSE de \$3.01 si un MAE de \$2.01 pe setul de test (947 zile). MAPE-ul de 1.06% indica faptul ca, in medie, predictia se abate cu aproximativ un sfert de la valoarea reala.

Coeficientul de determinare R² este 0.9978, ceea ce inseamna ca modelul nu reuseste sa explice variatia preturilor mai bine decat media istorica. Acest rezultat este frecvent intalnit in predictia financiara din cauza naturii aproape aleatorii (random walk) a pietelor eficiente.

Acuratetea directionala de 70.25% este apropiata de 50% (nivelul aleatoriu), ceea ce confirma dificultatea intrinseca a predictiei directiei preturilor pe piete financiare lichide.

# 7. Discutii si Concluzii

## 7.1 Limitari identificate

**Subestimarea volatilitatii:** Modelul produce predictii cu variatie redusa, concentrandu-se in jurul mediei. Aceasta este o consecinta directa a utilizarii MSE ca functie de loss, care favorizeaza estimari conservatoare.

**Bias sistematic:** Media erorilor de \$0.31 indica o subestimare constanta a preturilor reale, probabil cauzata de trendul ascendent pe termen lung al actiunii JPM.

**Complexitate redusa a modelului:** Cu doar 32 de neuroni ascunsi si un singur strat LSTM, modelul are o capacitate limitată de a captura pattern-uri complexe pe termen lung.

**Lipsa variabilelor macroeconomice:** Modelul foloseste exclusiv date de pret si volum, ignorand factori externi precum dobanzile, inflatia, stirile financiare sau evenimentele geopolitice.

**Fereastra de intrare scurta:** Cele 15 zile de context pot fi insuficiente pentru a captura cicluri economice sau tendinte pe termen lung.

## 7.2 Posibile imbunatatiri

- Utilizarea unui model mai complex (ex: 2 straturi LSTM cu 128 neuroni, GRU, sau Transformers) pentru a creste capacitatea de invatare.

- Adaugarea de features suplimentare: RSI, MACD, Bollinger Bands, volume profile, indicatori macroeconomici (dobanzi, inflatie, VIX).

- Extinderea ferestrei de intrare la 60 sau 120 de zile pentru a captura tendinte pe termen lung.

- Utilizarea unei functii de loss asimetrice (ex: Quantile Loss, Huber Loss) care sa penalizeze diferit subestimarea si supraestimarea.

- Implementarea unui mecanism de atentie (Attention) peste iesirile LSTM pentru a permite modelului sa se concentreze pe cele mai relevante momente din trecut.

- Antrenarea cu validare walk-forward (backtesting) pentru a simula conditii reale de trading.

- Explorarea modelelor hibride (LSTM + CNN, LSTM + XGBoost) care combina invatarea secventiala cu feature engineering bazat pe arbori.

## 7.3 Concluzii finale

Proiectul demonstreaza aplicabilitatea retelelor LSTM pentru predictia seriilor financiare, evidentiind atat potentialul, cat si limitarile acestei abordari. Desi modelul reuseste sa captureze tendinta generala a pretului, predictiile punctuale raman imprecise din cauza naturii stochastic inerente a pietelor financiare.

Principala concluzie este ca, in forma sa actuala, modelul este mai potrivit pentru analiza de trend si identificarea directiei generale a pietei decat pentru predictii exacte ale pretului. Imbunatatirile propuse (mai multe features, arhitecturi complexe, mecanisme de atentie) ar putea creste semnificativ performanta.

Proiectul ofera o baza solida pentru explorari ulterioare si demonstreaza implementarea practica a intregului pipeline de Deep Learning: de la incarcarea si preprocesarea datelor, trecand prin feature engineering si antrenare, pana la evaluare, vizualizare si interpretarea rezultatelor.

# 8. Tehnologii Utilizate

  ---------------------------------------------------------------------------------------------------
  **Python 3.14**                     Limbajul de programare principal al proiectului
  ----------------------------------- ---------------------------------------------------------------
  **PyTorch**                         Framework de Deep Learning pentru antrenarea retelelor LSTM

  **Pandas**                          Manipularea si procesarea datelor tabulare si serii temporale

  **NumPy**                           Operatii numerice eficiente pe array-uri multidimensionale

  **Scikit-learn**                    Preprocesare (StandardScaler) si metrici de evaluare

  **Matplotlib**                      Generarea graficelor de analiza si vizualizare a rezultatelor

  **python-docx**                     Generarea automata a acestui raport in format Word

                                      
  ---------------------------------------------------------------------------------------------------
