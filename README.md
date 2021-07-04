# cnn_ships

Članovi tima:
Šašić Natalija SW-14/2018, grupa 1

Asistenti:
Veljko Maksimović

Problem koji se rešava:
Klasifikacija brodova na osnovu slike u 5 glavnih kategorija

Algoritam:
Konvolucione neuronske mreže

Dataset:
Skup podataka koji će se koristiti se nalazi na sledećem linku:
https://www.kaggle.com/arpitjain007/game-of-deep-learning-ship-datasets
Skup sadrži slike brodova koji se dele u 5 kategorija:
Cruiser - Kruzer
Tanker - Tanker
Carrier - Nosač
Military - Vojni brod
Cargo - Teretni brod

Metrika za merenje performansi:
Za metriku se koristi procenat uspešno klasifikovanih slika, odnosno accuracy.

Validacija rešenja:
Dataset sadrži podelu na 6252 (70%) slike iz trening skupa,893(10%) slike iz validacionog skupa i 1787 (20%) slika iz test skupa.


Pokretanje aplikacije:
Pokretanje aplikacije se moze uraditi iz PyCharm IDE. U sklopu projekta je koriscen dataset sa linka https://www.kaggle.com/arpitjain007/game-of-deep-learning-ship-datasets. Kada je dataset preuzet sa interneta, u source-kodu promeniti putanje do navedenog skupa podataka.
Takođe se može koristiti i već sačuvani model i nad njim testirati sistem. Potrebno skinuti biblioteke: pandas, numpy, cv2, sklearn, tensorflow, keras i u okviru njega- Conv2D, MaxPooling2D, Flatten, Dropout, Dense.

