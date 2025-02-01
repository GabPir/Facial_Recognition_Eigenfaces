import os
import numpy as np
import matplotlib.pyplot as plt
from scipy.sparse.linalg import eigsh


# FUNZIONE PER LA SEPARAZIONE DELLE IMMAGINI IN TRAINING E TEST
def training_test_sets(path, num_immagini_training=6):
    num_immagini_test = 10 - num_immagini_training
    num_individui=40 
    immagini_training = []
    immagini_test = []
    ID_individui_training = []
    ID_individui_test = []

    for individuo in range(1, num_individui + 1):
        individuo_path = os.path.join(path, f"s{individuo}")
        immagine_individuo = []
        ID_individui = []

        for image_num in range(1, num_immagini_training+num_immagini_test + 1):
            img_path = os.path.join(individuo_path, f"{image_num}.pgm")
            ID_individui.append(f"{individuo}")
            img = plt.imread(img_path)
            img_array = np.array(img)
            img_vector = img_array.flatten()
            immagine_individuo.append(img_vector)

        immagini_training.extend(immagine_individuo[:num_immagini_training])
        immagini_test.extend(immagine_individuo[num_immagini_training:])
        ID_individui_training.extend(ID_individui[:num_immagini_training])
        ID_individui_test.extend(ID_individui[num_immagini_training:])
    return np.array(immagini_training), np.array(immagini_test), ID_individui_training, ID_individui_test, img.shape      #img.shape sarà la dimensione delle immagini 




# FUNZIONE PER IL RICONOSCIMENTO DELL'IMMAGINE
def riconoscimento_immagine(soglia, im_new, autofacce, media_training_proj, faccia_media):
    im_centrata = im_new - faccia_media
    im_proj =  autofacce @ (autofacce.T @ im_centrata)
    media_training_proj = media_training_proj.T      #le medie delle immagini di training sono lungo le colonne => trasponiamo lungo le righe
    ID_media = np.arange(1,41)
    distanza = np.linalg.norm(im_proj - im_centrata)   
    if distanza < soglia:
        minima_distanza = np.linalg.norm(im_proj - media_training_proj[0,:])       
        minimo_i = ID_media[0]
        for i, im in zip(ID_media[1:], media_training_proj[1:,:]):
            distanza_temp = np.linalg.norm(im_proj-im)   
            if minima_distanza > distanza_temp:
                minima_distanza = distanza_temp
                minimo_i = i
        #print('L\'immagine viene riconosciuta come: ', minimo_i)
        return minimo_i
    else: 
        #print("Immagine non riconosciuta")
        return None




# FUNZIONE PER L'IMPLEMENTAZIONE DELL'ALGORITMO
def classificazione_immagini(position, num_immagini_training, theta, k):  
    
    # TRASFORMIAMO LE IMMAGINI IN VETTORI
    training_set, test_set, ID_training, ID_test, (m, n)  = training_test_sets(position, num_immagini_training)           # mxn è la dimensione delle immagini
    #print(f'Le dimensioni delle immagini sono {m}*{n}')

    ''' OPZIONALE:
    #CONTROLLARE CHE LE IMMAGINI SIANO IN SCALA DI GRIGI
    if np.logical_and(np.all(training_set)>=0, np.all(training_set)<=255):
        print('Le immagini di training sono in scala di grigi.')    
    else:
        print('Le immagini di training non sono in scala di grigi.')

    if np.logical_and(np.all(test_set)>=0, np.all(training_set)<=255):
        print('Le immagini di test sono in scala di grigi.') 
    else:
            print('Le immagini di test non sono in scala di grigi.')
    '''       
    
    
    # CALCOLO DELLA FACCIA MEDIA DEL TRAINING SET
    faccia_media = np.mean(training_set, axis = 0)


    # SOTTRAIAMO LA FACCIA MEDIA ALLE IMMAGINI DI TRAINING
    training_set_centrato = training_set - faccia_media
    matcov = (1/training_set_centrato.shape[0]) * np.dot(training_set_centrato.T, training_set_centrato)


    # CALCOLOLO AUTOVETTORI MATRICE COVARIANZA, PRENDENDO SOLO k AUTOVETTORI IN BASE AI k AUTOVALORI PIU' GRANDI
    L = training_set_centrato.shape[0]  #numero immagini
    eigenvalues, eigenvectors = eigsh(matcov, k = L, which = 'LM')
    indici_ordinati = np.argsort(eigenvalues)[::-1]
    autovalori_ordinati = eigenvalues[indici_ordinati]
    autovettori_ordinati = eigenvectors[:, indici_ordinati]


    # PLOT AUTOVALORI
    plt.figure()
    plt.plot((autovalori_ordinati))
    plt.ylabel('Valore degli autovalori')
    plt.title('Autovalori ordinati in ordine decrescente')
    plt.xticks(np.arange(0,len(training_set_centrato)+1,20))
    plt.grid()
    plt.show()

      
    # SELEZIONE AUTOVETTORI ED AUTOVALORI (in base a k, n°componenti principali)
    #selected_eigenvalues = autovalori_ordinati[0 : k]
    autovettori_scelti = autovettori_ordinati[:,0 : k]


    # PROIEZIONE IMMAGINI DI TRAINING SUL SOTTOSPAZIO DI k COMPONENTI (Le immagini sono per riga, mentre gli autovettori per colonna)
    matrice_proiezioni = autovettori_scelti @ (autovettori_scelti.T @ training_set_centrato.T)       
    
    
    # SI RITRASFORMA QUESTE PROIEZIONI IN IMMAGINI
    immagini_ricostruite = matrice_proiezioni + faccia_media[:, np.newaxis]
    immagini_ricostruite = immagini_ricostruite.T.reshape((-1, m, n))


    # VISUALIZZAZIONE DI 15 IMMAGINI RICOSTRUITE:
    plt.figure(figsize=(m, n))
    for i in range(24):
        plt.subplot(4, 6, i+1)
        plt.imshow(immagini_ricostruite[i], cmap='gray')
        plt.axis('off')
    plt.show()
    
    
    # CALCOLO MEDIE delle IMMAGINI PROIETTATE PER CIASCUN INDIVIDUO (TRAINING)
    media_training_proj = np.zeros((40, matrice_proiezioni.shape[0]))
    for i in range(40):
        media_training_proj[i, :] = np.mean(matrice_proiezioni[:,i*num_immagini_training:i*num_immagini_training+num_immagini_training], axis = 1)    
 
    media_training_proj = media_training_proj.T      # Trasponendo le immagini saranno lungo le colonne
    media_ricostruita = media_training_proj + faccia_media[:, np.newaxis]
    media_ricostruita = media_ricostruita.T.reshape((-1, m, n))
    
    
    # VISUALIZZAZIONE DELLE PRIME 15 IMMAGINI MEDIE RICOSTRUITE:
    plt.figure(figsize=(m, n))
    for i in range(15):
        plt.subplot(3, 5, i+1)
        plt.imshow(media_ricostruita[i], cmap='gray')
        plt.axis('off')
        plt.title("media soggetti")
    plt.show()
    
    
    # RISULTATI RICONOSCIMENTO/CLASSIFICAZIONE
    corretti = 0;
    for i_test, im_test in zip(ID_test, test_set):
        risultato = riconoscimento_immagine(theta, im_test, autovettori_scelti, media_training_proj, faccia_media)
        if risultato != None:
            #print('L\'immaine dovrebbe corrispondere a: ', i_test)
            if  int(i_test) == int(risultato):
                corretti = corretti + 1;
    #print(f"\nNumero classificati correttamente {corretti} su {len(ID_test)} ({corretti/len(ID_test)}%)")
    
    return corretti/len(ID_test)        


#%%
#################################
### IMPLEMENTAZIONE ALGORITMO ###

posizione = 'C:/Documenti/Desktop/analisi tempo frequenza/HOMEWORK/archive'
n_im_training_per_individuo = 6
soglia_riconoscimento = 2500      #SOGLIA sopra la quale le immagini non vengono riconosciute
num_componenti_principali = 20

classificazione_immagini(posizione,  n_im_training_per_individuo, soglia_riconoscimento, num_componenti_principali)


#%%
###################
### GRID SEARCH ###
posizione = 'C:/Documenti/Desktop/analisi tempo frequenza/HOMEWORK/archive'
n_training = [4, 5, 6, 7, 8]
soglie = [1500, 2000,2500,3000, 3500]
n_cp = [10,20,30,60,100]


# SELEZIONE DEI PARAMETRI MIGLIORI
tupla_parametri_migliori = (None, None, None)

perc_migliore = classificazione_immagini(posizione, n_training[0], soglie[0], n_cp[0])
for i in n_training[1:]:
    for j in soglie[1:]:
        for k in n_cp[1:]:
             percentuale = classificazione_immagini(posizione, i, j, k)
             if percentuale > perc_migliore:
                 perc_migliore = percentuale
                 tupla_parametri_migliori = (i, j, k)
            
print('La combinazione dei parametri migliore è: ', tupla_parametri_migliori, 'con frazione corretti pari a: ', perc_migliore)





#%%
################################
### VARIAZIONE DEI PARAMETRI ###
posizione = 'C:/Documenti/Desktop/analisi tempo frequenza/HOMEWORK/archive'
n_training = [4, 5, 6, 7, 8]
soglie = [1500, 1750, 2000, 2250, 2500, 2750, 3000, 3250, 3500]
n_cp = [10, 20, 30, 40, 50, 60, 70, 80, 90, 100]


# VARIANDO N_TRAINING:
percentuale_corretta = []
for i in n_training:
    percentuale_corretta.append(classificazione_immagini(posizione, i, 3000, 20))
print("\nFrazione corretti al variare del numero di immagini di training:\n", percentuale_corretta)

plt.figure()
plt.plot(n_training,percentuale_corretta)
plt.ylabel('Percentuale corretti')
plt.xlabel('Num. campioni training set per individuo')
#plt.title('% classificati correttamente vs numero di dati di training')
plt.xticks([4, 5, 6, 7, 8])
plt.grid()
plt.show()


# VARAINDO LA SOGLIA: 
percentuale_corretta = []
for j in soglie:
    percentuale_corretta.append(classificazione_immagini(posizione, 7, j, 20))
print("\nFrazione corretti al variare della soglia:\n", percentuale_corretta)

plt.figure()
plt.plot(soglie,percentuale_corretta)
plt.ylabel('Percentuale corretti')
plt.xlabel('Soglia')
#plt.title('% classificati correttamente vs soglia')
plt.grid()
plt.show()


# VARIANDO IL NUMERO DI COMPONENTI PRINCIPALI: 
percentuale_corretta = []
for k in n_cp:
    percentuale_corretta.append(classificazione_immagini(posizione, 7, 3000, k))
print("\nFrazione corretti al variare delle componenti principali:\n", percentuale_corretta)


plt.figure()
plt.plot(n_cp, percentuale_corretta)
plt.ylabel('Percentuale corretti')
plt.xlabel('Numero componenti principali')
#plt.title("classificati correttamente vs numero comp. principali")
plt.grid()
plt.show()
