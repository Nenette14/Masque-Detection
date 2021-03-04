# -*- coding: utf-8 -*-
"""
Created on Wed Feb 17 17:58:04 2021

@author: clair
"""
# -*- coding: utf-8 -*-
"""
Created on Fri Jan 29 09:11:50 2021

@author: clair
"""
import cv2
import sys
import os
from matplotlib import pyplot as plt


#On effectue les tests sur tous les classifieurs créés (mymaskedfacedetetor 0, 1, 2 et 3 )
classCascade = cv2.CascadeClassifier('mymaskedfacedetector1.xml')


"""
Premiers tests avec des images venant de différentes sources
Affichage des rectangles pour vérifier la localisation du masque détecté
"""

imagePath = 'data/positive/positive_test/459-with-mask.jpg' #modifiable

#Affichage de l'image
image = cv2.imread(imagePath)
plt.imshow(image)

# Detection
faces = classCascade.detectMultiScale(
    image,
    scaleFactor=1.1,
    minNeighbors=10,#5 initialement
    minSize=(30, 30),
    flags = cv2.CASCADE_SCALE_IMAGE
)
print("Il y a {0} visage(s) masqué(s).".format(len(faces)))



# Dessine des rectangles autour des visages masqués trouvés
for (x, y, w, h) in faces:
    cv2.rectangle(image, (x, y), (x+w, y+h), (0, 255, 0), 2)
plt.imshow(image)
plt.show()

"""
(1) Test sur des images où le masque à été ajouté avec photoshop
('data/positive/positive_test')

Si notre classifieur fonctionne pour détecter ce type de masque
on s'attend à un score de 100% sur le jeu de données test
"""

"""
(2)Test sur des images de visages masqués avec des masques variés 
(différents de celui utilisé pour l'entraînement)
 ('data/positive/positive_varied_masks_test')
"""

path_test = 'data/positive/positive_test' #modifiable

files_test = os.listdir(path_test)

i = 0
presence_mask = 0
sum_masks = 0
max_masks = []
images_bug = []
names_images_bug = []
for name in files_test :
    
    image = cv2.imread(path_test+'/'+name)
    #gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) #optionnelle
    # Detection
    faces = classCascade.detectMultiScale(
        image,
        scaleFactor=1.1,
        minNeighbors=5,
        minSize=(30, 30),
        flags = cv2.CASCADE_SCALE_IMAGE
        )
    #Affichage des rectangles
    for (x, y, w, h) in faces:
        cv2.rectangle(image, (x, y), (x+w, y+h), (0, 255, 0), 2)
    plt.imshow(image)
    plt.show()
    
    #Analyse de la performance
    i+= 1
    sum_masks = sum_masks + len(faces)
    if len(faces)>= 1 : 
        presence_mask += 1
    
    #On cherche les bugs
    if len(faces)> 1 : 
        max_masks.append(len(faces)) #par curiosité on regarde toutes les images 'mal traitrées'
        bug = image
        images_bug.append(bug)
        names_images_bug.append(name)

"""
#affichage des images bug
for image in images_bug :  
    plt.imshow(image)
    plt.show()
"""   
print("Il y a {0} visage(s) masqué(s) détéctés sur {1} images traitées.".format(sum_masks,i))
print("La présence de masque(s) a été observée sur {0} images parmi les sur {1} images traitées.".format(presence_mask,i))
print("Le score du classifieur est donc de {0}".format(presence_mask/i))
print("Les nombres maximum de visages masqués sur une photo sont de {0}.".format(max_masks))

"""
Test sur des images négatives (sans visages masqués)
On en profitera pour remarquer la réaction du classifieur pour des images avec masque seul
"""

"""
(3) On regarde, sur un petit jeu de données,
si le classifieur détecte le masque seul. 
On remarque que le classifieur réagit bien, il ne considère pas l'objet 'masque' seul
"""
#path_test = 'data/negative/negative_masques_seuls'

"""
(4) On teste ensuite sur des images variées : visages sans masques / fruits / voitures ...
"""

# path_test = 'data/negative/negative_test' 


# files_test = os.listdir(path_test)

# i = 0
# presence_mask = 0
# sum_masks = 0
# max_masks = []
# images_bug = []
# names_images_bug = []
# for name in files_test :
    
#     image = cv2.imread(path_test+'/'+name)
#     #gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) #optionnelle
#     # Detection
#     faces = classCascade.detectMultiScale(
#         image,
#         scaleFactor=1.1,
#         minNeighbors=5,
#         minSize=(30, 30),
#         flags = cv2.CASCADE_SCALE_IMAGE
#         )
    
#     #Analyse de la performance
#     i+= 1
#     sum_masks = sum_masks + len(faces)
#     if len(faces)>= 1 : 
#         presence_mask += 1
    
#     #On cherche les bugs
#     if len(faces)>= 1 : 
#         max_masks.append(len(faces)) #par curiosité on regarde toutes les images 'mal traitrées'
#         bug = image
#         images_bug.append(bug)
#         names_images_bug.append(name)

# #affichage des images bug
# for name in names_images_bug :      
#     image = cv2.imread(path_test+'/'+name)
#     # Detection
#     faces = classCascade.detectMultiScale(
#         image,
#         scaleFactor=1.1,
#         minNeighbors=5,
#         minSize=(30, 30),
#         flags = cv2.CASCADE_SCALE_IMAGE
#         )
#     #Affichage des rectangles
#     for (x, y, w, h) in faces:
#         cv2.rectangle(image, (x, y), (x+w, y+h), (0, 255, 0), 2)
#     plt.imshow(image)
#     plt.show()
    
# print("Il y a {0} visage(s) masqué(s) détéctés sur {1} images traitées.".format(sum_masks,i))
# print("La présence de masque(s) a été observée sur {0} images parmi les sur {1} images traitées.".format(presence_mask,i))
# print("Le score du classifieur est donc de {0}".format(1 - presence_mask/i))
# print("Le nombre maximum de visages masqués sur une photo est de {0}.".format(max_masks))