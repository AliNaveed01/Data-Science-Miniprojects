# %%
import numpy as np
import pandas as pd
import cv2
import matplotlib.pyplot as plt
import face_recognition


# %%
imgNaveed = face_recognition.load_image_file('ImageAttendance/Naveed.jpeg')
imgNaveed = cv2.cvtColor(imgNaveed, cv2.COLOR_BGR2RGB)

#showing the image
plt.imshow(imgNaveed)

# %%
#apply the same for the test image
imgTest = face_recognition.load_image_file('ImageAttendance/hanan.jpeg')
imgTest = cv2.cvtColor(imgTest, cv2.COLOR_BGR2RGB)

#showing the image
plt.imshow(imgTest)

# %%
#step 2
# now we will find the faces in the image and encode them
# we will use the face_recognition library for this 
# we will use the face_locations function to find the faces in the image
faceloc = face_recognition.face_locations(imgNaveed)[0]
encodeNaveed = face_recognition.face_encodings(imgNaveed)[0] 
cv2.rectangle(imgNaveed, (faceloc[3], faceloc[0]), (faceloc[1], faceloc[2]), (255,0,255), 2)

#print the faceloc
plt.imshow(imgNaveed)

# %%
#doing the same for the test image
facelocTest = face_recognition.face_locations(imgTest)[0]
encodeTest = face_recognition.face_encodings(imgTest)[0]
cv2.rectangle(imgTest, (facelocTest[3], facelocTest[0]), (facelocTest[1], facelocTest[2]), (255,0,255), 2)

#showing the image
plt.imshow(imgTest)

# %%
#step 3
#comparing the faces and finding the distance between them
# we will use the compare_faces function to compare the faces
checkDist = face_recognition.compare_faces([encodeNaveed], encodeTest)
faceDis = face_recognition.face_distance([encodeNaveed], encodeTest)
print(checkDist, faceDis)
#lower the distance, more the similarity

# %%
#final step
#displaying the result
cv2.putText(imgTest, f'{checkDist} {round(faceDis[0],2)}', (50,50), cv2.FONT_HERSHEY_COMPLEX, 1, (0,0,255), 2)
plt.imshow(imgTest)

# %%



