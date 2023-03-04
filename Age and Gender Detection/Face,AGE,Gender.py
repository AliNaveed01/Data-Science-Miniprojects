# %%
import cv2
import numpy as np
import face_recognition
import os
from datetime import datetime

# Path of directory containing images for attendance
path = 'ImageAttendance'

# Load images and class names for recognition
images = []
classNames = []
myList = os.listdir(path)
print(myList)
for cl in myList:
    curImg = cv2.imread(f'{path}/{cl}')
    images.append(curImg)
    classNames.append(os.path.splitext(cl)[0])
print(classNames)

# Function to find the face encodings for images
def findEncodings(images):
    encodeList = []
    for img in images:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        encode = face_recognition.face_encodings(img)[0]
        encodeList.append(encode)
    return encodeList


# def markAttendance(name, gender, age_range):
#     with open('A.csv','a') as f:
#         now = datetime.now()
#         # if the name is not in the list then write the name and time in the csv file
#         if name not in f.read():
#             dtString = now.strftime('%H:%M:%S')
#             f.write(f'{name},{dtString},{gender},{age_range}\n')



def markAttendance(name, gender, age):
    with open('A.csv','r+') as f:
        myDataList = f.readlines()
        nameList = []
        for line in myDataList:
            entry = line.split(',')
            nameList.append(entry[0])
        if name not in nameList:
            now = datetime.now()
            dtString = now.strftime('%H:%M:%S')
            f.writelines(f'\n{name},{dtString},{gender},{age}')
                     
# define the gender mean values for the model
MODEL_MEAN_VALUES = (78.4263377603, 87.7689143744, 114.895847746)

# Load the Gender Recognition model
genderProto = "gender_deploy.prototxt"
genderModel = "gender_net.caffemodel"
genderNet = cv2.dnn.readNet(genderModel, genderProto)

#load the age recognition model
age_net = cv2.dnn.readNet('age_deploy.prototxt', 'age_net.caffemodel')


# now make a list of the age labels
ageList = ['(0-2)', '(4-6)', '(8-12)', '(15-20)', '(25-32)', '(38-43)', '(48-53)', '(60-100)']


# imgNaveed = face_recognition.load_image_file('ImageAttendance/beardguy.jpeg')
# imgNaveed = cv2.cvtColor(imgNaveed, cv2.COLOR_BGR2RGB)

# #showing the image
# #plt.imshow(imgNaveed)

# faceloc = face_recognition.face_locations(imgNaveed)[0]
# encodeNaveed = face_recognition.face_encodings(imgNaveed)[0] 
# cv2.rectangle(imgNaveed, (faceloc[3], faceloc[0]), (faceloc[1], faceloc[2]), (255,0,255), 2)

# age = age_detection(imgNaveed) # img[y1:y2, x1:x2] is the face detected


# gender = findGender(faceLoc)


def age_detection(frame):
    # Preprocess the input frame
    blob = cv2.dnn.blobFromImage(frame, 1.0, (227, 227), MODEL_MEAN_VALUES, swapRB=False)

    # Feed the preprocessed image to the age_net
    age_net.setInput(blob)
    age_preds = age_net.forward()

    # Calculate the minimum and maximum age index with probability greater than 0.5
    min_age_index = 0
    max_age_index = len(ageList) - 1
    for i in range(len(ageList)):
        if age_preds[0, i] > 0.5:
            min_age_index = i
            break
    for i in range(len(ageList)-1, -1, -1):
        if age_preds[0, i] > 0.5:
            max_age_index = i
            break

    # Return the age range as a string
    age_range = f'({ageList[min_age_index]}-{ageList[max_age_index]})'

    return age_range


# the function to detect the age of the detected face
# def age_detection(frame):
#     # Preprocess the input frame
#     blob = cv2.dnn.blobFromImage(frame, 1.0, (227, 227), MODEL_MEAN_VALUES, swapRB=False)

#     # Feed the preprocessed image to the age_net
#     age_net.setInput(blob)
#     age_preds = age_net.forward()

#     # Parse the output of age_net to get the predicted age
#     age = int(np.argmax(age_preds))

#     return age

# Define the list of gender labels
genderList = ['Male', 'Female']

# Function to find the gender of the detected face
def findGender(face):
    # Extract the face ROI
    (top, right, bottom, left) = face
    faceImg = img[top:bottom, left:right]
    faceBlob = cv2.dnn.blobFromImage(faceImg, 1.0, (227, 227), MODEL_MEAN_VALUES, swapRB=False)

    # Predict gender of the face
    genderNet.setInput(faceBlob)
    genderPreds = genderNet.forward()
    gender = genderList[genderPreds[0].argmax()]

    return gender

# Find the face encodings for known images
encodeListKnown = findEncodings(images)
print('Encoding Complete')

# Initialize the webcam
cap = cv2.VideoCapture(0)

while True:
    success, img = cap.read()
    imgS = cv2.resize(img,(0,0),None,0.25,0.25)
    imgS = cv2.cvtColor(imgS, cv2.COLOR_BGR2RGB)

    facesCurFrame = face_recognition.face_locations(imgS)
    encodingsCurFrame = face_recognition.face_encodings(imgS, facesCurFrame)

    for encodeFace, faceLoc in zip(encodingsCurFrame, facesCurFrame):
        matches = face_recognition.compare_faces(encodeListKnown, encodeFace)
        faceDis = face_recognition.face_distance(encodeListKnown, encodeFace)
        # print(faceDis)
        matchIndex = np.argmin(faceDis)

        if matches[matchIndex]:
            name = classNames[matchIndex].upper()
            y1, x2, y2, x1 = faceLoc
            y1, x2, y2, x1 = y1*4, x2*4, y2*4, x1*4
            cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.rectangle(img, (x1, y2-35), (x2, y2), (0, 255, 0), cv2.FILLED)
            cv2.putText(img, name, (x1+6, y2-6), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

            # Predict gender of the detected face
            crop_img = imgS[faceLoc[0]:faceLoc[2], faceLoc[3]:faceLoc[1]]
            blob = cv2.dnn.blobFromImage(crop_img, 1, (227, 227), MODEL_MEAN_VALUES, swapRB=False)
            genderNet.setInput(blob)
            genderPreds = genderNet.forward()
            gender = genderList[genderPreds[0].argmax()]
            #print(f"Gender: {gender}")

            # Get the age of the detected face
            age = age_detection(img[y1:y2, x1:x2])

            # Add the predicted age to the output along with the gender
            text = f'{name}, {gender}, {age}'
            cv2.putText(img, text, (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)

            markAttendance(name, gender, age)

    cv2.imshow('Webcam', img)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()



