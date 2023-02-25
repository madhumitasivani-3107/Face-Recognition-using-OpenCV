import urllib.request
import imutils ,cv2,os
import numpy as np
haar_file = 'haarcascade_frontalface_default.xml"
face_cascade = cv2.CascadeClassifier(haar_file)
datasets = r'C:\Users\Desktop\Codes\Face Detection_Code\database'
print('Face Tracking Started')
(images, labels, names, id) = ([], [], {}, 0)
url = "http:192.168.1.11///shot.jpg"

for (subdirs, dirs, files) in os.walk(datasets):
    for subdir in dirs:
        names[id] = subdir
        subjectpath = os.path.join(datasets, subdir)
        for filename in os.listdir(subjectpath):
            path = subjectpath + '/' + filename
            label = id
            images.append(cv2.imread(path,0))
            labels.append(int(label))
        id +=1

(images, labels) = [np.array(lis) for lis in [images, labels]]
print(labels)
(width, height) = (130,100)

# model = cv2.face.LBPHFaceRecognizer_create()
model = cv2.face.FisherFaceRecognizer_create()

model.train(images,labels)

while True:
    imgPath = urllib.request.urlopen(url)
    imgNp = np.array(bytearray(imgPath.read()),dtype=np.uint8)
    img = cv2.imdecode(imgNp,-1)
    img = imutils.resize(img,450,450)
    cv2.imshow("Camerafeed",img)
    if ord('q') == cv2.waitKey(1):
        exit(0)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)
    flag = False
    for (x,y,w,h) in faces:
        flag = True
        cv2.rectangle(img, (x,y), (x+w, y+h), (255,255,0), 2)
        face = gray[y:y+h, x:x+w]
        face_resize = cv2.resize(face, (width, height))

        prediction = model.predict(face_resize)
        cv2.rectangle(img, (x,y), (x+w, y+h), (0,255,0), 3)
        print(prediction[1])
        if prediction[1] < 800:
            cv2.putText(img, "%s-%.0f"%(names[prediction[0]],prediction[1]),(x-10,y-10),cv2.FONT_HERSHEY_PLAIN,1,(0,255,0),3)
            print(names[prediction[0]])
            cnt = 0 
        else:
            cnt +=1 
            cv2.putText(img, "Unknown", (x-10, y-10), cv2.FONT_HERSHEY_PLAIN,1,(0,0,255),3)
            print("Unknown person")
            if(cnt>100):
                print("Unknown Person")
    if flag is False:
        cv2.putText(img,"No Face Detected",(10,20),cv2.FONT_HERSHEY_SIMPLEX,0.7,(0,0,255),1)
                
    cv2.imshow("FaceRecognition", img)
    key = cv2.waitKey(1) & 0xFF
    if key == ord('q'):
        break
    
cv2.destroyAllWindows()
