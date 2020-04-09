import face_recognition as fr
import cv2
import os 
import numpy as np 

known_encodings = []
Known_face = []

for dirpath,dname,fname in os.walk("./faces"):
    for f in fname:
        if f.endswith(".jpg") or f.endswith(".png"):
            face = fr.load_image_file("faces/"+f)
            encoding = fr.face_encodings(face)[0]
            known_encodings.append(encoding)
            Known_face.append(f)

print(Known_face)
def merge(list1, list2): 
    merged_list = [(list1[i], list2[i]) for i in range(0, len(list1))] 
    return merged_list 

cap = cv2.VideoCapture(0)

while True:
    _,img = cap.read()
#    img = cv2.imread(image)
    new_face_locs = fr.face_locations(img)
    new_face_encods = fr.face_encodings(img,new_face_locs)

    merged_list = merge(new_face_locs,new_face_encods)

    for loc, encode in merged_list:
        args = []
        match = fr.compare_faces(known_encodings,encode)
        print(match)
        k = [i for i,x in enumerate(match) if x==True]
        print(k)
        if len(k)==1:
            name = Known_face[np.argmax(match)][:-4]
        else:
            name="Unknown"
        cv2.rectangle(img,(loc[3],loc[0]),(loc[1],loc[2]),(0,255,0),1)
        cv2.putText(img,name,(loc[3],loc[0]+10),cv2.FONT_HERSHEY_SIMPLEX,1,(255,0,0))
        
    cv2.imshow("image",img)
    k = cv2.waitKey(5) & 0xFF
    if k==27:
        break
    

cv2.destroyAllWindows()
cap.release()