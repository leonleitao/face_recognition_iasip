import face_recognition
import cv2
import glob
import os
import numpy as np

def encode_faces():
    '''Function to encode the faces of the 5 characters'''
    encoded_faces={}
    for file in glob.glob("iasip_characters/*.jpg"):
        name=os.path.basename(file).split('.')[0]
        face=face_recognition.load_image_file(file)
        encoded_faces[name]=face_recognition.face_encodings(face)[0]
    return encoded_faces

def label_faces(vid,show=True,scale=0.5,output=False,output_name=None):
    '''Function that takes an input clip and labels the character faces'''
    # Find the number of frames in the video
    length = int(vid.get(cv2.CAP_PROP_FRAME_COUNT))
    # Check if the video is valid
    if (vid.isOpened()== False): 
        print("Error opening video stream or file")
        return
    
    #Track the frame number
    i=1
    while(vid.isOpened()):
        ret, frame = vid.read()
        if ret == True:
            # Scale the frame for faster face recognition processing
            small_frame = cv2.resize(frame, (0, 0), fx=scale, fy=scale)
            # Convert the image from BGR color (which OpenCV uses) to RGB color (which face_recognition uses)
            rgb_frame=small_frame[:,:,::-1]
            face_locs=face_recognition.face_locations(rgb_frame,model='hog')
            face_encs=face_recognition.face_encodings(rgb_frame,face_locs)
            
            if output & (i==1):
                height, width, channels = frame.shape
                out = cv2.VideoWriter(f'videos/{output_name}', -1, 30, (width, height))

            face_names=[]
            for enc in face_encs:
                matches=face_recognition.compare_faces(ias_faces,enc,tolerance=0.6)
                face_distances = face_recognition.face_distance(ias_faces, enc)
                best_match_index = np.argmin(face_distances)
                name="Unknown"
                if matches[best_match_index]:
                    name = ias_names[best_match_index]
                face_names.append(name)
            for (t,r,b,l),name in zip(face_locs,face_names):
                # Scale back the face locations
                t *= int(1/scale)
                r *= int(1/scale)
                b *= int(1/scale)
                l *= int(1/scale)
                # Draw and label a rectangular bo for each detected face
                cv2.rectangle(img=frame,pt1=(l,t),pt2=(r,b),color=(255,0,0),thickness=2)
                cv2.putText(img=frame,text=name,org=(l+30,b+30),fontFace=cv2.FONT_HERSHEY_COMPLEX,fontScale=0.6,color=(255,255,255))
            if output:
                out.write(frame)
            if show:
                cv2.imshow('Output',frame)
                if cv2.waitKey(30) & 0xFF == ord('q'):
                    break
            print(f'Frame {i}/{length} done.')
            i=i+1

        # Break the loop after last frame
        else: 
            break
    if output:
        out.release()
    vid.release()
    cv2.destroyAllWindows()
    return

if __name__=='__main__':
    encoded_faces=encode_faces()
    ias_names=list(encoded_faces.keys())
    ias_faces=list(encoded_faces.values())
    vid= cv2.VideoCapture('videos/video_1.mp4')
    label_faces(vid,show=False,output=True,output_name='output_1.mp4')