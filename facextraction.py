


import cv2, dlib, os
from utils import extract_left_eye_center, extract_right_eye_center, get_rotation_matrix, crop_image

detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")


for filename in os.listdir('Test Videos/Original'):

    
    videoPath = filename
    videoName = videoPath[:-4]
    video = cv2.VideoCapture('Test Videos/Original/{0}'.format(videoPath))
    
    
    currentframe = 0
    videolength = int(video.get(cv2.CAP_PROP_FRAME_COUNT))

    
    
    frames = []
    count = 0
    
    while video.isOpened():
        ret, frame = video.read()
        
        if ret:
            frames.append(frame)
            count += 60
            video.set(1, count)
        else:
            video.release()
            break
         
    
        
        
    #NOT MINE  https://github.com/nlhkh/face-alignment-dlib/blob/master/app.py
    
    for j in range(0 , len(frames)):
        img = frames[j]
   
        
        height, width = img.shape[:2]
        
        img = cv2.resize(img, (width, height))
        
        dets = detector(img, 1)
        
        for i, det in enumerate(dets):
            shape = predictor(img, det)
            left_eye = extract_left_eye_center(shape)
            right_eye = extract_right_eye_center(shape)
        
            M = get_rotation_matrix(left_eye, right_eye)
            rotated = cv2.warpAffine(img, M, (width, height), flags=cv2.INTER_CUBIC)
        
            cropped = crop_image(rotated, det)
            cropped = cv2.resize(cropped, (256, 256))

            
            cv2.imwrite('Generated Faces/Normal/normal_{0}_{1}_{2}.jpg'.format(videoName, j, i), cropped)
    
    video.release()

       
            
