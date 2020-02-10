


import cv2, dlib
from utils import extract_left_eye_center, extract_right_eye_center, get_rotation_matrix, crop_image

#videoName = input("Enter in the filename...")
video = cv2.VideoCapture("01__exit_phone_room.mp4")
currentframe = 0
videolength = int(video.get(cv2.CAP_PROP_FRAME_COUNT))



frames = []
count = 0

while video.isOpened():
    ret, frame = video.read()
    
    if ret:
 
        
        frames.append(frame)
        count += 30 
        video.set(1, count)
    else:
        video.release()
        break
     

    
    
#NOT MINE  https://github.com/nlhkh/face-alignment-dlib/blob/master/app.py

input_image = frames[7]



detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")
img = cv2.imread(input_image, cv2.IMREAD_GRAYSCALE)
height, width = img.shape[:2]
s_height, s_width = height // 1, width // 1
img = cv2.resize(img, (s_width, s_height))

dets = detector(img, 1)

for i, det in enumerate(dets):
    shape = predictor(img, det)
    left_eye = extract_left_eye_center(shape)
    right_eye = extract_right_eye_center(shape)

    M = get_rotation_matrix(left_eye, right_eye)
    rotated = cv2.warpAffine(img, M, (s_width, s_height), flags=cv2.INTER_CUBIC)

    cropped = crop_image(rotated, det)

    output_image_path = output_image.replace('.jpg', '_%i.jpg' % i)
    
    cv2.imwrite(output_image_path, cropped)    
