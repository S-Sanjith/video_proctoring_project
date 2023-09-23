import cv2
import numpy as np
import math

cap = cv2.VideoCapture("eye_recording.flv")
# cap = cv2.VideoCapture(0)

# cap = cv2.VideoCapture("../sample.webm")
# cap = cv2.VideoCapture("video.mp4")

output_width = 1280
output_height = 720

#capturing the first frame
_, firstframe = cap.read()
firstframe = cv2.resize(firstframe, (output_width, output_height))

#capturing the second frame(or any frame where eye is open for that matter)
cap.set(cv2.CAP_PROP_POS_MSEC,1200)
_, fixedframe = cap.read()
fixedframe = cv2.resize(fixedframe, (output_width, output_height))

#setting the video back to the start
# cap.set(cv2.CAP_PROP_POS_MSEC, 0)

#writing a function that creates contours and draws lines
def drawcontourgrids(roi, fixedcontours):
    rows, cols, _ = roi.shape
    gray_roi = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    gray_roi = cv2.GaussianBlur(gray_roi, (7, 7), 0)

    _, threshold = cv2.threshold(gray_roi, 3, 255, cv2.THRESH_BINARY_INV)
    contours, _ = cv2.findContours(threshold, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    contours = sorted(contours, key=lambda x: cv2.contourArea(x), reverse=True)

    #initializing an array for keeping track of where centers of certain contours lie
    current_centers = []
    #initializaing yet another array for keeping track of where the centroids lie
    fixed_centers = []

    for cnt in contours:
        (x, y, w, h) = cv2.boundingRect(cnt)
        
        #keeping track of center for current boxes
        current_centers.append(((2*x+w)//2, (2*y+h)//2)) 
        
        #cv2.drawContours(roi, [cnt], -1, (0, 0, 255), 3)
        cv2.rectangle(roi, (x, y), (x + w, y + h), (255, 0, 0), 2)
        cv2.line(roi, (x + int(w/2), 0), (x + int(w/2), rows), (0, 255, 0), 2)
        cv2.line(roi, (0, y + int(h/2)), (cols, y + int(h/2)), (0, 255, 0), 2)
        break

    #draw the centroid
    if fixedcontours:
        for cnt in fixedcontours:
            (x, y, w, h) = cv2.boundingRect(cnt)

            #keeping track of center for fixed boxes
            fixed_centers.append(((2*x + w)//2, (2*y+h)//2))

            #cv2.drawContours(roi, [cnt], -1, (0, 0, 255), 3)
            cv2.rectangle(roi, (x, y), (x + w, y + h), (255, 0, 0), 2)
            cv2.line(roi, (x + int(w/2), 0), (x + int(w/2), rows), (0, 255, 0), 2)
            cv2.line(roi, (0, y + int(h/2)), (cols, y + int(h/2)), (0, 255, 0), 2)
            break
        
    #draw a line from centroid to current center
    for i in range(len(fixed_centers)):
        if i < len(current_centers):
            x1, y1 = fixed_centers[i]
            x2, y2 = current_centers[i]

            length = math.sqrt(pow((x2-x1), 2) + pow((y2-y1), 2))

            if length > 70:
                cv2.line(roi, (x1, y1), (x2, y2), (0, 0, 255), 2)

            else:
                cv2.line(roi, (x1, y1), (x2, y2), (255, 255, 0), 2)

    return [threshold, contours]

#draw the centroid using the above function
fixedcontours = []
result = drawcontourgrids(fixedframe, fixedcontours)
threshold = result[0]
fixedcontours = result[1]
cv2.imshow("fixedframe", fixedframe)

#do the same in a video
while True:
    ret, preframe = cap.read()
    frame = cv2.resize(preframe, (output_width, output_height))
    
    if ret is False:
        break

    # roi = frame[100: 895, 237: 1516]
    roi = frame
    # roi = frame[290: 500, 600: 1416]

    #obtaining current contours and threshold from the function above
    results = drawcontourgrids(roi, fixedcontours)
    threshold = results[0]
    contours = results[1]

    cv2.imshow("Threshold", threshold)
    # cv2.imshow("contours", contours)
    cv2.imshow("Roi", roi)
    # cv2.imshow("fixedframe", fixedframe)
    key = cv2.waitKey(30)
    if key == 27:
        break

cv2.destroyAllWindows()