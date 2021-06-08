import cv2
import numpy as np

cap = cv2.VideoCapture("traffic.mp4")
fgbg = cv2.createBackgroundSubtractorMOG2()

ret, frame = cap.read()

while cap.isOpened():
    ret, frame = cap.read()
    if ret == True:
        img_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        #cv2.imshow("img_gray", img_gray)
        gblur = cv2.GaussianBlur(img_gray,(5,5),0)
        #cv2.imshow("gblur",gblur)
        fg_mask = fgbg.apply(gblur)
        #cv2.imshow("fg_mask", fg_mask)
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        closing = cv2.morphologyEx(fg_mask, cv2.MORPH_CLOSE, kernel)
        #cv2.imshow("closing", closing)
        opening = cv2.morphologyEx(closing, cv2.MORPH_OPEN, kernel)
        #cv2.imshow("opening", opening)
        dilation = cv2.dilate(opening, kernel, iterations=3)
        #cv2.imshow("dilation", dilation)
        # removes shadows
        retvalbin, bins = cv2.threshold(dilation, 220, 255, cv2.THRESH_BINARY)

        contours, hierarchy = cv2.findContours(bins, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

        minarea = 1000
        maxarea = 50000
        cxx = np.zeros(len(contours))
        cyy = np.zeros(len(contours))

        for i in range(len(contours)):
            # IMPORTANT
            # using hierarchy to only count parent contours (contours not within others)
            if hierarchy[0, i, 3] == -1:
                area = cv2.contourArea(contours[i])
                if minarea < area < maxarea:

                    cnt = contours[i]
                    m = cv2.moments(cnt)
                    cx = int(m["m10"] / m["m00"])
                    cy = int(m["m01"] / m["m00"])

                    x,y,w,h = cv2.boundingRect(contours[i])
                    if area > 10000:
                        cv2.rectangle(frame, (x, y), (x + w, y + h), (100, 46, 200), 2)
                        cv2.putText(frame, "WARNING", (cx + 10, cy + 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (100, 46, 200),2)
                    cv2.rectangle(frame, (x,y), (x+w,y+h), (100,46,200), 2)
                    cv2.putText(frame, str(cx) + "," + str(cy), (cx + 10, cy + 10), cv2.FONT_HERSHEY_SIMPLEX, .3,
                                (100,46,200), 1)
                    cv2.drawMarker(frame, (cx, cy), (255, 255, 255), cv2.MARKER_CROSS, markerSize=4, thickness=3,
                                   line_type=cv2.LINE_8)

        cv2.imshow("Traffic Detector", frame)


        if cv2.waitKey(40) == 27:
            break
    else:
        break

cap.release()
cv2.destroyAllWindows()
