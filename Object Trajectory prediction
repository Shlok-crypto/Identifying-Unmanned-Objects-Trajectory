import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt
import ROI_image
import math
import VideoSaving as vs

# initialize variables
posX = []
posY = []

# values to check if obj will enter the No Entry ZONE
Y_limit = 590
High_X_lim = 1741
low_X_lim = 1715

# zone ROI values
#0:606, 500:1920
RoiX, RoiY, RoiW, RoiH = 0,0,1920,960

X_PosiblePoints = [item for item in range(RoiX,RoiW)]

# Initialize the Frame
capture = cv.VideoCapture('sample.mov')

# object of background remover
Background_Remover = cv.createBackgroundSubtractorKNN(detectShadows=False)

fourcc = cv.VideoWriter_fourcc('M','J','P','G')

Save_Video = vs.VideoFileSaving("Path_Prediction",Winsize=(1850, 950))

while True:
    # capture the Frame
    _, frame = capture.read()
    print(frame.shape)
    frame = cv.flip(frame,1)
    frame = frame[0:950, 0:1850] # cropping the frame
    #frame = cv.flip(frame, 1)
    frameCp = frame.copy()


    # Apply the Mask onto Frame
    motion = Background_Remover.apply(frame)
    # remove smaller differences
    Smooth_motion = cv.medianBlur(motion,5)
    Smooth_motion = cv.erode(Smooth_motion,None,iterations=3)
    Smooth_motion = cv.dilate(Smooth_motion,None,iterations=3)
    cv.imshow("movemet",Smooth_motion)


    # motion = cv.cvtColor(Smooth_motion, cv.COLOR_GRAY2BGR)
    # motionBit = cv.bitwise_and(frame, motion)

    # Mask the The object
    # Track the Object
    # Find the Contours
    contours, _ = cv.findContours(Smooth_motion[RoiY: RoiY+RoiH, RoiX: RoiX+RoiW], cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)

    for c in contours:
        hull = cv.convexHull(c)
        if 215 < cv.contourArea(hull) < 500:
            print("AREA = ",cv.contourArea(hull))
            # get the x,y,w,h of the contour
            x, y, w, h = cv.boundingRect(hull)
            # Center of Circle
            x1 = w / 2
            y1 = h / 2
            cx = int(x + x1)
            cy = int(y + y1)

            posX.append(cx)
            posY.append(cy)

            # Draw a Circle at the Center the contour
            cv.drawContours(frameCp[RoiY: RoiY+RoiH, RoiX: RoiX+RoiW], [hull], -1, (0, 0, 255), 3)  # DRAW TIGHT FITTING BOUNDARY
            cv.putText(frameCp, "Red: Predicted Trajectory ", (50, 40), cv.FONT_HERSHEY_COMPLEX, 1, (0, 0, 255), thickness=2)
            cv.putText(frameCp, "Green: Actual Path ", (50, 90), cv.FONT_HERSHEY_COMPLEX, 1, (0, 255, 0), thickness=2)

    if posX:  # if PosX is not empty

    # Polynomial Regression  y = Ax^2 + Bx + C
        # find the values of A B C from the give points(center)
        A, B, C = np.polyfit(posX, posY, 2)

        # Plot the location of the ball
        for i, (X, Y) in enumerate(zip(posX, posY)):
            points = (X, Y)
            cv.circle(frameCp, points, 8, (0, 255, 0), cv.FILLED)

        # plot the Estimated path of the Ball
        for X in X_PosiblePoints:
            Y_estimation = int(A * X ** 2 + B * X + C)
            if Y_estimation < 680:
                cv.circle(frameCp, (X,Y_estimation), 2, (0, 0, 255), cv.FILLED) # Drawing the Predicted Path
    # check if it will make it
        a = A
        b = B
        c = C-Y_limit
        try:
            X_value_prediction = int( (-b - math.sqrt(b ** 2 -(4 * a * c) ) ) / (2 * a) )
            if High_X_lim > X_value_prediction > low_X_lim:
                cv.putText(frameCp, "Object enter hoop", (300, 500), cv.FONT_HERSHEY_COMPLEX, 1, (255, 0, 0),
                           thickness=2)
            else:
                cv.putText(frameCp, "Object will miss hoop", (300, 500), cv.FONT_HERSHEY_COMPLEX, 1, (255, 0, 0),
                           thickness=2)
        except:
            print("Value Error")



    Save_Video.videoSaver(frameCp)
    # Convert masks(gray) to BGR (2 Channel to 3 Channel )
    # mask = cv.cvtColor(mask, cv.COLOR_GRAY2BGR)
    # # Extract the object From the Frame
    # RoiFrame = cv.bitwise_and(frame, mask)
    #
    # cv.imshow("ROI", RoiFrame)
    cv.imshow("frame", frameCp)

    key = cv.waitKey(1)

    if key == ord("q"):
        break
    elif key == ord("c"):
        print("clear list")
        posX.clear()
        posY.clear()

    # Select the ROI to detect contours
    elif key == ord("s"):
        temp, _, (RoiX, RoiY, RoiW, RoiH) = ROI_image.RoiCliper(frame)
        print("ROiX, ROiW: ",RoiX,RoiW)

plt.imshow(frame)
plt.show()
capture.release()
cv.destroyAllWindows()
