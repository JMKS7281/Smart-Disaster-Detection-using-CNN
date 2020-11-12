def predict():# USAGE
    # python predict.py 
    # import the necessary packages
    from tensorflow.keras.models import load_model
    from pyimagesearch import config
    from collections import deque
    import numpy as np
    import argparse
    import cv2

    # construct the argument parser and parse the arguments


    # load the trained model from disk
    print("[INFO] loading model and label binarizer...")
    model = load_model(config.MODEL_PATH)

    # initialize the predictions queue
    Q = deque(maxlen=128)

    # initialize the video stream, pointer to output video file, and
    # frame dimensions
    print("[INFO] processing video...")
    vs = cv2.VideoCapture('videos/floods_101_nat_geo.mp4')
    writer = None
    (W, H) = (None, None)
     
    # loop over frames from the video file stream
    while True:
            # read the next frame from the file
            (grabbed, frame) = vs.read()
     
            # if the frame was not grabbed, then we have reached the end
            # of the stream
            if not grabbed:
                    break
     
            # if the frame dimensions are empty, grab them
            if W is None or H is None:
                    (H, W) = frame.shape[:2]

            # clone the output frame, then convert it from BGR to RGB
            # ordering and resize the frame to a fixed 224x224
            output = frame.copy()
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frame = cv2.resize(frame, (224, 224))
            frame = frame.astype("float32")
            
            # make predictions on the frame and then update the predictions
            # queue
            preds = model.predict(np.expand_dims(frame, axis=0))[0]
            Q.append(preds)

            # perform prediction averaging over the current history of
            # previous predictions
            results = np.array(Q).mean(axis=0)
            i = np.argmax(results)
            label = config.CLASSES[i]

            # draw the activity on the output frame
            text = "activity: {}".format(label)
            cv2.putText(output, text, (35, 50), cv2.FONT_HERSHEY_SIMPLEX,
                    1.25, (0, 255, 0), 5)
     
            # check if the video writer is None
            if writer is None:
                    # initialize our video writer
                    fourcc = cv2.VideoWriter_fourcc(*"MJPG")
                    writer = cv2.VideoWriter('output/natural_disasters.avi', fourcc, 30,
                            (W, H), True)
     
            # write the output frame to disk
            writer.write(output)
     
            # check to see if we should display the output frame to our
            # screen
            if 1 > 0:
                    # show the output image
                    cv2.imshow("Output", output)
                    key = cv2.waitKey(1) & 0xFF
             
                    # if the `q` key was pressed, break from the loop
                    if key == ord("q"):
                            break
     
    # release the file pointers
    print("[INFO] cleaning up...")
    writer.release()
    vs.release()
    import cProfile
    cProfile.run(predict.py)
import cProfile
cProfile.run(predict())
