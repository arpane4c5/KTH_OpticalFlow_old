import cv2
import numpy as np
import os
#import sys


def draw_flow(img, flow, step=16):
    h, w = img.shape[:2]
    y, x = np.mgrid[step/2:h:step, step/2:w:step].reshape(2,-1)
    fx, fy = flow[y,x].T
    lines = np.vstack([x, y, x+fx, y+fy]).T.reshape(-1, 2, 2)
    lines = np.int32(lines + 0.5)
    vis = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    cv2.polylines(vis, lines, 0, (0, 255, 0))
    for (x1, y1), (x2, y2) in lines:
        cv2.circle(vis, (x1, y1), 1, (0, 255, 0), -1)
    return vis

# function to iterate over the training frames and generate optical flow for the same
def generateOneOF(trainPath, kthContents_train, destFolder):
	import csv
	labelsFilename = os.path.join(os.path.join(os.getcwd(), destFolder), "labels.csv")
	print(labelsFilename)
	# decide the destination folder
	if not os.path.exists(os.path.dirname(labelsFilename)):
		os.makedirs(os.path.dirname(labelsFilename))
	
	w = csv.writer(open(labelsFilename, "w"))
    
	#w = csv.writer(open("labels_train.csv", "w"))
	actionNo = 0
	trainDict = {}
	for action in kthContents_train:
		actionNo = actionNo + 1
		print(' Training Action : '+action)
		trainPathAction = os.path.join(trainPath, action)

		# iterating over the individual examples in the training set
		for example in sorted(os.listdir(trainPathAction)):
			print("Example : "+example+" : "+str(actionNo))
			trainPathActionEg = os.path.join(trainPathAction, example)

			prevFrame = None
			trainDict[action+example] = actionNo 	# value is the action no 1-6 	
			flowFile = file(os.path.join(os.path.join(os.getcwd(), destFolder),
				destFolder+action+example+'.bin'), 'wb')
			# iterate over the frames of the video and calculate optical flow
			for frameName in sorted(os.listdir(trainPathActionEg)):
				nextFrame = cv2.imread(os.path.join(trainPathActionEg, frameName), 
					cv2.IMREAD_GRAYSCALE) 
				print("Frame : "+ frameName + " ---> Shape : "+ str(nextFrame.shape))
				if prevFrame is None:
					prevFrame = nextFrame
					continue
				#print(nextFrame)
				#cv2.imshow(action , nextFrame)
				#cv2.waitKey(100)
				#hsv = np.zeros_like(nextFrame)
				#hsv[...,1] = 255
				# Calculate Farneback Optical Flow
				flow = cv2.calcOpticalFlowFarneback(prevFrame,nextFrame, 
					None, 0.5, 3, 15, 3, 5, 1.2, 0)
				np.save(flowFile, flow)		# no of flow vectors saved = no of frames - 1
				#print("Flow : " + str(type(flow)))
				#print('Shape : '+ str(flow.shape))
				#print(flow)	# flow has dimensions 120 X 160 X 2 (2 for magnitude and angle)
				#print('Mag : ' +str(np.amin(flow[...,0]))+' : '+str(np.amax(flow[...,0]))+' : '
				#	+ str(np.mean(flow[...,0])))
				#print('Ang : ' + str(np.amin(flow[...,1]))+' : '+str(np.amax(flow[...,1]))+ ' : '
				#	+ str(np.mean(flow[...,1])))
				#mag, ang = cv2.cartToPolar(flow[...,0], flow[...,1])
				#hsv[...,0] = ang*180/np.pi/2
				#hsv[...,2] = cv2.normalize(mag,None,0,255,cv2.NORM_MINMAX)
				#bgr = cv2.cvtColor(hsv,cv2.COLOR_HSV2BGR)
				#cv2.imshow(action+" Flow "+frameName, draw_flow(nextFrame, flow))
				#cv2.waitKey(0)
				
			flowFile.close()

	for key, val in trainDict.items():
		w.writerow([key, val])
	return


if __name__ == '__main__':

	#dir = sys.argv[1]

	# get the contents of the directory
	trainPath = 'kth/train'
	valPath = 'kth/val'
	testPath = 'kth/test'
	trainvalPath = 'kth/trainval'
	kthContents_train = sorted(os.listdir(trainPath))
	kthContents_val = sorted(os.listdir(valPath))
	kthContents_test = sorted(os.listdir(testPath))
	kthContents_trainval = sorted(os.listdir(trainvalPath))

	print(kthContents_train)
	print(kthContents_test)
	# get the optical flow in training set
	# read the frame in videos one by one

	generateOneOF(trainPath, kthContents_train, 'kth_train')
	#generateOneOF(valPath, kthContents_val, 'kth_val')
	#generateOneOF(testPath, kthContents_test, 'kth_test')
	#generateOneOF(trainvalPath, kthContents_trainval, 'kth_trainval')
	cv2.destroyAllWindows()
	print('Generation Done !!')
