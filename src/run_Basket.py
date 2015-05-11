#!/usr/bin/env python2
import numpy as np
import cv2
import os
import random
from numpy import cumsum

from dnn_predict import get_instance
from video_processor import VideoProcessor


def getMSR2GroundTruth(GroundTruthFile):
	labels = {}
	with open(GroundTruthFile,'r') as f:
		data = f.read();
		for line in data.splitlines():
			if line[0]=='#':
				#comment
				continue;
			seg={};
			words=line.split()
			#video_name, left, width, top, height, start, time duration, action(1-clapping-2-waving-3-boxing)
			seg['action']=int(words[7])
			seg['start']=int(words[5])
			seg['length']=int(words[6])
			video=(words[0].strip('".avi'));
			try:
				labels[video]
			except KeyError:
				labels[video]=list();
			finally:
				labels[video].append(seg);
	return labels;

def getLabels(File,GroundTruthFile,framecount,nolabel=10):
	filename=os.path.basename(File).strip('".avi');
	labels = getMSR2GroundTruth(GroundTruthFile);
	labels = labels[filename]
	lbls = [list() for i in xrange(framecount)];
	for label in labels:
		action = label['action']-1
		for i in xrange(label['start'],(label['length']+label['start'])):
			lbls[i-1].append(action);
	
	lbls=[list([nolabel]) if len(labellist) == 0 else labellist for labellist in lbls]

	return lbls


def getVideoCapture(File):
	cap = cv2.VideoCapture(File)
	cap.set(cv2.cv.CV_CAP_PROP_POS_FRAMES,0)

	framecount = int(cap.get(cv2.cv.CV_CAP_PROP_FRAME_COUNT))
	width = int(cap.get(cv2.cv.CV_CAP_PROP_FRAME_WIDTH))
	height = int(cap.get(cv2.cv.CV_CAP_PROP_FRAME_HEIGHT))
	fps = int(cap.get(cv2.cv.CV_CAP_PROP_FPS));

	print("No of Frames        : " + str(framecount));
	print("Original Width      : " +str(width)); 
	print("Original Height     : " +str(height));
	print("Original FrameRate  : " +str(fps));

	return cap,framecount,width,height,fps;


class BasketBallProcessor(VideoProcessor):
	def __init__(self,file,dataFile,labelfile,modelFile,labelListFile,exportPath,scale):
		cap,framecount,width,height,fps = getVideoCapture(file)
		super(BasketBallProcessor, self).__init__(modelFile, labelListFile, exportPath,
			width, height, scale=scale, fps=fps);
		self.cap = cap
		self.N = framecount
		self.numRows = int(height*scale)
		self.numCols = int(width*scale)
		self.lbls = getLabels(file,labelfile,framecount);
		self.frameIdx = 0
		self.frameInd2 = 0
		self.prevframe = None;
		self.datafilehandle = open(dataFile,'rb');
		header = self.datafilehandle.readline();
	
	def __readNextFrame__(self):
		if(self.frameIdx<self.N):
			ret, frame = self.cap.read()
			if not ret:
				self.frameIdx = self.N;
				return (None,-1);
			print 'Testing ....................... [{0}%]\r'.format((self.frameIdx*100/self.N)),
			self.frameIdx += 1;
			return frame, self.lbls[self.frameIdx];
		else:
			return (None,-1)
			print 'Testing ....................... [DONE]\r',
		
	def __process_block__(self,frames):
		_frames = [];
		for frame in frames:
			if(frame.dtype == np.dtype('uint8')):
				values = self.datafilehandle.readline().split()
				if values.__len__()==0: #No more values available in the data file
					print "error";
				else :
					fvalues=np.asarray([float(value) for value in values]);
					outframe = fvalues.reshape((3,self.numRows,self.numCols))
			else:
				outframe = frame
			#print outframe.shape,(3,self.numRows,self.numCols)
			#outframe = outframe.reshape((3,self.numRows,self.numCols))
			_frames.extend([(outframe/float(255))])
		_frames = np.array(_frames);
		return _frames;

if __name__ == '__main__':
	import argparse
	parser = argparse.ArgumentParser(description='Testing MNIST dataset with model')
	parser.add_argument("Video",help = "Path of video");
	parser.add_argument("dataFile",help = "Path of video",);
	parser.add_argument("groundtruthFile",help = "Path of groundtruthFile");
	parser.add_argument("labelmap",help = "File with label mappings");
	parser.add_argument("model",help = "path of model config file");
	parser.add_argument('-o',"--output",help = "save path", dest="output",default="out.avi");
	parser.add_argument('-s','--scale', action="store", dest="scale", type=float,default=0.5)
	args = parser.parse_args();
	#print args.inputFile,args.groundtruthFile,args.model,args.labelmap,args.output,args.scale
	vidProcessor = BasketBallProcessor(args.Video,args.dataFile,args.groundtruthFile,args.model,args.labelmap,args.output,args.scale);
	vidProcessor.process();
	print 'Testing ....................... [DONE]\n',
	print 'Processing  [DONE]'
	
