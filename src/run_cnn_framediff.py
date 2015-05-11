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

def getLabels(File,GroundTruthFile,framecount,nolabel=3):
	filename=os.path.basename(File).strip('".avi');
	labels = getMSR2GroundTruth(GroundTruthFile);
	labels = labels[filename]
	lbls = [list() for i in xrange(framecount)];
	for label in labels:
		action = label['action']-1
		for i in xrange(label['start'],(label['length']+label['start'])):
			lbls[i].append(action);
	
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


class MSRProcessor(VideoProcessor):
	def __init__(self,file,labelfile,modelFile,labelListFile,exportPath,scale):
		cap,framecount,width,height,fps = getVideoCapture(file)
		super(MSRProcessor, self).__init__(modelFile, labelListFile, exportPath,
			width, height, scale=scale, fps=fps);
		self.cap = cap
		self.N = framecount
		self.numRows = int(height*scale)
		self.numCols = int(width*scale)
		self.lbls = getLabels(file,labelfile,framecount);
		self.frameIdx = 0
		self.prevframe = None;
	
	def __readNextFrame__(self):
		if(self.frameIdx<self.N):
			ret, frame = self.cap.read()
			if not ret:
				self.frameIdx = self.N;
				return (None,-1);
			self.frameIdx += 1;
			return frame, self.lbls[self.frameIdx];
		else:
			return (None,-1)
		
	def __process_block__(self,frames):
		_frames = [];
		i=0;
		for frame in frames:
			if(frame.dtype == np.dtype('uint8')):
				gray = cv2.resize(frame,None, fx=self.scale, fy=self.scale, interpolation = cv2.INTER_LINEAR)
				gray = cv2.cvtColor( gray, cv2.COLOR_RGB2GRAY )
				#gray = cv2.cvtColor( frame, cv2.COLOR_RGB2GRAY )
				#gray = np.uint8(gray)
				edges = cv2.Canny(gray,100,300)

				if self.prevframe is None:
					difframe = edges
				else:
					#print gray.shape, self.prevframe.shape
					difframe = cv2.absdiff(gray,self.prevframe)

				self.prevframe = gray;
				outframe=np.dstack((gray,edges,difframe))

				outframe = np.swapaxes(outframe,0,2);
				outframe = np.swapaxes(outframe,1,2);
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
	parser.add_argument("input",nargs='?',help = "input path of data",default="/home/abil/MSR2-action-data/Videos/1.avi");
	parser.add_argument("label",nargs='?',help = "input path of actual label",default="/home/abil/MSR2-action-data/Videos/groundtruth.txt");
	parser.add_argument("labelmap",nargs='?',help = "label mappings",default="/home/abil/MSR2-action-data/Videos/label.txt");
	parser.add_argument("model",nargs='?',help = "path of model config file",default="/home/abil/MSR2-action-data/final/CNN/model_conf.json");
	parser.add_argument("output",nargs='?',help = "save path",default="test.avi");
	parser.add_argument("scale",nargs='?',help = "scale",default=0.5);
	args = parser.parse_args();
	vidProcessor = MSRProcessor(args.input,args.label,args.model,args.labelmap,args.output,args.scale);
	vidProcessor.process();
	print 'Processing  [DONE]'
	
