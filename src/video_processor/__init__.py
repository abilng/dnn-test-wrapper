import cv2,numpy as np
from multiprocessing.pool import ThreadPool
from collections import deque
from time import sleep
from io_func.video_writer import VideoWriter;
from dnn_predict import get_instance
from io_func.block_reader import BlockReader

def load_labels(fileName):
	idx = 0; labels ={}
	with open(fileName,'r') as f:
		for line in f:
			labels[idx] = line[:-1];
			idx += 1;
		labels[-1] = "None"
	#print labels
	return labels;
	
def __draw_str__(dst, (x, y), s, fontsize = 0.6, color=[255, 255, 255]):
	cv2.putText(dst, s, (x, y), cv2.FONT_HERSHEY_PLAIN, fontsize, tuple(color), lineType=cv2.CV_AA)

class VideoTask(object):	
	def __init__(self,frameBlock,scores,labels):
		self.frameBlock = frameBlock;
		self.scores = scores;
		self.labels = labels;
	
	def process(self,vidwriter,designFrameBaner):
		shape = vidwriter.shape
		for idx in range(len(self.frameBlock)):
			frame = designFrameBaner(self.frameBlock[idx],self.scores[idx],self.labels[idx]);
			vidwriter.write(np.uint8(frame));
	
class VideoProcessor(object):
	def __init__(self,modelFile,labelListFile,fileName,width,height,scale=1,banerWidth=50,fps=20):
		self.width,self.height,self.banerWidth = scale*width,scale*height,banerWidth
		self.scale = scale;
		self.predictor = get_instance(modelFile);
		self.vidWriter = VideoWriter(fileName,self.banerWidth+self.width,self.height,fps = fps);
		self.vidWriter.build();
		self.isFinished = False;
		self.labels = load_labels(labelListFile);
		self.tasks = deque();
		n_outs = self.predictor.model_config['n_outs'];
		self.colors = np.random.randint(256, size=(n_outs, 3))
	
	def __design_frame_banner__(self,frame,_score,_labels,top=3):
		if not self.scale == 1:
			frame = cv2.resize(frame,None,fx=self.scale, fy=self.scale, interpolation = cv2.INTER_CUBIC)
		if frame.ndim == 2:
			frame = np.dstack((frame,frame,frame));
		assert(frame.ndim==3),"given frame not in shape"
		baner_frame = np.zeros((self.height,self.banerWidth,3));
		_indices = np.argsort(_score)[::-1];
		col = 5; row = 10; steps = int(((self.height-30)/(top+1))-5);
		for classLbl in _indices[:top]:
			#print classLbl,_labels
			_str = "{0}".format(self.labels[classLbl]);
			__draw_str__(baner_frame,(col,row),_str,color=self.colors[classLbl]); row += steps
		#if not _labels is None:
		#	_str = ">{0}<".format(self.labels[_label]);
		#	__draw_str__(baner_frame,(col,row),_str,fontsize=0.5,color=self.colors[_label]);
		
		if _indices[0] not in _labels:
			cv2.circle(baner_frame,(int(self.banerWidth/2),int(self.height-10)),8,(0,0,255),-1);
		else:	
			cv2.circle(baner_frame,(int(self.banerWidth/2),int(self.height-10)),8,(0,255,0),-1);
			
		#print baner_frame.shape,frame.shape
		return np.hstack((baner_frame,frame));
	
	def __videoWriter__(self):		
		while (len(self.tasks) > 0) or (not self.isFinished):
			if (len(self.tasks) > 0):
				task = self.tasks.popleft()
				task.process(self.vidWriter,self.__design_frame_banner__);
		self.vidWriter.close();
		
	def process(self):
		blockReader = BlockReader(self.predictor.batch_size,self.predictor.input_shape,self.__readNextFrame__);
		pool = ThreadPool(processes = 2);
		task = pool.apply_async(self.__videoWriter__);
		p_frames_cnt = 0
		while not blockReader.isFinished:
			(frameCnt,frames,labels)=blockReader.readNextBlock();
			if frameCnt > 0:
				processed_frames = self.__process_block__(frames);
				scores = self.predictor.get_score(processed_frames);

				vidTask = VideoTask(frames[:frameCnt],scores[:frameCnt],labels[:frameCnt])
				#vidTask.process(self.vidWriter,self.__design_frame_banner__)
				self.tasks.append(vidTask);
		self.isFinished = True;	
		while not task.ready():
			sleep(1);
			
	def __process_block__(self):
		raise NotImplementedError;
		
	def __readNextFrame__(self,frames):
		raise NotImplementedError;
		

