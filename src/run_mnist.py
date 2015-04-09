import gzip
from numpy import zeros, uint8,array,swapaxes
from struct import unpack

from dnn_predict import get_instance
from video_processor import VideoProcessor

def readMNIST(imagefile,labelfile):
	# Open the images with gzip in read binary mode
	images = gzip.open(imagefile, 'rb')
	labels = gzip.open(labelfile, 'rb')
	
	images.read(4)  # skip the magic_number
	numImages = images.read(4)
	numImages = unpack('>I', numImages)[0]
	
	numRows = images.read(4)
	numRows = unpack('>I',numRows)[0]
	
	numCols = images.read(4)
	numCols = unpack('>I',numCols)[0]

	# Get metadata for labels
	labels.read(4)  # skip the magic_number
	N = labels.read(4)
	N = unpack('>I', N)[0]

	if numImages != N:
		raise Exception('The number of labels did not match '
							'the number of images.')
	return (images,labels,numRows,numCols,N)	


class MNISTProcessor(VideoProcessor):
	def __init__(self,imagefile,labelfile,modelFile,labelListFile,exportPath):
		self.imgReader,self.lblReader,self.numRows,self.numCols,self.N = readMNIST(imagefile,labelfile)
		super(MNISTProcessor, self).__init__(modelFile,labelListFile,exportPath,self.numCols,self.numRows,scale=3,fps=10);
		self.frameIdx = 0
	
	def __readNextFrame__(self):
		if(self.frameIdx<self.N):
			print 'Reading MNIST Dataset... {0}%\r'.format((self.frameIdx*100/self.N)),
			tempx=zeros((self.numRows,self.numCols), dtype=float)
			for row in range(self.numRows):
				for col in range(self.numCols):
					tmp_pixel = self.imgReader.read(1)  # Just a single byte
					tmp_pixel = unpack('>B', tmp_pixel)[0]
					tempx[row][col] = tmp_pixel
			tmpLabel = self.lblReader.read(1)
			tmpLabel = unpack('>B', tmpLabel)[0]
			self.frameIdx += 1;
			return (tempx,tmpLabel);
		else:
			return (None,-1)	
		
	def __process_block__(self,frames):
		_frames = [];
		for frame in frames:
			_frame = frame/float(256)
			_frame = swapaxes(_frame,0,1);
			_frame = _frame.reshape((1,self.numRows,self.numCols))
			_frames.extend([_frame])
		_frames = array(_frames);
		return _frames;

if __name__ == '__main__':
	import argparse
	parser = argparse.ArgumentParser(description='Testing MNIST dataset with model')
	parser.add_argument("input",nargs='?',help = "input path of data",default="/home/sudhar/MNIST/data/t10k-images-idx3-ubyte.gz");
	parser.add_argument("label",nargs='?',help = "input path of actual label",default="/home/sudhar/MNIST/data/t10k-labels-idx1-ubyte.gz");
	parser.add_argument("labelmap",nargs='?',help = "label mappings",default="/home/sudhar/MNIST/label.txt");
	parser.add_argument("model",nargs='?',help = "path of model config file",default="/home/sudhar/MNIST/config/model_conf.json");
	parser.add_argument("output",nargs='?',help = "save path",default="test.avi");
	args = parser.parse_args();
	vidProcessor = MNISTProcessor(args.input,args.label,args.model,args.labelmap,args.output);
	vidProcessor.process();
	print 'Processing MNIST dataset [DONE]'
	
