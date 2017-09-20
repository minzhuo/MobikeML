from datetime import date, datetime
import csv
import tensorflow as tf 

filePath = "data.csv"

Geo = { '0':0,'1':1,'2':2,'3':3,'4':4,'5':5,'6':6,'7':7,'8':8,'9':9,'b':10,
		'c':11,'d':12,'e':13,'f':14,'g':15,'h':16,'j':17,'k':18,'m':19,'n':20,
		'p':21,''q:22,'r':23,'s':24,'t':25,'u':26,'v':27,'w':28,'x':29,'y':30,'z':31}

class FNN:
	def __init__(self):
		self.xSize = 8
		self.ySize = 4
		self.layer1Size = 20
		self.layer2Size = 20
		# self.sess = tf.Session()
		# self.sess.run(tf.global_variables_initializer())
	def initial_weight(shape, regularizer, w_initializer):
		weights = tf.get_variable("weights", shape, initializer = w_initializer)
		if regularizer != None:
			tf.add_to_collection('losses', regularizer(weights))
		return weights

	def buildFNN(self):
		w_initial = tf.random_normal_initializer(0, 0.1)
		b_initail = tf.constant_initializer(0.1) 
		regularizer = tf.contrib.layers.l2_regularizer(REGULARAZTION_RATE)

		self.x = tf.placeholder(tf.float32,  [None, self.xSize], name = "x-input")
		self.y = tf.placeholder(tf.float32, [None, self.ySize], name = "y-output")

		with tf.variable_scope('layer1'):
			w1 = initial_weight([xSize, layer1Size], regularizer, w_initial)
			b1 = tf.get_variable("biases", [layer1Size], initializer = b_initail);
			l1 = tf.nn.relu(tf.matmul(x, w1) + b1)

		with tf.variable_scope('layer2'):
			w2 = initial_weight([layer1Size, layer2Size], regularizer, w_initial)
			b2 = tf.get_variable("biases", [layer2Size], initializer = b_initail)
			l2 = tf.nn.relu(tf.matmul(l1, w2) + b2)

		with tf.variable_scope('outputlayer'):
			w3 = initial_weight([layer2Size, ySize], regularizer, w_initial)
			b3 = tf.get_variable("biases", [ySize], initializer = b_initail)
			self.y_ = tf.matmul(l2, w3) + b3

	def train():
		global_step = tf.Variable(0, trainable = False)
		variable_averages = tf.train.ExponentialMovingAverage(MOVING_AVERGAGE_DECAY, global_step)
		variable_averages_op = variable_averages.apply(tf.trainable_variable())
		cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(logits = y, labels = tf.argmax(y, 1))
		cross_entropy_mean = tf.reduce_mean(cross_entropy)
		self.loss = cross_entropy_mean + tf.add(tf.get_collection('losses'))
		

class dataProcess:
	def __init__(self, userID, bikeType, startTime, startLoc, endLoc = '0000000'):
		self.userID = userID
		self.bikeType = bikeType
		self.startTime = startTime
		self.startLoc = startLoc
		self.endLoc = endLoc

	def geohashProcess(self, location):
		digitLocation = []
		location = location[3:]
		for l in location:
			digitLocation.append(Geo[l])
		return digitLocation

	def timeProcess(self):
		d = datetime.strptime(self.time, '%Y-%m-%d %H:%M')
		return d.weekday(), d.hour

	def getTrainingData(self):
		return self.userID, self.bikeType, timeProcess(), geohashProcess(startLoc), geohashProcess(endLoc)

	def getTestData(self):
		return self.userID, self.bikeType, timeProcess(), geohashProcess(startLoc)


def readFile():
	pass
      
