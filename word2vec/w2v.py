'''
This file is to study how word2vec works.

at first, try to construct the skipgram word2vec with tensorflow, 
which basic on the tensorflow training example. 

but using gensim word2vec directly at the end (for speed concern)

@by Johnson Wu
'''

import numpy as np
import tensorflow as tf
from tensorflow.contrib.tensorboard.plugins import projector



#=====================================
#- model parameters
#=====================================
flags = tf.app.flags

flags.DEFINE_boolean("Train_Model", False, "training model or using model")
# flags.DEFINE_string("save_path", '.\data', "saving data path")
# flags.DEFINE_string("train_data", '.\text8', "Training text file.")
# flags.DEFINE_string("eval_data", '.\questions-words.txt', "File consisting of analogies of four tokens."
# 					"embedding 2 - embedding 1 + embedding 3 should be close "
# 					"to embedding 4."
# 					"E.g. https://word2vec.googlecode.com/svn/trunk/questions-words.txt.")
# flags.DEFINE_integer("embedding_size", 200, "The embedding dimension size.")
# flags.DEFINE_integer("epochs_to_train", 15, "Number of epochs to train. Each epoch processes the training data once completely.")
# flags.DEFINE_float("learning_rate", 0.2, "Initial learning rate.")
# flags.DEFINE_integer("num_neg_samples", 100, "Negative samples per training example.")
# flags.DEFINE_integer("batch_size", 16, "(size of a minibatch).")
# flags.DEFINE_integer("concurrent_steps", 12, "The number of concurrent training steps.")
# flags.DEFINE_integer("window_size", 5, "The number of target words to the left and rightf")
# flags.DEFINE_integer("min_count", 5, "The minimum number of word occurrences for it to be, included in the vocabulary.")
# flags.DEFINE_float("subsample", 1e-3,
# 					"Subsample threshold for word occurrence. Words that appear "
# 					"with higher frequency will be randomly down-sampled. Set "
# 					"to 0 to disable.")
# flags.DEFINE_boolean("interactive", False,
# 					"If true, enters an IPython interactive session to play with the trained "
# 					"model. E.g., try model.analogy('france', 'paris', 'russia') and "
# 					"model.nearby(['proton', 'elephant', 'maxwell']")
# flags.DEFINE_integer("statistics_interval", 5, "Print statistics every n seconds.")
# flags.DEFINE_integer("summary_interval", 5,
# 					"Save training summary to file every n seconds (rounded "
# 					"up to statistics interval.")
# flags.DEFINE_integer("checkpoint_interval", 600,
# 					"Checkpoint the model (i.e. save the parameters) every n "
# 					"seconds (rounded up to statistics interval.")

FLAGS = flags.FLAGS

# ''' 
# use Options class to refer the flags conviniently
# which is model parameters
# '''
# class Options(object):
# 	def __init__(self):
# 		# Model options.
# 		# Embedding dimension. [200, int32]
# 		self.emb_dim = FLAGS.embedding_size
		
# 		# Training options.
# 		# The training text file. ['.\text8', string]
# 		self.train_data = FLAGS.train_data
		
# 		# Number of negative samples per example. [100, int32]
# 		self.num_samples = FLAGS.num_neg_samples
		
# 		# The initial learning rate. [0.2, float32]
# 		self.learning_rate = FLAGS.learning_rate
		
# 		# Number of epochs to train. After these many epochs, the learning
# 		# rate decays linearly to zero and the training stops. [15, int32]
# 		self.epochs_to_train = FLAGS.epochs_to_train
		
# 		# Concurrent training steps. [12, int32]
# 		self.concurrent_steps = FLAGS.concurrent_steps
		
# 		# Number of examples for one training step. [16, int32]
# 		self.batch_size = FLAGS.batch_size
		
# 		# The number of words to predict to the left and right of the target word.
# 		self.window_size = FLAGS.window_size		# [5, int32]
		
# 		# The minimum number of word occurrences for it to be included in the
# 		# vocabulary. low freq. word [5, int32]
# 		self.min_count = FLAGS.min_count
		
# 		# Subsampling threshold for word occurrence.  [1e-3, float32]
# 		self.subsample = FLAGS.subsample
		
# 		# How often to print statistics. [5, int32]
# 		self.statistics_interval = FLAGS.statistics_interval
		
# 		# How often to write to the summary file (rounds up to the nearest
# 		# statistics_interval).	[5, int32]
# 		self.summary_interval = FLAGS.summary_interval
		
# 		# How often to write checkpoints (rounds up to the nearest statistics
# 		# interval).	[600, int32]
# 		self.checkpoint_interval = FLAGS.checkpoint_interval
		
# 		# Where to write out summaries.	['.\data', string]
# 		self.save_path = FLAGS.save_path
		
# 		# Eval options.
# 		# The text file for eval.	['.\questsion_word.txt', string]
# 		self.eval_data = FLAGS.eval_data

# class Word2Vec(object):
# 	def __init__(self, options, session):
# 		self._options = options
# 		self._session = session
# 		self._word2id = {}
# 		self._id2word = []
# 		self.build_graph()
# 		self.build_eval_graph()
# 		self.save_vocab()
# 		self._read_analogies()

# 	def _read_analogies(self):
# 		"""Reads through the analogy question file.
# 		Returns:
# 		questions: a [n, 4] numpy array containing the analogy question's word ids.
# 		questions_skipped: questions skipped due to unknown words.
# 		"""
# 		questions = []
# 		questions_skipped = 0
# 		with open(self._options.eval_data, "rb") as analogy_f:
# 			for line in analogy_f:
# 				if line.startswith(b":"):  # Skip comments.
# 					continue
# 				words = line.strip().lower().split(b" ")
# 				ids = [self._word2id.get(w.strip()) for w in words]
# 				if None in ids or len(ids) != 4:
# 					questions_skipped += 1
# 				else:
# 					questions.append(np.array(ids))
# 		print("Eval analogy file: ", self._options.eval_data)
# 		print("Questions: ", len(questions))
# 		print("Skipped: ", questions_skipped)
# 		self._analogy_questions = np.array(questions, dtype=np.int32)

# 	def forward(self, examples, labels):
# 		"""Build the graph for the forward pass."""
# 		opts = self._options
		
# 		# Declare all variables we need.
# 		# Embedding: [vocab_size, emb_dim]
# 		init_width = 0.5 / opts.emb_dim
# 		emb = tf.Variable(tf.random_uniform([opts.vocab_size, opts.emb_dim], -init_width, init_width), name="emb")
# 		self._emb = emb
		
# 		# Softmax weight: [vocab_size, emb_dim]. Transposed.
# 		sm_w_t = tf.Variable(tf.zeros([opts.vocab_size, opts.emb_dim]), name="sm_w_t")
		
# 		# Softmax bias: [emb_dim].
# 		sm_b = tf.Variable(tf.zeros([opts.vocab_size]), name="sm_b")
		
# 		# Global step: scalar, i.e., shape [].
# 		self.global_step = tf.Variable(0, name="global_step")
		
# 		# Nodes to compute the nce loss w/ candidate sampling.
# 		labels_matrix = tf.reshape(tf.cast(labels, dtype=tf.int64), [opts.batch_size, 1])
		
# 		# Negative sampling.
# 		sampled_ids, _, _ = (tf.nn.fixed_unigram_candidate_sampler(
# 							true_classes=labels_matrix,
# 							num_true=1,
# 							num_sampled=opts.num_samples,
# 							unique=True,
# 							range_max=opts.vocab_size,
# 							distortion=0.75,
# 							unigrams=opts.vocab_counts.tolist()))
		
# 		# Embeddings for examples: [batch_size, emb_dim]
# 		example_emb = tf.nn.embedding_lookup(emb, examples)
		
# 		# Weights for labels: [batch_size, emb_dim]
# 		true_w = tf.nn.embedding_lookup(sm_w_t, labels)
# 		# Biases for labels: [batch_size, 1]
# 		true_b = tf.nn.embedding_lookup(sm_b, labels)
		
# 		# Weights for sampled ids: [num_sampled, emb_dim]
# 		sampled_w = tf.nn.embedding_lookup(sm_w_t, sampled_ids)
# 		# Biases for sampled ids: [num_sampled, 1]
# 		sampled_b = tf.nn.embedding_lookup(sm_b, sampled_ids)
		
# 		# True logits: [batch_size, 1]
# 		true_logits = tf.reduce_sum(tf.mul(example_emb, true_w), 1) + true_b
		
# 		# Sampled logits: [batch_size, num_sampled]
# 		# We replicate sampled noise lables for all examples in the batch
# 		# using the matmul.
# 		sampled_b_vec = tf.reshape(sampled_b, [opts.num_samples])
# 		sampled_logits = tf.matmul(example_emb, sampled_w, transpose_b=True) + sampled_b_vec
# 		return true_logits, sampled_logits

# 	def nce_loss(self, true_logits, sampled_logits):
# 		"""Build the graph for the NCE loss."""

# 		# cross-entropy(logits, labels)
# 		opts = self._options
# 		true_xent = tf.nn.sigmoid_cross_entropy_with_logits(true_logits, tf.ones_like(true_logits))
# 		sampled_xent = tf.nn.sigmoid_cross_entropy_with_logits(sampled_logits, tf.zeros_like(sampled_logits))

# 		# NCE-loss is the sum of the true and noise (sampled words)
# 		# contributions, averaged over the batch.
# 		nce_loss_tensor = (tf.reduce_sum(true_xent) + tf.reduce_sum(sampled_xent)) / opts.batch_size
# 		return nce_loss_tensor

# 	def optimize(self, loss):
# 		"""Build the graph to optimize the loss function."""

# 		# Optimizer nodes.
# 		# Linear learning rate decay.
# 		opts = self._options
# 		words_to_train = float(opts.words_per_epoch * opts.epochs_to_train)
# 		lr = opts.learning_rate * tf.maximum(0.0001, 1.0 - tf.cast(self._words, tf.float32) / words_to_train)
# 		self._lr = lr
# 		optimizer = tf.train.GradientDescentOptimizer(lr)
# 		train = optimizer.minimize(loss,
# 															 global_step=self.global_step,
# 															 gate_gradients=optimizer.GATE_NONE)
# 		self._train = train

# 	def build_eval_graph(self):
# 		"""Build the eval graph."""
# 		# Eval graph

# 		# Each analogy task is to predict the 4th word (d) given three
# 		# words: a, b, c.  E.g., a=italy, b=rome, c=france, we should
# 		# predict d=paris.

# 		# The eval feeds three vectors of word ids for a, b, c, each of
# 		# which is of size N, where N is the number of analogies we want to
# 		# evaluate in one batch.
# 		analogy_a = tf.placeholder(dtype=tf.int32)  # [N]
# 		analogy_b = tf.placeholder(dtype=tf.int32)  # [N]
# 		analogy_c = tf.placeholder(dtype=tf.int32)  # [N]

# 		# Normalized word embeddings of shape [vocab_size, emb_dim].
# 		nemb = tf.nn.l2_normalize(self._emb, 1)

# 		# Each row of a_emb, b_emb, c_emb is a word's embedding vector.
# 		# They all have the shape [N, emb_dim]
# 		a_emb = tf.gather(nemb, analogy_a)  # a's embs
# 		b_emb = tf.gather(nemb, analogy_b)  # b's embs
# 		c_emb = tf.gather(nemb, analogy_c)  # c's embs

# 		# We expect that d's embedding vectors on the unit hyper-sphere is
# 		# near: c_emb + (b_emb - a_emb), which has the shape [N, emb_dim].
# 		target = c_emb + (b_emb - a_emb)

# 		# Compute cosine distance between each pair of target and vocab.
# 		# dist has shape [N, vocab_size].
# 		dist = tf.matmul(target, nemb, transpose_b=True)

# 		# For each question (row in dist), find the top 4 words.
# 		_, pred_idx = tf.nn.top_k(dist, 4)

# 		# Nodes for computing neighbors for a given word according to
# 		# their cosine distance.
# 		nearby_word = tf.placeholder(dtype=tf.int32)  # word id
# 		nearby_emb = tf.gather(nemb, nearby_word)
# 		nearby_dist = tf.matmul(nearby_emb, nemb, transpose_b=True)
# 		nearby_val, nearby_idx = tf.nn.top_k(nearby_dist,
# 																				 min(1000, self._options.vocab_size))

# 		# Nodes in the construct graph which are used by training and
# 		# evaluation to run/feed/fetch.
# 		self._analogy_a = analogy_a
# 		self._analogy_b = analogy_b
# 		self._analogy_c = analogy_c
# 		self._analogy_pred_idx = pred_idx
# 		self._nearby_word = nearby_word
# 		self._nearby_val = nearby_val
# 		self._nearby_idx = nearby_idx



# 	def skipgram(self):
# 		filename=self._options.train_data,
# 		batch_size=self._options.batch_size,
# 		window_size=self._options.window_size,
# 		min_count=self._options.min_count,
# 		subsample=self._options.subsample
# 		pass
# 		# return words, counts, words_per_epoch, epoch, words, examples, labels


# 	def build_graph(self):
# 		opts = self._options
# 		# The training data. A text file.
# 		(words, counts, words_per_epoch, self._epoch, self._words, examples,
# 		 labels) = self.skipgram(filename=opts.train_data,
# 																 batch_size=opts.batch_size,
# 																 window_size=opts.window_size,
# 																 min_count=opts.min_count,
# 																 subsample=opts.subsample)
# 		(opts.vocab_words, opts.vocab_counts, opts.words_per_epoch) = self._session.run([words, counts, words_per_epoch])
		
# 		opts.vocab_size = len(opts.vocab_words)
# 		print("Data file: ", opts.train_data)
# 		print("Vocab size: ", opts.vocab_size - 1, " + UNK")
# 		print("Words per epoch: ", opts.words_per_epoch)
# 		self._examples = examples
# 		self._labels = labels
# 		self._id2word = opts.vocab_words
# 		for i, w in enumerate(self._id2word):
# 			self._word2id[w] = i

# 		true_logits, sampled_logits = self.forward(examples, labels)
# 		loss = self.nce_loss(true_logits, sampled_logits)
# 		tf.scalar_summary("NCE loss", loss)
# 		self._loss = loss
# 		self.optimize(loss)

# 		# Properly initialize all variables.
# 		tf.initialize_all_variables().run()

# 		self.saver = tf.train.Saver()

# 	def save_vocab(self):
# 		"""Save the vocabulary to a file so the model can be reloaded."""
# 		opts = self._options
# 		with open(os.path.join(opts.save_path, "vocab.txt"), "w") as f:
# 			for i in xrange(opts.vocab_size):
# 				f.write("%s %d\n" % (tf.compat.as_text(opts.vocab_words[i]),
# 														 opts.vocab_counts[i]))

# 	def _train_thread_body(self):
# 		initial_epoch, = self._session.run([self._epoch])
# 		while True:
# 			_, epoch = self._session.run([self._train, self._epoch])
# 			if epoch != initial_epoch:
# 				break

# 	def train(self):
# 		"""Train the model."""
# 		opts = self._options

# 		initial_epoch, initial_words = self._session.run([self._epoch, self._words])

# 		summary_op = tf.merge_all_summaries()
# 		summary_writer = tf.train.SummaryWriter(opts.save_path, graph_def=self._session.graph_def)
# 		workers = []
# 		for _ in xrange(opts.concurrent_steps):
# 			t = threading.Thread(target=self._train_thread_body)
# 			t.start()
# 			workers.append(t)

# 		last_words, last_time, last_summary_time = initial_words, time.time(), 0
# 		last_checkpoint_time = 0
# 		while True:
# 			time.sleep(opts.statistics_interval)  # Reports our progress once a while.
# 			(epoch, step, loss, words, lr) = self._session.run(
# 					[self._epoch, self.global_step, self._loss, self._words, self._lr])
# 			now = time.time()
# 			last_words, last_time, rate = words, now, (words - last_words) / (
# 					now - last_time)
# 			print("Epoch %4d Step %8d: lr = %5.3f loss = %6.2f words/sec = %8.0f\r" %
# 						(epoch, step, lr, loss, rate), end="")
# 			sys.stdout.flush()
# 			if now - last_summary_time > opts.summary_interval:
# 				summary_str = self._session.run(summary_op)
# 				summary_writer.add_summary(summary_str, step)
# 				last_summary_time = now
# 			if now - last_checkpoint_time > opts.checkpoint_interval:
# 				self.saver.save(self._session, opts.save_path + "model", global_step=step.astype(int))
# 				last_checkpoint_time = now
# 			if epoch != initial_epoch:
# 				break

# 		for t in workers:
# 			t.join()

# 		return epoch

# 	def _predict(self, analogy):
# 		"""Predict the top 4 answers for analogy questions."""
# 		idx, = self._session.run([self._analogy_pred_idx], {
# 				self._analogy_a: analogy[:, 0],
# 				self._analogy_b: analogy[:, 1],
# 				self._analogy_c: analogy[:, 2]
# 		})
# 		return idx

# 	def eval(self):
# 		"""Evaluate analogy questions and reports accuracy."""

# 		# How many questions we get right at precision@1.
# 		correct = 0

# 		total = self._analogy_questions.shape[0]
# 		start = 0
# 		while start < total:
# 			limit = start + 2500
# 			sub = self._analogy_questions[start:limit, :]
# 			idx = self._predict(sub)
# 			start = limit
# 			for question in xrange(sub.shape[0]):
# 				for j in xrange(4):
# 					if idx[question, j] == sub[question, 3]:
# 						# Bingo! We predicted correctly. E.g., [italy, rome, france, paris].
# 						correct += 1
# 						break
# 					elif idx[question, j] in sub[question, :3]:
# 						# We need to skip words already in the question.
# 						continue
# 					else:
# 						# The correct label is not the precision@1
# 						break
# 		print()
# 		print("Eval %4d/%d accuracy = %4.1f%%" % (correct, total,
# 																							correct * 100.0 / total))

# 	def analogy(self, w0, w1, w2):
# 		"""Predict word w3 as in w0:w1 vs w2:w3."""
# 		wid = np.array([[self._word2id.get(w, 0) for w in [w0, w1, w2]]])
# 		idx = self._predict(wid)
# 		for c in [self._id2word[i] for i in idx[0, :]]:
# 			if c not in [w0, w1, w2]:
# 				return c
# 		return "unknown"

# 	def nearby(self, words, num=20):
# 		"""Prints out nearby words given a list of words."""
# 		ids = np.array([self._word2id.get(x, 0) for x in words])
# 		vals, idx = self._session.run(
# 				[self._nearby_val, self._nearby_idx], {self._nearby_word: ids})
# 		for i in xrange(len(words)):
# 			print("\n%s\n=====================================" % (words[i]))
# 			for (neighbor, distance) in zip(idx[i, :num], vals[i, :num]):
# 				print("%-20s %6.4f" % (self._id2word[neighbor], distance))

# def _start_shell(local_ns=None):
# 	# An interactive shell is useful for debugging/development.
# 	import IPython
# 	user_ns = {}
# 	if local_ns:
# 		user_ns.update(local_ns)
# 	user_ns.update(globals())
# 	IPython.start_ipython(argv=[], user_ns=user_ns)

# def main(_):
# 	"""Train a word2vec model."""
# 	opts = Options()
# 	with tf.Graph().as_default(), tf.Session() as session:
# 		model = Word2Vec(opts, session)
# 		for _ in xrange(opts.epochs_to_train):
# 			model.train()  # Process one epoch
# 			model.eval()  # Eval analogies.
# 		# Perform a final save.
# 		model.saver.save(session, os.path.join(opts.save_path, "model.ckpt"), global_step=model.global_step)
# 		if FLAGS.interactive:
# 			# E.g.,
# 			# [0]: model.analogy('france', 'paris', 'russia')
# 			# [1]: model.nearby(['proton', 'elephant', 'maxwell'])
# 			_start_shell(locals())

from gensim.models import Word2Vec
from gensim.models.word2vec import LineSentence
from gensim.models.word2vec import Text8Corpus
import multiprocessing
import sys, os
import logging

def main(unused_argv):
	program = os.path.basename(sys.argv[0])
	log = logging.getLogger(program)				# when word2vec training, it will output loger, define 1 to get it
	logging.basicConfig(format='%(asctime)s: %(levelname)s: %(message)s')
	logging.root.setLevel(level=logging.INFO)

	if (FLAGS.Train_Model):
		# sentences = LineSentence('text8')
		sentences = Text8Corpus('text8')
		model = Word2Vec(sentences,
						size = 200, 					# [vocabary_size, embed_size], here size=embed_size
						alpha = 0.05,					# initial value of lr	
						min_alpha = 0.001,
						window = 5,						# [-5 context, target, 5 context]
						min_count = 5, 					# counter<5 will be low freq word, remove
						sample = 0.001,					# the threshold of the subsample (high freq word remove threshold)
						workers = multiprocessing.cpu_count(),	# multi thread number
						sg = 1,							# [0:CBOW, 1:skip gram]
						hs = 0,							# [0:negtive sampling; 1:hierarchical softmax]
						negative = 25, 					# neg sampling, noise word number = 25
						ns_exponent = 0.75,				# 0.75=3/4 power of the neg sampling, see paper formula
						# hashfxn = ,					# there are 2 w and 1 b, use different initial, see basic.py
						iter = 15,						# number of epoch to train
						batch_words = 25)				# batch size

		model.save('./data.model')
		print(model.get_latest_training_loss())
		# model.wv.save_word2vec_format('.\model.bin', binary=true)		# save to bin file

		# todo: 中文语料(进行分词 jieba)整理
	else:
		model = Word2Vec.load('./data.model')
		# model = Word2Vec.load('./200d/data.model')
		# print(type(model.wv.index2word))
		# print(np.shape(model.wv.index2word))
		# print(type(model.)
		# print(np.shape(model._word2id))

		# sys.exit(0)
		# # most 3 similar word
		# print(model.wv.similar_by_word('good', 3), '\n')
		# # most similar 5 word
		# print(model.wv.most_similar('fuck', topn=5), '\n')
		# # Synonym, antonym
		# print(model.wv.most_similar(positive=['night', 'moon'], negative=['day']), '\n')
		# print(model.wv.most_similar(positive=['athens', 'greece'], negative=['baghdad']), '\n')
		# # Cosine approximation of 2 words
		# print(model.wv.n_similarity(['angel'], ['devil']), '\n')
		# # pick the word doesn't match in the list
		# print(model.wv.doesnt_match(['cat', 'knife', 'desk', 'chair', 'TV']), '\n')
		# #predict_output_word
		# print(model.wv['get'], '\n')						# get vector d=200
		# print(model.predict_output_word(['girl']), '\n')

		# # add new word to train after training done
		# model = Word2Vec.load('.\data.model')
		# sentences = Text8Corpus('text8')
		# model.train(sentences, epochs=15, total_words=255078105, start_alpha=0.001, end_alpha=0.0001)

		## peom creator......

		# # vector virsualization (200-d vector seems not good presentation in graph!!)
		# f = open('./td/metadata.tsv','a')
		# f.write("Index\tLabel\n")
		# for index,label in enumerate(model.wv.index2word):
		# 	f.write("%d\t%s\n" % (index,label))
		# f.close()
		# with tf.Graph().as_default():
		# 	# set a tensor to read the embed vector
		# 	img_emd = tf.Variable(model.wv.vectors, name='embeded')
		# 	saver = tf.train.Saver()
		# 	sess = tf.Session()
		# 	sess.run(img_emd.initializer)
		# 	summary_writer = tf.summary.FileWriter('./td', sess.graph)
		# 	checkpoint_path = os.path.join('./td/', 'img.ckpt')
		# 	saver.save(sess, checkpoint_path)
		# 	config = projector.ProjectorConfig()
		# 	# One can add multiple embeddings.
		# 	embedding = config.embeddings.add()
		# 	embedding.tensor_name = img_emd.name
		# 	# Link this tensor to its metadata file (e.g. labels).
		# 	embedding.metadata_path = 'metadata.tsv'
		# 	# Saves a config file that TensorBoard will read during startup.
		# 	projector.visualize_embeddings(summary_writer, config)
		# 	summary_writer.close()

		# #- solution 1, about 48%
		# score, _ = model.wv.evaluate_word_analogies('questions-words.txt')
		#- solution 2, about 51%
		model.wv.accuracy('questions-words.txt')
		# #- solution 3, the accuray is very low, only 41.5%, fine tune emb parameter may improve the accuracy
		# def predict(_a, _b, _c):
		# 	# get embed vector from model
		# 	_emb_vector = tf.Variable(model.wv.vectors, name='emb_vector')
		# 	# Normalized word embeddings of shape [vocab_size, emb_dim].
		# 	emb_vector = tf.nn.l2_normalize(_emb_vector, 1)

		# 	a_emb = tf.gather(emb_vector, _a)			# a's embs
		# 	b_emb = tf.gather(emb_vector, _b)			# a's embs
		# 	c_emb = tf.gather(emb_vector, _c)			# a's embs

		# 	result = c_emb + (b_emb - a_emb)	# [N, emb_size]
		# 	# dist has shape [N, vocab_size].
		# 	dist = tf.matmul(result, emb_vector, transpose_b=True)
		# 	# For each question (row in dist), find the top 4 words.x
		# 	_, result = tf.nn.top_k(dist, 4)
		# 	return result
		# q1 = []
		# questions_skipped = 0
		# with open('./questions-words.txt', "rb") as f:
		# 	for line in f:				# equivalent to f.readline()
		# 		if line.startswith(b":"):  # Skip comments
		# 			continue
		# 		words = line.strip().lower().split(b' ')			# split the word with ' '
		# 		# get the idx of word [idx1,idx2,idx3,idx4]
		# 		try:
		# 			ids = [model.wv.vocab[w.strip().decode(encoding='utf-8')].index for w in words]
		# 		except:
		# 			questions_skipped += 1					# remove unknow word
		# 		else:
		# 			q1.append(np.array(ids))				# [[idx1,idx2,idx3,idx4],...]
		# questions = np.array(q1)
		# print('skip number:', questions_skipped)			# change the type for easy dimension operation

		# total = np.shape(questions)[0]
		# start, correct, batch_size = 0, 0, 2500
		# with tf.Graph().as_default():
		# 	a = tf.placeholder(dtype=tf.int32, name='a_word')
		# 	b = tf.placeholder(dtype=tf.int32, name='b_word')
		# 	c = tf.placeholder(dtype=tf.int32, name='c_word')
		# 	result = predict(_a=a, _b=b, _c=c)
		# 	sess = tf.Session()
		# 	sess.run(tf.global_variables_initializer())
		# 	for step in range(int(total/batch_size)):
		# 		limit = start + batch_size					# 2500 questions every batch
		# 		if limit > total: 
		# 			limit = total 							# in case total%batch_size != 0
		# 		sub = questions[start:limit, :]				# generate the batch
		# 		_res = sess.run([result], feed_dict={a:sub[:, 0], b:sub[:, 1], c:sub[:, 2]})
		# 		start = limit					# next batch
		# 		res = _res[0]					# change tuple to list
		# 		for q in range(sub.shape[0]):
		# 			print('step %d'%step, [model.wv.index2word[x] for x in sub[q]], '==>', [model.wv.index2word[x] for x in res[q]])
		# 			for j in range(4):
		# 				if res[q, j] == sub[q, 3]:
		# 					# Bingo! We predicted correctly. E.g., [italy, rome, france, paris].
		# 					correct += 1
		# 					break
		# 				elif res[q, j] in sub[q, :3]:
		# 					# We need to skip words already in the question.
		# 					continue
		# 				else:
		# 					# The correct label is not the precision@1
		# 					break
		# print("Eval %4d/%d accuracy = %4.1f%%" % (correct, total, correct * 100.0 / total))

if __name__ == "__main__":
	tf.app.run()

		