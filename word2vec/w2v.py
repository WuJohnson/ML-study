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
		
		#-- normal usage of the model
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

		## todo: peom creator......

		# #-- vector virsualization (200-d vector seems not good presentation in graph!!)
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

		# #-- evaluate the model
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

		
