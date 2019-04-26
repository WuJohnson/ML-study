'''
chinese word pre-process for later word2vec 
@by Johnson Wu
'''

import tensorflow as tf
import jieba
import re           # regular expression model
from snownlp import SnowNLP as snlp
#===========
#- todo: remove some word by subsampling, e.g. 的，着，了等等
#- todo: 挑选短语,重新再分词一遍

def PreProcess():
	# sys.setdefaultencoding('utf8')
	chs_eng = {ord(c):ord(e) for c, e in zip(u'“”：；，。！？【】（）％＃＠＆１２３４５６７８９０',
											u'"":;,.!?[]()%#@&1234567890')}

	data = []
	with open('./cdata.txt', "rb") as f:
		for l in f:                         # equivalent to f.readline()
			l1 = l.strip().decode(encoding='utf-8')         # bytes to string
			if len(l1) <= 1: continue       # delete blank line which only include \n \r \r\n
			l2 = l1.translate(chs_eng)      # change chinese punctuation to english
			# line = re.split('[.?;!]+', l2)     # split the line with multi sub-string
			l3 = re.split('([.?;!]+)', l2)   # split the line with multi sub-string and keep sub-string
			# connect the punctuation and sentence, handle the odd length first
			if (len(l3)%2) != 0 and l3[-1] != '':
				l3.append('')
			line = [''.join(i) for i in zip(l3[0::2], l3[1::2])]
			data.extend(line)
		f.close()
	with open('./cdata_line.txt', 'wb') as f2:
		for x in data:
			if len(x) > 1:
				f2.write(x.encode(encoding='utf-8'))
				f2.write(b'\r\n')
		f2.close()

def TrySnowNLP():
	'''tryig the usage of SnowNLP word splitting'''
	with open('./cdata_line.txt', "rb") as f:        # the file encode is unicode
		doc = f.read()
		doc_code = doc.decode('utf-8')
		doc = snlp(doc_code)
		result = '/'.join(doc.words)              # '/' is the split character
		result = result.encode('utf-8')
		with open('./cdata_cut_snlp.txt', 'wb') as f2:
			f2.write(result)
			f2.close()
		f.close()
	pass

def TryJieba():
	'''trying the usage of jieba word splitting'''
	jieba.suggest_freq('伍定远', True)             # define a word which can't be split
	with open('./cdata_line.txt', "rb") as f:        # the file encode is unicode
		doc = f.read()
		doc_code = doc.decode('utf-8')

		doc_cut = jieba.cut(doc_code)
		result = '/'.join(doc_cut)              # '/' is the split character
		result = result.encode('utf-8')
		with open('./cdata_cut.txt', 'wb') as f2:
			f2.write(result)
			f2.close()
		f.close()

def main(_):
	# TryJieba()
	# PreProcess()
	TrySnowNLP()
if __name__ == "__main__":
	tf.app.run()
