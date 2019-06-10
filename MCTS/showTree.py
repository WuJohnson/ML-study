#coding:utf-8

"""
 This file provide 2 menthod to display a MCTS tree and its q()/n():
    1. matplotlib (can't solve the "too many nodes to display in the figure --out of range" problem)
    2. the Widget of PYQT
"""

import matplotlib.pyplot as plt
import numpy as np
from mcts.nodes import TwoPlayersGameMonteCarloTreeSearchNode
from games.common import TwoPlayersGameState

import sys
from PyQt5.QtWidgets import *

class TreeView(QMainWindow):
	"""
	use pyqt's TreeWidget to show tree
	"""

	def __init__(self, tree: TwoPlayersGameMonteCarloTreeSearchNode, parent=None):
		super(TreeView, self).__init__(parent)
		self.setWindowTitle('Display Tree')
		# create tree
		self.view = QTreeWidget()
		# set tree view column, just 1 column
		self.view.setColumnCount(1)
		self.view.setHeaderHidden(True)

		self.tree = tree
		root = QTreeWidgetItem(self.view)
		root.setText(0, 'MCTS tree')
		self.LoadTree(root, self.tree)

		# add all widget
		self.view.addTopLevelItem(root)
		# expand tree
		# self.view.expandAll()
		self.setCentralWidget(self.view)
		self.show()


	def LoadTree(self, viewNode, node):
		viewNode = QTreeWidgetItem(viewNode)
		viewNode.setText(0, 'q:%d/n:%d'%(node.q, node.n))

		for ch in node.children:
			self.LoadTree(viewNode, ch)


class TreeFigure():
	"""
	use matplotlib to display a tree
	input a tree
	"""
	def __init__(self, tree: TwoPlayersGameMonteCarloTreeSearchNode):
		self.tree = tree
		self.depth = self.GetTreeDepth(tree, 0, 0)

	def GetTreeDepth(self, node:TwoPlayersGameMonteCarloTreeSearchNode, j, depth):
		"""get tree depth"""
		j += 1                                  # depth counter++
		depth = j if depth < j else depth       # update the deepest depth
		for chi in node.children:
			depth = self.GetTreeDepth(chi, j, depth)
		return depth

	def GetTreeDepthNodesNumber(self, node:TwoPlayersGameMonteCarloTreeSearchNode, depth):
		"""
		node: the root node of the tree
		depth: tree depth level, depth 0 means root

		calculate all nodes in same layer of node. 不能用递归,因为递归是深度优先搜索,
		所以对应某一层, 会先深入到其子节点再回溯
		e.g. the children of the root layer = len(root.children)
			 the next layer  = sum([len(x.children) for x in root.children])

		:return: int
		"""
		childList = [node]          # default is root node
		j = 0
		while j < depth:
			tl = []
			for x in childList:
				tl.extend(x.children)   # get all child nodes of the depth level
			childList = tl
			j += 1
		return len(childList)

	def DrawNode(self, node:TwoPlayersGameMonteCarloTreeSearchNode, ax, xy, xytext):
		"""
		draw a single node, and an arrow connect to its parent
		:node: the node need to be drawn
		:param ax: the ax of the figure
		:xy: the parent node coordinate
		:xytext: the coordinate of current node
		:return: annotate handler
		"""
		if (node.parent is None):
			an = ax.annotate(s=('q:%d' % node.q + '\n' + 'n:%d' % node.n), xy=xytext, xytext=xytext,
			                 textcoords='axes fraction', va="center", ha="center",
			                 bbox=dict(boxstyle='round', fc='y'),
			                 arrowprops=dict(arrowstyle='<-'))
		else:
			an = ax.annotate(s=('q:%d'%node.q+'\n' + 'n:%d'%node.n), xy=xy, xytext=xytext,
			             textcoords='axes fraction', va="center", ha="center",
			             bbox=dict(boxstyle='round', fc='y'),
			             arrowprops=dict(arrowstyle='<-'))
		return an

	def DrawTree(self, node:TwoPlayersGameMonteCarloTreeSearchNode, xy, x_interval, y_interval, ax, layer, idx_l):
		"""
		draw whole tree
		:param node: the root node of the tree
		:param xy:  parent node coordinate (recursion--can't use local variable)
		:param x_interval: the nodes interval at same layer in x coordinate
		:param y_interval: the nodes interval at different layer
		:param ax: the sub-plot handler, use plt.subplot to create it
		:param layer: indicate the current layer of the tree (recursion--can't use local variable)
		:param idx_l: list of the nodes index at x coordinate e.g. idx_l[1]=3: 3rd node of the layer1 at x coordinate
		"""

		xytext = (idx_l[layer]*x_interval, (layer+1)*y_interval)
		self.DrawNode(node, ax, xy=xy, xytext=xytext)

		layer += 1              # next layer
		xy = xytext
		xytext = (xytext[0], xytext[1]+y_interval)
		for ch in node.children:
			self.DrawTree(ch, xy, x_interval, y_interval, ax, layer, idx_l)
			idx_l[layer] += 1  # point next neighbor position


	def ShowTree(self):
		"""	draw the tree """
		nl = [self.GetTreeDepthNodesNumber(self.tree, x) for x in range(self.depth)]
		wi = max(nl)
		fig = plt.figure(1, facecolor='white', figsize=(0.04*wi, 8))          # 定义一个画布，背景为白色
		fig.clf()                                       # 清空画布
		axprops = dict(xticks=[], yticks=[])            # 定义横纵坐标轴，无内容
		ax = plt.subplot(111, frameon=False, **axprops) # 绘制图像,无边框,无坐标轴, x:0~1; y:0~1
		# ax = plt.subplot(111, frameon=False) # 绘制图像,无边框, x:0~1; y:0~1

		# - define the Y interval through the self.depth, X interval by nodes number of the depth
		y_interval = 1.0/(self.depth+1)             # 1: 0.5 of top margin + 0.5 of bottom margin

		# # -get max nodes of all layers
		# nl = [self.GetTreeDepthNodesNumber(self.tree, x) for x in range(self.depth)]
		# wi = max(nl)
		# # - define the X interval by  the wi
		# x_interval = 1./(wi+1)                       # 1: 0.5 of left margin + 0.5 of right margin
		x_interval = 0.04                       # 1: 0.5 of left margin + 0.5 of right margin

		layer = 0
		idx_l = [1.]*self.depth
		xy = (0., 0.)
		self.DrawTree(self.tree, xy, x_interval, y_interval, ax, layer, idx_l)
		plt.show()

if __name__ == "__main__":

	#============== test for GetTreeDepth========
	state = np.zeros((3, 3))
	state = TwoPlayersGameState(state, (0,0,1))
	tree = TwoPlayersGameMonteCarloTreeSearchNode(state, parent=None)
	a = TwoPlayersGameMonteCarloTreeSearchNode(state, parent= tree)
	tree.children.append(a)
	c = TwoPlayersGameMonteCarloTreeSearchNode(state, parent= tree)
	tree.children.append(c)
	d = TwoPlayersGameMonteCarloTreeSearchNode(state, parent= tree)
	tree.children.append(d)
	b = TwoPlayersGameMonteCarloTreeSearchNode(state, parent= a)
	a.children.append(b)
	e = TwoPlayersGameMonteCarloTreeSearchNode(state, parent= a)
	a.children.append(e)
	h = TwoPlayersGameMonteCarloTreeSearchNode(state, parent= e)
	e.children.append(h)
	f = TwoPlayersGameMonteCarloTreeSearchNode(state, parent= c)
	c.children.append(f)
	g = TwoPlayersGameMonteCarloTreeSearchNode(state, parent= d)
	d.children.append(g)

	# tr = TreeFigure(tree)
	# tr.ShowTree()

	# create a app (window)
	app = QApplication(sys.argv)
	tr = TreeView(tree)
	sys.exit(app.exec_())