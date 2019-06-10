import numpy as np
from collections import defaultdict
from games.tictactoe import *
from games.common import TwoPlayersGameState

class TwoPlayersGameMonteCarloTreeSearchNode:
	"""
	base class
	"""

	def __init__(self, state: TwoPlayersGameState, parent = None):
		self.state = state
		self.parent = parent
		self.children = []

	@property
	def untried_actions(self):
		raise NotImplemented()

	@property
	def q(self):
		raise NotImplemented()

	@property
	def n(self):
		raise NotImplemented()

	def expand(self):
		raise NotImplemented()

	def is_terminal_node(self):
		raise NotImplemented()

	def rollout(self):
		raise NotImplemented()

	def backpropagate(self, reward):
		raise NotImplemented()


	def is_fully_expanded(self):
		return len(self.untried_actions) == 0

	def best_child(self, c_param = 1.4):
		choices_weights = [
			(c.q / (c.n)) + c_param * np.sqrt((2 * np.log(self.n) / (c.n)))
			for c in self.children
		]
		return self.children[np.argmax(choices_weights)]

	def rollout_policy(self, possible_moves):
		return possible_moves[np.random.randint(len(possible_moves))]

class TwoPlayersGameMonteCarloTreeSearchNode(TwoPlayersGameMonteCarloTreeSearchNode):

	def __init__(self, state: TwoPlayersGameState, parent):
		super(TwoPlayersGameMonteCarloTreeSearchNode, self).__init__(state, parent)
		self._number_of_visits = 0.
		# - collections.defaultdict: if there is key_err, default value for error key is (int) == 0
		self._results = defaultdict(int)

	@property
	def untried_actions(self):
		if not hasattr(self, '_untried_actions'):
			self._untried_actions = self.state.get_legal_actions()
		return self._untried_actions

	@property
	def q(self):
		try:
			wins = self._results[self.parent.state.next_to_move]    # - if parent==None, default=0
			# - if there is not -next_to_move, value=0
			loses = self._results[-1 * self.parent.state.next_to_move]
		except:
			wins  = 0
			loses = 0
		return wins - loses

	@property
	def n(self):
		return self._number_of_visits

	def attach(self, state):
		"""
		search state in che child nodes, return the node if exist,
		create a new node if didn't exist,
		:return: TwoPlayersGameMonteCarloTreeSearchNode
		"""
		child_board_list = [c.state.board for c in self.children]
		try:
			idx = child_board_list.index(state.board)
			return self.children[idx]
		except:
			print('can not find state in the child node list, create a new node...')
			child_node = TwoPlayersGameMonteCarloTreeSearchNode(state, parent = self)
			self.children.append(child_node)
			return child_node

	def expand(self):
		action = self.untried_actions.pop()
		next_state = self.state.move(action)
		child_node = TwoPlayersGameMonteCarloTreeSearchNode(next_state, parent = self)
		self.children.append(child_node)
		return child_node

	def is_terminal_node(self):
		return self.state.is_game_over()

	def rollout(self):
		"""
		expand create the node, rollout didn't create the node
		:return: game_result
		"""
		current_rollout_state = self.state
		while not current_rollout_state.is_game_over():
			possible_moves = current_rollout_state.get_legal_actions()
			action = self.rollout_policy(possible_moves)           # - random select moves
			current_rollout_state = current_rollout_state.move(action)
		return current_rollout_state.game_result

	def backpropagate(self, result):
		"""
		self = the child node of expand
		:param result: int
		"""
		self._number_of_visits += 1.
		self._results[result] += 1.
		if self.parent:
			self.parent.backpropagate(result)
