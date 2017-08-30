import numpy as np
import time, math, random, sys
from datetime import timedelta, datetime
from copy import copy
from multiprocessing.pool import ThreadPool as Pool

#from multiprocessing import Pool
from itertools import repeat, starmap

from gp_active_learning import get_distance_matrix, \
	get_distance_matrix_gps, read_entry_gp_distance_matrix, \
	get_gp_dm_mean_stdev, \
	get_entry_gp_dm_element
from solve_tsp import get_shortest_tour, check_tour_length
from functools import partial

class memoize(object):
    """
    from https://code.activestate.com/recipes/577452-a-memoize-decorator-for-instance-methods/

    cache the return value of a method
    
    This class is meant to be used as a decorator of methods. The return value
    from a given method invocation will be cached on the instance whose method
    was invoked. All arguments passed to a method decorated with memoize must
    be hashable.
    
    If a memoized method is invoked directly on its class the result will not
    be cached. Instead the method will be invoked like a static method:
    class Obj(object):
        @memoize
        def add_to(self, arg):
            return self + arg
    Obj.add_to(1) # not enough arguments
    Obj.add_to(1, 2) # returns 3, result is not cached
    """
    def __init__(self, func):
        self.func = func
    def __get__(self, obj, objtype=None):
        if obj is None:
            return self.func
        return partial(self, obj)
    def __call__(self, *args, **kw):
        obj = args[0]
        try:
            cache = obj.__cache
        except AttributeError:
            cache = obj.__cache = {}
        key = (self.func, args[1:], frozenset(kw.items()))
        try:
            res = cache[key]
        except KeyError:
            res = cache[key] = self.func(*args, **kw)
        return res

def prob_round(x):
	'''
	Probabilistically round up or down based on decimal value.
	'''
	sign = np.sign(x)
	x = abs(x)
	is_up = random.random() < x-int(x)
	round_func = math.ceil if is_up else math.floor
	return sign * round_func(x)

def round_time(dt=None, roundTo=60):
	"""Round a datetime object to any time laps in seconds
	dt : datetime.datetime object, default now.
	roundTo : Closest number of seconds to round to, default 1 minute.
	Author: Thierry Husson 2012 - Use it as you want but don't blame me.
	"""
	if dt == None : dt = datetime.now()
	seconds = (dt.replace(tzinfo=None) - dt.min).seconds
	rounding = (seconds+roundTo/2) // roundTo * roundTo
	return dt + timedelta(0,rounding-seconds,-dt.microsecond)

# These classes aim to closely mimic the openAI gym interface
class TSP:
	def __init__(self, addresses, departure_time):
		self.addresses = addresses
		self.initial_time = departure_time
		self.dm = get_distance_matrix(addresses, departure_time)
		self.reset()

	def copy(self):
		other = copy(self)
		other.tour = self.tour[:]
		return other

	def reset(self):
		self.current_city = 0
		self.tour_length = 0
		self.tour = [0]

	def get_edge_cost(self,i,j):
		return self.dm[i,j]

	def step(self, action):
		done = False
		r = 0
		# if valid action, do action
		if action not in self.tour or \
			action == 0  and len(self.tour) == len(self.addresses):

			edge_length = self.get_edge_cost(self.current_city, action)

			self.tour_length += edge_length
			self.current_city = action
			self.tour += [action]
			# if we've now visited all cities, return to origin
			if len(self.action_space()) == 0:
				edge_length = self.get_edge_cost(self.current_city, 0)
				self.tour_length += edge_length
				self.current_city = 0
				r = -self.tour_length
				done = True
			info = {'valid':True}
		else:
			info = {'valid':False}
		return (self.state, r, done, info)

	def action_space(self):
		return [a for a in range(len(self.addresses)) if a not in self.tour]

	def state(self):
		return (self.current_city, self.tour)

	def __repr__(self):
		return str({'state': self.state(), 'addresses': self.addresses, 'class': self.__class__,
			'tour_length': self.tour_length})

class DCTSP(TSP):
	def __init__(self, addresses, departure_time):
		self.addresses = addresses
		self.initial_time = departure_time
		self.gp_dm = get_distance_matrix_gps(addresses)
		self.reset()

	def reset(self):
		super().reset()
		self.current_time = self.initial_time

	@memoize
	def _check_dm(self, i, j, datetime):
		return read_entry_gp_distance_matrix(self.gp_dm[i,j], datetime)[1]

	def get_edge_cost(self,i,j):
		rounded_time = round_time(self.current_time, 60*10)
		edge_cost = self._check_dm(i, j, rounded_time)
		self.current_time += timedelta(hours=1, seconds=edge_cost)
		return edge_cost

class PCTSP(TSP):
	def __init__(self, addresses, departure_time):
		self.addresses = addresses
		self.initial_time = departure_time
		self.gp_dm = get_distance_matrix_gps(addresses)
		self.means, self.stdevs = get_gp_dm_mean_stdev(self.gp_dm, departure_time)
		self.reset()

	def get_edge_cost(self,i,j):
		if i == j: return 0
		mean = self.means[i,j]
		stdev = self.stdevs[i,j]
		min_cost = self.gp_dm[i,j][1]

		val = np.random.normal(loc=mean, scale=stdev)
		edge_cost = np.exp(val) + min_cost
		return edge_cost

class STSP(TSP):
	def __init__(self, addresses, departure_time):
		self.addresses = addresses
		self.initial_time = departure_time
		self.gp_dm = get_distance_matrix_gps(addresses)
		self.reset()

	def reset(self):
		super().reset()
		self.current_time = self.initial_time

	@memoize
	def _check_dm(self, i, j, datetime):
		return get_entry_gp_dm_element(self.gp_dm[i,j], datetime)

	def get_edge_cost(self,i,j):
		rounded_time = round_time(self.current_time, 60*10)
		mean, stdev, min_cost = self._check_dm(i, j, rounded_time)

		val = np.random.normal(loc=mean, scale=stdev)
		edge_cost = np.exp(val) + min_cost

		self.current_time += timedelta(hours=1, seconds=edge_cost)
		return edge_cost

def test_tour(tsp, tour):
	tsp.reset()
	for a in tour[1:]:
		s,r,d,i = tsp.step(a)
	return tsp.tour_length

def get_tour_length_distribution(tsp, tour, num_iter=50):
	if type(tsp) in (TSP, DCTSP):
		num_iter = 1 # no probabilistic effects, no need to repeat

	rs = []
	for i in range(num_iter):
		rs += [test_tour(tsp, tour)]
	return rs

def get_lkh_tour(tsp, traffic_model = 'best_guess'):
	if hasattr(tsp, 'dm') and traffic_model == 'best_guess':
		dm = tsp.dm
	else:
		dm = get_distance_matrix(tsp.addresses, 
			departure_time=tsp.initial_time, traffic_model=traffic_model)
	tour, tour_length = get_shortest_tour(dm)
	# enforce tour begins at 0
	idx = tour.index(0)
	tour = tour[idx:] + tour[:idx]
	return tour

def get_dynamic_lkh_tour(tsp, return_reward=False, traffic_model = 'best_guess'):
	tsp.reset()
	current_time = tsp.initial_time

	dm = get_distance_matrix(tsp.addresses, 
		departure_time=current_time, traffic_model=traffic_model)
	tour, tour_length = get_shortest_tour(dm)
	# enforce tour begins at 0
	idx = tour.index(0)
	tour = tour[idx:] + tour[:idx]

	final_tour = tour[0:2]
	tsp.step(final_tour[1])

	if hasattr(tsp, 'current_time') and return_reward:
		current_time = tsp.current_time
	else:
		trip_length = dm[final_tour[-2],final_tour[-1]]
		current_time += timedelta(hours=1, seconds=int(trip_length))
	# construct tour 1 step at a time
	while len(final_tour) < len(tsp.addresses):
		# get distance matrix for current time
		trip_length = dm[final_tour[-2],final_tour[-1]]
		dm = get_distance_matrix(tsp.addresses, 
			departure_time=current_time, traffic_model=traffic_model)
		# encourage following previous steps
		for i,j in zip(final_tour[:-1], final_tour[1:]):
			dm[i,j] = -10000
		# get next step of tour
		tour, tour_length = get_shortest_tour(dm)
		next_move = tour[(tour.index(final_tour[-1])+1) % len(tour)]
		_,r,_,_ = tsp.step(next_move)
		final_tour += [next_move]
		if hasattr(tsp, 'current_time') and return_reward:
			current_time = tsp.current_time
		else:
			trip_length = dm[final_tour[-2],final_tour[-1]]
			current_time += timedelta(hours=1, seconds=int(trip_length))

	if return_reward:
		return (final_tour, r)
	return final_tour

def do_lkh_rollout(partial_tour, dm):
	#print("Rollout", tsp)
	original_dm = dm.copy()

	# encourage following previous steps
	for i,j in zip(partial_tour[:-1], partial_tour[1:]):
		dm[i,j] = -10000
	# use LKH to get tour
	tour, tour_length = get_shortest_tour(dm)
	# check that tour length in the DM we got
	tour_length = check_tour_length(original_dm, tour)
	
	return -tour_length

def get_mc_lkh_tour(tsp, num_sims=20, return_reward = False, use_max = False):
	tsp.reset()
	action_sequence = [0]
	reward = 0
	level = 0
	#pool = Pool()
	# main loop to select actions one at a time
	# (greedy approach?)
	done = False
	while not(done) > 0:
		level_budget = num_sims/(level+2)
		level += 1
		#print("Level budget:",level_budget)

		actions = tsp.action_space()
		action_rewards = []

		action_budget = level_budget/len(actions)
		#print("Actions:",len(actions))
		#print("Action budget:",action_budget)
		# evaluate each available action
		dm = np.zeros((len(tsp.addresses),len(tsp.addresses)))
		for i in range(len(tsp.addresses)):
			for j in range(len(tsp.addresses)):
				dm[i,j] = int(tsp.get_edge_cost(i,j))

		for action in actions:
			args = [(tsp.tour + [action], dm.copy()) for _ in range(math.floor(action_budget))]
			rollout_rewards = list(starmap(do_lkh_rollout, args))
			if use_max:
				action_rewards += [np.max(rollout_rewards)]
			else:
				action_rewards += [np.mean(rollout_rewards)]

		# select action to maximise reward
		selected_action = actions[np.argmax(action_rewards)]
		action_sequence += [selected_action]
		_,r,done,_ = tsp.step(selected_action)
		reward += r
		#print("Action Rewards:", action_rewards)
		#print("Select Action:", selected_action)
		#print("Action Sequence:", action_sequence)
		#print("Reward so far:", reward)
	if return_reward:
		return (action_sequence, reward)
	return action_sequence

def get_mc_tour(tsp, num_sims=20000, return_reward = False, use_max = False):
	tsp.reset()
	action_sequence = [0]
	reward = 0
	sims_done = 0
	level = 0
	# main loop to select actions one at a time
	# (greedy approach?)
	done = False
	while not(done) > 0:
		level_budget = num_sims/(level+2)
		level += 1
		#print("Level budget:",level_budget)

		actions = tsp.action_space()
		action_rewards = []

		action_budget = level_budget/len(actions)
		#print("Actions:",len(actions))
		#print("Action budget:",action_budget)
		# evaluate each available action
		for action in actions:
			# MC rollouts
			rollout_rewards = []
			for k in range(math.floor(action_budget)):
				sims_done+=1
				tsp_copy = tsp.copy()
				_,r,done_rollout,_ = tsp_copy.step(action)
				while not(done_rollout):
					_,r,done_rollout,_ = tsp_copy.step(
						np.random.choice(tsp_copy.action_space()))
				rollout_rewards += [r]
			if use_max:
				action_rewards += [np.max(rollout_rewards)]
			else:
				action_rewards += [np.mean(rollout_rewards)]

		# select action to maximise reward
		selected_action = actions[np.argmax(action_rewards)]
		action_sequence += [selected_action]
		_,r,done,_ = tsp.step(selected_action)
		reward += r
	if return_reward:
		return (action_sequence, reward)
	return action_sequence

def get_mcts_tour(tsp, num_sims=20000, return_reward = False, use_max = False):
	'''
	Based on https://github.com/haroldsultan/MCTS/blob/master/mcts.py
	'''
	# MCTS scalar.  Larger scalar will increase exploration, smaller will increase exploitation. 
	# aim to have scalar such that exploit (reward) is on [0-1] scale
	tsp.reset()
	lkh_tour = get_lkh_tour(tsp)
	lkh_tour_length = test_tour(tsp, lkh_tour)

	SCALAR = lkh_tour_length / math.sqrt(2.0)

	class State():
		def __init__(self, tsp, reward, done):
			self.tsp = tsp
			self.r = reward
			self.done = done
			self.num_moves = len(self.tsp.action_space())
		def next_state(self):
			#print("nextstating")
			nextmove = random.choice(self.tsp.action_space())
			next_tsp = self.tsp.copy()
			_,r,d,i = next_tsp.step(nextmove)
			return State(next_tsp, r, d)
		def copy_state(self, state):
			if self.tsp.tour != state.tsp.tour[:len(self.tsp.tour)]:
				print(self)
				print(state)
				raise Exception("Trying to copy different states")

			copy_tsp = self.tsp.copy()
			actions_to_do = [a for a in state.tsp.tour if a not in self.tsp.tour]
			for action in state.tsp.tour[len(self.tsp.tour):]:
				_,r,d,i = copy_tsp.step(action)
			return State(copy_tsp, r, d)
		def terminal(self):
			return self.done
		def reward(self):
			return self.r
		def __hash__(self):
			return int(hashlib.md5(str(self.tsp.state)).hexdigest(),16)
		def __eq__(self,other):
			return self.tsp.state() == other.tsp.state()
		def __repr__(self):
			return str(self.tsp.__repr__())
			return "MCTS_State:" + str(self.tsp.__repr__()) + ",reward:" + str(self.r) + ",done:" + str(self.done)

	class Node():
		def __init__(self, state, parent=None):
			self.visits=1
			self.reward=0.0	
			self.rewards = []
			self.state=state
			self.children=[]
			self.parent=parent	
		def add_child(self,child_state):
			child=Node(child_state,self)
			self.children.append(child)
		def update(self,reward):
			self.reward+=reward
			self.rewards += [reward]
			self.visits+=1
		def fully_expanded(self):
			if len(self.children)==self.state.num_moves:
				return True
			return False
		def __repr__(self):
			s="Node; children: %d; visits: %d; reward: %f"%(len(self.children),self.visits,self.reward)
			return str({'Children':len(self.children), 'Visits':self.visits, 
				'Reward': self.reward, 'State': self.state})

	def UCTSEARCH(budget,root):
		for iter in range(math.floor(budget)):
			front=TREEPOLICY(root)
			reward=DEFAULTPOLICY(front)
			BACKUP(front,reward)
		return BESTCHILD(root,0)

	def TREEPOLICY(node):
		while node.state.terminal()==False:
			if node.fully_expanded()==False:
				return EXPAND(node)
			else:
				node=BESTCHILD(node,SCALAR)
		return node

	def EXPAND(node):
		tried_children=[c.state for c in node.children]
		new_state=node.state.next_state()
		while new_state in tried_children:
			new_state=node.state.next_state()
		node.add_child(new_state)
		return node.children[-1]

	# current this uses the most vanilla MCTS formula 
	# it is worth experimenting with THRESHOLD ASCENT (TAGS)
	def BESTCHILD(node,scalar):

		def get_score(c):
			# score by max reward, or average reward?
			# np.mean or np.max
			if use_max:
				exploit = np.max(c.rewards)
			else:
				exploit = np.mean(c.rewards)
			explore = math.sqrt(math.log(2*node.visits)/float(c.visits))
			#print(exploit, scalar*explore)
			return exploit+scalar*explore

		scores = list(map(get_score, node.children))
		best = np.array(node.children)[np.argwhere(scores == np.amax(scores))]
		return np.random.choice(best.flatten())

	def DEFAULTPOLICY(node):
		# duplicate state by rolling out from current state
		state = current_node.state.copy_state(node.state)
		# roll out to termination
		while state.terminal() == False:
			state = state.next_state()
		return state.reward()

	def BACKUP(node,reward):
		while node!=None:
			node.update(reward)
			node=node.parent
		return

	levels = len(tsp.addresses)-1

	tsp.reset()
	current_node = Node(State(tsp, 0, False))

	for l in range(levels):
		best_child = UCTSEARCH(num_sims/(l+2),current_node)
		#print("level %d"%l)
		#print("Num Children: %d"%len(current_node.children))
		#for i,c in enumerate(current_node.children):
		#	print(i,c)
		#print("Best Child: %s"%best_child.state)
		state = current_node.state.copy_state(best_child.state)
		current_node = best_child
		current_node.state = state
		#print("Tour so far:", state.tsp.tour)
		#print("Length so far:", state.tsp.tour_length)
		#print("--------------------------------")
		# clear tree below, as taking action has now updated the problem to solve
		current_node.children = []
		

	if return_reward:
		return (current_node.state.tsp.tour, -current_node.state.tsp.tour_length)
	return current_node.state.tsp.tour

def build_test_times():
	addresses = ['B70 8HL UK', 'ST16 2PD UK', 'ST3 4LD UK', 'ST3 4QP UK', 'ST5 6QX UK']
	departure_time = datetime.strptime('01/01/18 08:00','%d/%m/%y %H:%M')

	print("Building TSP");t=time.time()
	tsp = TSP(addresses, departure_time)
	print("\tTime:",time.time()-t)
	print("Building DCTSP");t=time.time()
	dctsp = DCTSP(addresses, departure_time)
	print("\tTime:",time.time()-t)
	print("Building PCTSP");t=time.time()
	pctsp = PCTSP(addresses, departure_time)
	print("\tTime:",time.time()-t)
	print("Building STSP");t=time.time()
	stsp = STSP(addresses, departure_time)
	print("\tTime:",time.time()-t)

	tour = get_lkh_tour(tsp)

	print("Testing TSP");t=time.time()
	print(test_tour(tsp,tour))
	print("\tTime:",time.time()-t)
	print("Testing DCTSP");t=time.time()
	print(test_tour(dctsp,tour))
	print("\tTime:",time.time()-t)
	print("Testing PCTSP");t=time.time()
	print(test_tour(pctsp,tour))
	print("\tTime:",time.time()-t)
	print("Testing STSP");t=time.time()
	print(test_tour(stsp,tour))
	print("\tTime:",time.time()-t)

def test_tour_building():
	addresses = ['B70 8HL UK', 'ST16 2PD UK', 'ST3 4LD UK', 'ST3 4QP UK', 'ST5 6QX UK']
	departure_time = datetime.strptime('01/01/18 08:00','%d/%m/%y %H:%M')

	tsp = TSP(addresses, departure_time)
	t=time.time()
	lkh_tour = get_lkh_tour(tsp)
	print(lkh_tour)
	print("\tTime:",time.time()-t)

	pctsp = PCTSP(addresses, departure_time)
	t=time.time()
	mc_tour_pctsp = get_mc_tour(pctsp)
	print(mc_tour_pctsp)
	print("\tTime:",time.time()-t)
	t=time.time()
	mcts_tour_pctsp = get_mcts_tour(pctsp)
	print(mcts_tour_pctsp)
	print("\tTime:",time.time()-t)

	stsp = STSP(addresses, departure_time)
	t=time.time()
	mc_tour_stsp = get_mc_tour(stsp)
	print(mc_tour_stsp)
	print("\tTime:",time.time()-t)
	t=time.time()
	mcts_tour_stsp = get_mcts_tour(stsp)
	print(mcts_tour_stsp)
	print("\tTime:",time.time()-t)

	print("Getting tour length distributions")

	t=time.time()
	lkh_tour_lengths = get_tour_length_distribution(stsp, lkh_tour)
	print("\tTime:",time.time()-t)

	t=time.time()
	mc_tour_pctsp_lengths = get_tour_length_distribution(stsp, mc_tour_pctsp)
	print("\tTime:",time.time()-t)

	t=time.time()
	mc_tour_stsp_lengths = get_tour_length_distribution(stsp, mc_tour_stsp)
	print("\tTime:",time.time()-t)

	t=time.time()
	mcts_tour_pctsp_lengths = get_tour_length_distribution(stsp, mcts_tour_pctsp)
	print("\tTime:",time.time()-t)

	t=time.time()
	mcts_tour_stsp_lengths = get_tour_length_distribution(stsp, mcts_tour_stsp)
	print("\tTime:",time.time()-t)

	from matplotlib import pyplot as plt
	plt.hist(lkh_tour_lengths, normed=True, cumulative=True, 
		label='lkh_tour_lengths', histtype='step', bins = 50,
		alpha=0.8)
	plt.hist(mc_tour_pctsp_lengths, normed=True, cumulative=True, 
		label='mc_tour_pctsp_lengths', histtype='step', bins = 50,
		alpha=0.8)
	plt.hist(mc_tour_stsp_lengths, normed=True, cumulative=True, 
		label='mc_tour_stsp_lengths', histtype='step', bins = 50,
		alpha=0.8)
	plt.hist(mcts_tour_pctsp_lengths, normed=True, cumulative=True, 
		label='mcts_tour_pctsp_lengths', histtype='step', bins = 50,
		alpha=0.8)
	plt.hist(mcts_tour_stsp_lengths, normed=True, cumulative=True, 
		label='mcts_tour_stsp_lengths', histtype='step', bins = 50,
		alpha=0.8)
	plt.legend()
	plt.show()

def test_mcts():
	addresses = ['B70 8HL UK', 'ST16 2PD UK', 'ST3 4LD UK', 'ST3 4QP UK', 'ST5 6QX UK']
	departure_time = datetime.strptime('01/01/18 08:00','%d/%m/%y %H:%M')

	def get_tour_dists(get_x_tour, tsp, niter=500, online=True, lkh=False, **kwargs):
		# kwargs:
		# MC/MCTS: num_sims, use_max
		# LKH: traffic_model
		if lkh:
			if online:
				tour_dists = list(map(
					lambda x: -get_x_tour(tsp, return_reward=True, **kwargs)[1], 
					range(niter)))
			else:
				tour = get_x_tour(tsp, **kwargs)
				tour_dists = get_tour_length_distribution(tsp, tour, num_iter=niter)
		elif online:
			tour_dists = list(map(
				lambda x: -get_x_tour(tsp, return_reward=True, **kwargs)[1], 
				range(niter)))
		else:
			num_tours = 50
			tours = list(map(
				lambda x: get_x_tour(tsp, return_reward=False, **kwargs), 
				range(num_tours)))
			tour_dists = []
			for tour in tours:
				tour_dists.extend(
					get_tour_length_distribution(tsp, tour, num_iter=int(niter/num_tours)))
		return tour_dists

	'''tsp = PCTSP(addresses, departure_time)
	tours = set()
	for k in range(100):
		dm = np.zeros((len(tsp.addresses),len(tsp.addresses)))
		for i in range(len(tsp.addresses)):
			for j in range(len(tsp.addresses)):
				dm[i,j] = int(tsp.get_edge_cost(i,j))
		tour, tour_length = get_shortest_tour(dm)
		idx = tour.index(0)
		tour = tour[idx:] + tour[:idx]
		tours.add(tuple(tour))
		print(k,tour,len(tours))
	print(len(tours))

	raise Exception()'''

	# PMCS, a-prior vs real time
	import pandas as pd
	data = pd.read_csv('boiler_repair.csv')
	from data_analysis import strip_small_jobs, get_jobs, get_addresses
	jobs = strip_small_jobs(data, get_jobs(data))
	departure_time = datetime.strptime('01/01/18 08:00','%d/%m/%y %H:%M')
	distss = []
	for i,job in enumerate(jobs):
		print(job)
		print(i+1, '/', len(jobs))
		dists = None
		while i+1 > 18 and dists == None:
			try:
				addresses = get_addresses(job, data)
				tsp = STSP(addresses, departure_time)
				n = 100
				print("mc-lkh", n)
				dists = get_tour_dists(get_mc_lkh_tour, tsp, niter=100, online=True, num_sims=n)
				#print("real time lkh", n)
				#dists = get_tour_dists(get_dynamic_lkh_tour, tsp, niter=100, online=True)
				distss += [dists]
				print(dists)
			except Exception as e:
				print(e)
	print("FINISH")
	print(distss)

	# do these a-priori to save time
	# pmcs/mcts evaluation (ALL IN PROGRESS!)
		# TODO: this but with 100, 5000, 20000 rollouts on PMCS
		# TODO: this but with MCTS vs PMCS (10000 rollouts)
	# lkh planning approaches
		# this but with LKH static vs dynamic
		# this but with LKH different percentiles
	# costs of assumptions
		# this but with PMCS, 10000 rollouts on TSP, PCTSP, DCTSP then tours evaluated in STSP

	# a-priori, LKH
	'''tsp = STSP(addresses, departure_time)
	for tm in ['optimistic', 'best_guess', 'pessimistic']:
		print(tm)
		print("static lkh")
		lkh_tour_dists = get_tour_dists(
			get_lkh_tour, tsp, online=False, lkh=True, traffic_model=tm)
		print(lkh_tour_dists)
		print("dynamic lkh")
		lkh_tour_dists = get_tour_dists(
			get_dynamic_lkh_tour, tsp, online=False, lkh=True, traffic_model=tm)
		print(lkh_tour_dists)'''

	# a-priori, MC / MCTS
	'''tsp = STSP(addresses, departure_time)
	print("a-priori, MC")
	for i in [100,1000,5000,10000,20000]:
		print(i)
		print("mc max")
		print(get_tour_dists(get_mc_tour, tsp, online=False, num_sims=i, use_max=True))
		print("mc mean")
		print(get_tour_dists(get_mc_tour, tsp, online=False, num_sims=i, use_max=False))'''

	'''tsp = STSP(addresses, departure_time)
	print("a-priori, MCTS")
	for i in [100,1000,5000,10000,20000]:
		print(i)
		print("mcts max")
		print(get_tour_dists(get_mcts_tour, tsp, online=False, num_sims=i, use_max=True))
		print("mcts mean")
		print(get_tour_dists(get_mcts_tour, tsp, online=False, num_sims=i, use_max=False))'''

	# real time lkh
	'''tsp = STSP(addresses, departure_time)
	print("realtime, lkh")
	for tm in ['best_guess', 'pessimistic']:
		print(tm)
		print("online lkh")
		print(get_tour_dists(
			get_dynamic_lkh_tour, tsp, online=True, lkh=True, traffic_model=tm))'''

	# real time MC / MCTS
	'''tsp = STSP(addresses, departure_time)
	print("realtime, mc")
	for i in [100,1000,5000,10000,20000]:
		print(i)
		print("mc max")
		print(get_tour_dists(get_mc_tour, tsp, online=True, num_sims=i, use_max=True))
		print("mc mean")
		print(get_tour_dists(get_mc_tour, tsp, online=True, num_sims=i, use_max=False))'''

	'''tsp = STSP(addresses, departure_time)
	print("realtime, mcts")
	for i in [100,1000,5000,10000,20000]:
		print(i)
		print("mcts max")
		print(get_tour_dists(get_mcts_tour, tsp, online=True, num_sims=i, use_max=True))
		print("mcts mean")
		print(get_tour_dists(get_mcts_tour, tsp, online=True, num_sims=i, use_max=False))'''

	# real time MC-LKH
	'''tsp = STSP(addresses, departure_time)
	print("realtime, mc-lkh")
	for i in [500,1000,2000]:
		print(i)
		print("mc lkh max")
		print(get_tour_dists(get_mc_lkh_tour, tsp, online=True, num_sims=i, use_max=True))
		print("mc lkh mean")
		print(get_tour_dists(get_mc_lkh_tour, tsp, online=True, num_sims=i, use_max=False))'''


	'''from matplotlib import pyplot as plt
	for i,j,k in zip(iterations,mcts_tour_dists,mcts_tour_dists_min):
		plt.hist(j, 
			normed=True, cumulative=True, 
			label='Mean-MCTS (' + str(i) + ' rollouts)', histtype='step', bins = niter,
			alpha=0.8)
		plt.hist(k, 
			normed=True, cumulative=True, 
			label='Min-MCTS (' + str(i) + ' rollouts)', histtype='step', bins = niter,
			alpha=0.8)
	plt.legend(loc=4)
	plt.show()'''

if __name__=='__main__':
	#build_test_times()
	#test_tour_building()
	test_mcts()