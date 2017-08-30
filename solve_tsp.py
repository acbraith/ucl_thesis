import numpy as np
import subprocess, itertools, random, string, os
from datetime import datetime, timedelta
from textwrap import dedent
from multiprocessing.pool import ThreadPool
from multiprocessing import Pool
from functools import partial
from persistentdict import persistent_memoize

from gp_active_learning import get_distance_matrix, get_distance_matrix_element, \
	sample_distance_matrix_element

import cProfile
def do_cprofile(func):
    def profiled_func(*args, **kwargs):
        profile = cProfile.Profile()
        try:
            profile.enable()
            result = func(*args, **kwargs)
            profile.disable()
            return result
        finally:
            profile.print_stats(sort=2)
    return profiled_func

@persistent_memoize('get_shortest_tour')
def get_shortest_tour(distance_matrix):
	'''
	
	Given a distance matrix and problem name, write the necessary files
	defining this TSP, then find a solution using LKH.
	
	Parameters:
		distance_matrix (numpy array)
	
	Returns:
		tuple of (tour, tour_length)
	
	'''

	def write_problem_file(distance_matrix, problem_name):
		'''
		
		Writes the necessary files for LKH to solve the given TSP
		
		Paramters:
			distance_matrix
			problem_name
		
		'''
		problem_file = problem_name+'.tsp'
		param_file = problem_name+'.par'
		tour_file = problem_name+'.tour'

		with open(problem_file, 'w') as f:
			problem = dedent('''
				NAME: {}
				TYPE: ATSP
				DIMENSION: {}
				EDGE_WEIGHT_TYPE: EXPLICIT
				EDGE_WEIGHT_FORMAT: FULL_MATRIX
				EDGE_WEIGHT_SECTION\n'''.format(problem_name, str(np.shape(distance_matrix)[0])))

			np.set_printoptions(suppress = True, linewidth=1e8, threshold=1e8)
			to_remove = ['[',']','.']
			dm_string = ' ' + np.array2string(distance_matrix)
			for char in to_remove:
				dm_string = dm_string.replace(char, '')
			problem += dm_string + '\nEOF'

			f.write(problem)

		with open(param_file, 'w') as f:
			param = 'PROBLEM_FILE = ' + problem_file \
				+ '\nTOUR_FILE = ' + tour_file
			f.write(param)

	def clear_problem_files(problem_name):
		problem_file = problem_name+'.tsp'
		param_file = problem_name+'.par'
		tour_file = problem_name+'.tour'
		os.remove(problem_file)
		os.remove(param_file)
		os.remove(tour_file)

	def run_lkh(problem_name):
		'''
		
		Runs LKH on the given problem
		Expects files of <problem_name>.tsp/par to exist
		
		Parameters:
			problem_name (str)
		
		Returns:
			tuple of (tour_length, tour)
		
		'''
		problem_file = problem_name+'.tsp'
		param_file = problem_name+'.par'
		tour_file = problem_name+'.tour'
		LKH = '/home/alex/Downloads/LKH-2.0.7/LKH'
		subprocess.run([LKH, param_file], stdout=subprocess.PIPE)

		tour_length = None
		tour = []
		tour_started = False
		with open(tour_file) as f:
			for line in f:
				if tour_started:
					if line.startswith('-1'):
						tour_started = False
					else:
						tour += [int(line[:-1])]
				if line.startswith('TOUR_SECTION'):
					tour_started = True
				elif line.startswith('COMMENT : Length = '):
					tour_length = int(''.join([i for i in line if i.isdigit()]))
		return (tour_length, tour)

	if distance_matrix.shape[0] < 4:
		dimension = distance_matrix.shape[0]
		# trivial solution
		tour = []
		if dimension == 1: tour = [1]
		elif dimension == 2: tour = [1,2]
		elif dimension == 3: tour = [1,2,3]
		tour_length = check_tour_length(distance_matrix, tour)
		return (tour_length, tour)

	def id_generator(size=6, chars=string.ascii_uppercase + string.digits):
		return ''.join(random.choice(chars) for _ in range(size))

	prob_id = "tsps/" + str(int(datetime.now().timestamp())) + id_generator()
	write_problem_file(distance_matrix, prob_id)
	tour_length, tour = run_lkh(prob_id)
	clear_problem_files(prob_id)
	tour = [x-1 for x in tour]
	return (tour, tour_length)

@persistent_memoize('get_shortest_tour_dynamic_tsp')
def get_shortest_tour_dynamic_tsp(addresses, departure_time=datetime.now(), **kwargs):
	'''
	Brute force approach to finding shortest tour for dynamic TSP.
	O(n!); stick to smaller problems (n*n! requests).

	Parameters:
		addresses
		**kwargs: 
	
	Returns:
		tuple of (tour, tour_length)
	'''
	dim = len(addresses)
	tours = list(itertools.permutations(range(1,dim)))
	tours = [[0]+list(t) for t in tours]

	pool = ThreadPool(processes=8)
	partial_check_real_tour_length = partial(check_real_tour_length, **kwargs)
	args = [(addresses, tour, departure_time) for tour in tours]
	tour_lengths = pool.starmap(partial_check_real_tour_length, args)
	min_idx = np.argmin(tour_lengths)
	return (tours[min_idx], tour_lengths[min_idx])

def check_tour_length(distance_matrix, tour):
	'''
	
	Given a distance matrix and a tour, find the tour length.
	
	Parameters:
		distance_matrix (numpy array)
		tour (list of float or int)
	
	Returns:
		tour_length (float or int)
	
	'''
	tour_length = 0
	for i in range(len(tour)):
		a = tour[i]
		b = tour[(i+1) % len(tour)]
		tour_length += distance_matrix[a,b]
	return tour_length

def solve_static_deterministic_tsp(addresses, departure_time, 
	wait_time=timedelta(hours=1), **kwargs):
	# Assume no probabilistic or temporal effects
	# Get distance matrix, then use LKH
	dm = get_distance_matrix(addresses, departure_time=departure_time, **kwargs)
	tour, tour_length = get_shortest_tour(dm)
	# enforce tour begins at 0
	idx = tour.index(0)
	tour = tour[idx:] + tour[:idx]
	return tour

def solve_deterministic_tsp(addresses, departure_time, 
	wait_time=timedelta(hours=1), **kwargs):
	# Assume no probabilistic effects
	# on-line LKH
	# Use LKH to plan next step
	# we do this by setting distances of edges already taken to 0
	current_time = departure_time
	dm = get_distance_matrix(addresses, departure_time=current_time, **kwargs)
	tour, tour_length = get_shortest_tour(dm)

	# enforce tour begins at 0
	idx = tour.index(0)
	tour = tour[idx:] + tour[:idx]

	final_tour = tour[0:2]
	# construct tour 1 step at a time
	while len(final_tour) < len(addresses):
		# get distance matrix for current time
		trip_length = dm[final_tour[-2],final_tour[-1]]
		current_time += timedelta(seconds=int(trip_length)) + wait_time
		dm = get_distance_matrix(addresses, departure_time=current_time, **kwargs)
		# encourage following previous steps
		for i,j in zip(final_tour[:-1], final_tour[1:]):
			dm[i,j] = 0
		# get next step of tour
		tour, tour_length = get_shortest_tour(dm)
		final_tour += [tour[(tour.index(final_tour[-1])+1) % len(tour)]]

	return final_tour

#@do_cprofile
def solve_static_tsp(addresses, departure_time, 
	wait_time=timedelta(hours=1), **kwargs):

	def get_rollout_tour_length(addresses, departure_time, partial_tour):
		rollout_remaining_actions = [a for a in list(range(len(addresses))) if a not in partial_tour]
		rollout_tour = partial_tour[:]
		rollout_tour_length = 0
		#print("\t\t\tRemaining Actions:", rollout_remaining_actions)
		#print("\t\t\tRollout Tour:", rollout_tour)

		while len(rollout_remaining_actions) > 0:
			# Random rollout policy
			rollout_action = np.random.choice(rollout_remaining_actions)
			#print("\t\t\tRollout Action:", rollout_action)
			rollout_tour += [rollout_action]
			#print("\t\t\tRollout Tour:", rollout_tour)
			rollout_tour_length += sample_distance_matrix_element(
				addresses, rollout_tour[-2], rollout_action, 
				departure_time, **kwargs)
			#print("\t\t\tRollout Tour Length:", rollout_tour_length)
			rollout_remaining_actions = [a for a in rollout_remaining_actions if a != rollout_action]
			#print("\t\t\tRemaining Actions:", rollout_remaining_actions)
		# return to start
		rollout_tour_length += sample_distance_matrix_element(
			addresses, rollout_tour[-1], rollout_tour[0], 
			departure_time, **kwargs)
		#print("\t\t\tFinal Rollout Tour Length:", rollout_tour_length)
		return rollout_tour_length

	# Assume no temporal effects
	# Monte Carlo rollouts from original DM
	# done on-line
	tour = [0]
	for i in range(len(addresses)-1):
		actions = [a for a in list(range(len(addresses))) if a not in tour]

		#print("CHOOSING ACTION", i)
		#print("\tTour:", tour)
		#print("\tLength:", tour_length)

		tour_lengths = []
		# EVALUATE each action by performing rollouts after taking that action
		for j,action in enumerate(actions):

			# append action to tour
			action_tour_lengths = []
			partial_tour = tour + [action]

			# perform rollouts to terminal state
			num_rollouts = 3
			args = [(addresses, departure_time, partial_tour) for _ in range(num_rollouts)]
			action_tour_lengths = list(itertools.starmap(get_rollout_tour_length, args))

			tour_lengths += [action_tour_lengths]

		# SELECT action to minimise minimum tour lengths
		min_tour_lengths = [np.min(np.array(atl)) for atl in tour_lengths]
		action = actions[np.argmin(min_tour_lengths)]
		tour += [action]
		#print("\tMean Tour Lengths:", min_tour_lengths)
		#print("\tSelected Action:", action)
	return tour

def solve_tsp(addresses, departure_time, 
	wait_time=timedelta(hours=1), **kwargs):

	def get_rollout_tour_length(addresses, current_time, partial_tour):
		rollout_remaining_actions = [a for a in list(range(len(addresses))) if a not in partial_tour]
		rollout_tour = partial_tour[:]
		rollout_tour_length = 0
		if len(rollout_tour) > 1:
			edge_length = sample_distance_matrix_element(
				addresses, rollout_tour[-2], rollout_tour[-1], 
				current_time, **kwargs)
			rollout_tour_length += edge_length
			current_time = current_time + timedelta(seconds=int(edge_length)) + wait_time

		#print("\t\tRemaining Actions:", rollout_remaining_actions)
		#print("\t\tRollout Tour:", rollout_tour)

		while len(rollout_remaining_actions) > 0:
			# Random rollout policy
			#print("\t\tRollout Current Time:", current_time)
			rollout_action = np.random.choice(rollout_remaining_actions)
			#print("\t\t\tRollout Action:", rollout_action)
			rollout_tour += [rollout_action]
			#print("\t\t\tRollout Tour:", rollout_tour)
			edge_length = sample_distance_matrix_element(
				addresses, rollout_tour[-2], rollout_tour[-1], 
				current_time, **kwargs)
			rollout_tour_length += edge_length
			current_time = current_time + timedelta(seconds=int(edge_length)) + wait_time
			#print("\t\t\tRollout Tour Length:", rollout_tour_length)
			rollout_remaining_actions = [a for a in rollout_remaining_actions if a != rollout_action]
			#print("\t\t\tRemaining Actions:", rollout_remaining_actions)
		# return to start
		rollout_tour_length += sample_distance_matrix_element(
			addresses, rollout_tour[-1], rollout_tour[0], 
			current_time, **kwargs)
		#print("\t\t\tFinal Rollout Tour Length:", rollout_tour_length)
		return rollout_tour_length
	# Probabilistic and temporal effects
	# Monte Carlo rollouts, updating current time each iteration
	tour = [0]
	tour_length = 0
	current_time = departure_time
	for i in range(len(addresses)-1):
		actions = [a for a in list(range(len(addresses))) if a not in tour]

		print("CHOOSING ACTION", i)
		print("\tTour:", tour)
		print("\tLength:", tour_length)

		tour_lengths = []
		# EVALUATE each action by performing rollouts after taking that action
		for j,action in enumerate(actions):

			# append action to tour
			action_tour_lengths = []
			partial_tour = tour + [action]

			# perform rollouts to terminal state
			num_rollouts = 20
			args = [(addresses, current_time, partial_tour) for _ in range(num_rollouts)]
			action_tour_lengths = list(itertools.starmap(get_rollout_tour_length, args))

			tour_lengths += [action_tour_lengths]

		# SELECT action to minimise minimum tour lengths
		min_tour_lengths = [np.min(np.array(atl)) for atl in tour_lengths]
		action = actions[np.argmin(min_tour_lengths)]
		tour += [action]
		# keep track of real time
		edge_length = sample_distance_matrix_element(
			addresses, tour[-2], tour[-1], 
			current_time, **kwargs)
		tour_length += edge_length
		current_time += timedelta(seconds=edge_length) + wait_time
		print("\tMin Tour Lengths:", min_tour_lengths)
		print("\tSelected Action:", action)
		print("\tCurrent Time:", current_time)

	# return home
	edge_length = sample_distance_matrix_element(
		addresses, tour[-1], tour[0], 
		current_time, **kwargs)
	tour_length += edge_length
	current_time += timedelta(seconds=edge_length) + wait_time

	return (tour, tour_length)

def check_real_tour_length(addresses, tour, departure_time=datetime.now(), 
	wait_time=timedelta(hours=1), **kwargs):
	'''

	Given a list of addresses and a tour between them, 
	find the travel time assuming no probabilistic effects.

	Parameters:
		addresses (list of str)
		tour (list of int)
		**kwargs: get_distance kwargs

	Returns: 
		tour_length (float or int)

	'''
	tour_length = 0
	for i in range(len(tour)):
		a = tour[i]
		b = tour[(i+1) % len(tour)]
		travel_time = get_distance_matrix_element(addresses, a, b, 
			departure_time=departure_time, **kwargs)
		tour_length += travel_time
		delta = timedelta(seconds = travel_time)
		departure_time = departure_time + delta + wait_time
	return tour_length

def get_tour_length_distribution(addresses, tour, departure_time, 
	wait_time=timedelta(hours=1), num_iter=30, **kwargs):
	'''

	Given a list of addresses and a tour between them, 
	find the travel time assuming no probabilistic effects.

	Parameters:
		addresses (list of str)
		tour (list of int)
		**kwargs: get_distance kwargs

	Returns: 
		tour_lengths (list of (float or int))

	'''
	tour_lengths = []
	args = [(addresses, tour, departure_time, wait_time) for i in range(num_iter)]
	tour_lengths = list(itertools.starmap(get_tour_length, args))
	return tour_lengths

def get_tour_length(addresses, tour, departure_time, 
	wait_time=timedelta(hours=1)):
	
	tour_length = 0
	current_time = departure_time
	for i in range(len(tour)):
		a = tour[i]
		b = tour[(i+1) % len(tour)]
		travel_time = sample_distance_matrix_element(addresses, a, b, current_time)
		tour_length += travel_time
		delta = timedelta(seconds = travel_time)
		current_time = current_time + delta + wait_time

	return tour_length
