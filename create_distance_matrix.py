import pandas as pd 
import numpy as np
import subprocess, os, math, json, math, itertools, scipy.stats, logging, GPflow
from matplotlib import pyplot as plt
from datetime import datetime, timedelta
from textwrap import dedent
from gmaps_queries import gmaps_distance_matrix
from multiprocessing import Pool
from multiprocessing.pool import ThreadPool
from functools import partial
from persistentdict import memoize

from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels \
    import ExpSineSquared, WhiteKernel


def get_jobs(data):
	'''

	Return a list of all the jobs from data.

	Parameters:
		data

	Returns:
		list of jobs, each job being a tuple of (job_date, engineer_id)

	'''
	jobs = []
	job_dates = data.job_date.drop_duplicates()
	for job_date in job_dates:
		date_data = data[data.job_date == job_date]
		engineer_ids = date_data.engineer_id.drop_duplicates()
		for engineer_id in engineer_ids:
			jobs += [(job_date, engineer_id)]
	return jobs

def strip_small_jobs(data, jobs):
	'''

	Remove all jobs containing fewer than 4 addresses, as these have trivial 
	solutions.

	Parameters:
		data
		jobs: list of jobs, each job being a tuple of (job_date, engineer_id)

	Returns:
		jobs with greater than 3 addresses

	'''
	return [job for job in jobs if len(get_addresses(job, data)) > 3]

def get_addresses(job, data):
	'''
	
	Extract the start address and job addresses for a given engineer on a given date.
	
	Parameters:
		job: tuple of (job_date, engineer_id)
		data (dataframe)

	Returns:
		sorted list of job addresses and start address

	'''
	job_date, engineer_id = job
	engineer_jobs = data[data.engineer_id == engineer_id]
	engineer_day_jobs = engineer_jobs[engineer_jobs.job_date == job_date]
	addresses = list(engineer_day_jobs['job_address']) + list(engineer_day_jobs['start_address'])
	addresses = sorted(set(addresses))
	return addresses

@memoize('get_distance_matrix')
def get_distance_matrix(addresses, traffic=True, **kwargs):
	
	'''

	Build the distance matrix by querying Google Distance Matrix API
	
	Parameters:
		addresses (list of str)
		traffic (bool): default to True
		**kwargs: gmaps.distance_matrix kwargs
	
	Returns:
		numpy array of travel times between addresses, with or without traffic, at current time
	'''

	# traffic=True only valid for driving (and default to driving)
	if traffic and 'mode' in kwargs and kwargs['mode'] != 'driving':
		traffic = False
	traffic = 'duration_in_traffic' if traffic else 'duration'
	np.set_printoptions(suppress = True)
	dim = len(addresses)

	distance_matrix = np.zeros((dim,dim))

	# build in blocks of 10x10
	for x_min in range(math.ceil(dim/10)):
		x_min *= 10
		x_max = min(x_min+10, dim)
		origins = addresses[x_min:x_max]
		for y_min in range(math.ceil(dim/10)):
			y_min *= 10
			y_max = min(y_min+10, dim)
			destinations = addresses[y_min:y_max]

			# https://googlemaps.github.io/google-maps-services-python/docs/2.4.6/#googlemaps.Client.distance_matrix
			query = gmaps_distance_matrix(
				origins = origins,
				destinations = destinations,
				language='en-GB',
				**kwargs
				)

			for i, row in enumerate(query['rows']):
				for j, col in enumerate(row['elements']):
					distance_matrix[x_min+i,y_min+j] = col[traffic]['value']
	return distance_matrix

def get_distance(origin, destination, traffic=True, **kwargs):

	'''Find the distance from an origin to a destination location.
	
	Build the distance matrix by querying Google Distance Matrix API
	
	Parameters:
		origin (str)
		destination (str)
		traffic (bool): default to True
		**kwargs: gmaps.distance_matrix kwargs
	
	Returns:
		numpy array of travel times between addresses, with or without traffic, at current time

	'''

	# traffic=True only valid for driving (and default to driving)
	if traffic and 'mode' in kwargs and kwargs['mode'] != 'driving':
		traffic = False

	query = gmaps_distance_matrix(
		origins = origin,
		destinations = destination,
		language='en-GB',
		**kwargs
		)
	traffic = 'duration_in_traffic' if traffic else 'duration'
	return query['rows'][0]['elements'][0][traffic]['value']

@memoize('get_shortest_tour')
def get_shortest_tour(distance_matrix):
	'''
	
	Given a distance matrix and problem name, write the necessary files
	defining this TSP, then find a solution using LKH.
	
	Parameters:
		distance_matrix (numpy array)
	
	Returns:
		tuple of (tour_length, tour)
	
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
		# TODO: check if dimensionality < 3
		# for dimensionality = 2; return (check_tour_length([1,2]), [1,2]
		# for dimensionality = 1; return [1]
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
	write_problem_file(distance_matrix, 'temp')
	return run_lkh('temp')

@memoize('get_shortest_tour_adjust_time')
def get_shortest_tour_adjust_time(addresses, departure_time=datetime.now(), **kwargs):
	'''
	Find the shortest tour for a set of addresses. First apporximate tour time from 
	original departure time, then return results using travel times for half way
	through this original tour time.

	'''
	distance_matrix = get_distance_matrix(addresses, departure_time=departure_time, **kwargs)
	tour_length, tour = get_shortest_tour(distance_matrix)
	departure_time += timedelta(seconds=tour_length)
	distance_matrix = get_distance_matrix(addresses, departure_time=departure_time, **kwargs)
	return get_shortest_tour(distance_matrix)

@memoize('get_shortest_tour_dynamic_tsp')
def get_shortest_tour_dynamic_tsp(addresses, departure_time=datetime.now(), **kwargs):
	'''
	Brute force approach to finding shortest tour for dynamic TSP.
	O(n!); stick to smaller problems (n*n! requests).

	Parameters:
		addresses
		**kwargs: 
	
	Returns:
		tuple of (tour_length, tour)
	'''
	dim = len(addresses)
	tours = list(itertools.permutations(range(1,dim+1)))

	pool = ThreadPool(processes=8)
	partial_check_real_tour_length = partial(check_real_tour_length, **kwargs)
	args = [(addresses, tour, departure_time) for tour in tours]
	tour_lengths = pool.starmap(partial_check_real_tour_length, args)
	min_idx = np.argmin(tour_lengths)
	return (tour_lengths[min_idx], tours[min_idx])

def check_tour_length(distance_matrix, tour):
	'''
	
	Given a distance matrix and a tour, find the tour length.
	
	Parameters:
		distance_matrix (numpy array)
		tour (list of float or int)
	
	Returns:
		tour_length (float or int)
	
	'''
	tour = [x-1 for x in tour]
	tour_length = 0
	for i in range(len(tour)):
		a = tour[i]
		b = tour[(i+1) % len(tour)]
		tour_length += distance_matrix[a,b]
	return tour_length

def check_real_tour_length(addresses, tour, departure_time=datetime.now(), **kwargs):
	'''

	Given a list of addresses and a tour between them, 
	find the travel time in 'reality'

	Parameters:
		addresses (list of str)
		tour (list of int)
		**kwargs: get_distance kwargs

	Returns: 
		tour_length (float or int)

	'''
	tour = [x-1 for x in tour]
	tour_length = 0
	for i in range(len(tour)):
		a = tour[i]
		b = tour[(i+1) % len(tour)]
		travel_time = get_distance(addresses[a], addresses[b], departure_time=departure_time, **kwargs)
		tour_length += travel_time
		delta = timedelta(seconds = travel_time)
		departure_time = departure_time + delta
	return tour_length

def forwards_approx_num_destinations_tours(origin, destinations, departure_time, **kwargs):
	'''
	Plot a graph of tour length vs number of destinations visited, and save 
	html files containing Google Maps for each of the tours with each number
	of destinations.
	'''
	tour_lengths = [0]
	tours = [[1]]
	destinations = [origin] + destinations
	added_idx = [0]
	unadded_idx = list(range(len(destinations)))
	unadded_idx.remove(0) # origin must be visited
	#print(unadded_idx)

	lat_lng = []
	import gmplot
	gmap = gmplot.GoogleMapPlotter.from_geocode('London, UK')
	for addr in destinations:
		print(addr)
		lat,lng = gmap.geocode(addr)
		lat_lng += [(lat,lng)]

	# get distance matrix for all destinations
	distance_matrix = get_distance_matrix(lat_lng, departure_time, **kwargs)

	# try adding 1 destination to tour at a time
	# add destination leading to minimal increase in tour length
	for i in range(len(destinations)-1):
		temp_lengths = []
		temp_tours = []

		for idx in unadded_idx:
			#print(idx)
			small_distance_matrix = np.zeros((len(added_idx)+1, len(added_idx)+1))
			for i,idx_i in enumerate(added_idx+[idx]):
				for j,idx_j in enumerate(added_idx+[idx]):
					small_distance_matrix[i,j] = distance_matrix[idx_i,idx_j]

			tour_length, tour = get_shortest_tour(small_distance_matrix, 'temp')
			temp_lengths += [tour_length]
			temp_tours += [tour]

		# find minimal of these tours
		idx = np.argmin(temp_lengths)

		added_idx += [unadded_idx[idx]]
		tour_lengths += [temp_lengths[idx]]
		tours += [temp_tours[idx]]

		del unadded_idx[idx]

	stop_time = 30
	tour_lengths_inc_stops = [t+stop_time*i for i,t in enumerate(tour_lengths)]
	plt.plot(
		np.linspace(1,len(added_idx),num=len(added_idx)),
		np.array(tour_lengths)/60)
	plt.plot(
		np.linspace(1,len(added_idx),num=len(added_idx)),
		np.array(tour_lengths_inc_stops)/60)
	for i,(idx,tour,tour_length) in enumerate(zip(added_idx, tours, tour_lengths)):
		print("Total destinations:",i+1)
		print("\tTime  :", tour_length/60,"min")
		print("\t+stops:", (i*stop_time+tour_length)/60,"min")
		print("\tAdded :", destinations[idx])
		print("\tTour  :", tour)
		tour_addresses = [destinations[added_idx[i-1]] for i in tour]
		print("\t      :", tour_addresses)

		temp = [list(t) for t in zip(*lat_lng)]
		lats = temp[0]
		lngs = temp[1]
		tour_lats = [lats[added_idx[i-1]] for i in tour]
		tour_lngs = [lngs[added_idx[i-1]] for i in tour]
		gmap = gmplot.GoogleMapPlotter(lats[0],lngs[0],zoom=13)
		#gmap.polygon(tour_lats,tour_lngs)
		gmap.plot(tour_lats,tour_lngs)
		gmap.scatter(tour_lats,tour_lngs)
		gmap.draw(str(i)+'.html')


	print("Destinations in order added:",[destinations[i] for i in added_idx])
	plt.show()

def do_job(job):
	print('Starting job:', job)
	data = pd.read_csv('boiler_repair.csv')
	logging.info('Starting job: %s', job)
	addresses = get_addresses(job, data)
	departure_time = datetime.strptime('01/01/18 09:00','%d/%m/%y %H:%M')
	distance_matrix = get_distance_matrix(addresses, departure_time=departure_time)

	static_tour_length, static_tour = get_shortest_tour(distance_matrix)
	dynamic_tour_length, dynamic_tour = get_shortest_tour_dynamic_tsp(addresses, departure_time=departure_time)
	
	static_suboptimial_tour_length = check_tour_length(distance_matrix, dynamic_tour)
	dynamic_suboptimial_tour_length = check_real_tour_length(addresses, static_tour, departure_time=departure_time)

	optimal_tour_static_dynamic_ratio = static_tour_length/dynamic_tour_length
	static_suboptimal_tour_loss = static_suboptimial_tour_length/static_tour_length
	dynamic_suboptimal_tour_loss = dynamic_suboptimial_tour_length/dynamic_tour_length

	print("Finished Job:", job)
	print('\toptimal_tour_static_dynamic_ratio:',static_tour_length/dynamic_tour_length)
	print('\tstatic_suboptimal_tour_loss:',static_suboptimial_tour_length/static_tour_length)
	print('\tdynamic_suboptimal_tour_loss:',dynamic_suboptimial_tour_length/dynamic_tour_length)

	return (optimal_tour_static_dynamic_ratio,static_suboptimal_tour_loss,dynamic_suboptimal_tour_loss)

def ANALYSIS_static_dynamic():
	data = pd.read_csv('boiler_repair.csv')
	jobs = strip_small_jobs(data, get_jobs(data))

	optimal_tour_static_dynamic_ratios = []
	static_suboptimal_tour_losses = []
	dynamic_suboptimal_tour_losses = []

	pool = ThreadPool(processes=8)
	results = pool.map(do_job, jobs)
	optimal_tour_static_dynamic_ratios,static_suboptimal_tour_losses,dynamic_suboptimal_tour_losses = \
		zip(*results)

	'''for job in jobs:
		print(job)
		logging.info('Starting job: %s', job)
		addresses = get_addresses(job, data)
		distance_matrix = get_distance_matrix(addresses, departure_time=departure_time)

		static_tour_length, static_tour = get_shortest_tour(distance_matrix)
		dynamic_tour_length, dynamic_tour = get_shortest_tour_dynamic_tsp(addresses, departure_time=departure_time)
		
		static_suboptimial_tour_length = check_tour_length(distance_matrix, dynamic_tour)
		dynamic_suboptimial_tour_length = check_real_tour_length(addresses, static_tour, departure_time=departure_time)

		optimal_tour_static_dynamic_ratios += [static_tour_length/dynamic_tour_length]
		static_suboptimal_tour_losses += [static_suboptimial_tour_length/static_tour_length]
		dynamic_suboptimal_tour_losses += [dynamic_suboptimial_tour_length/dynamic_tour_length]
		print('optimal_tour_static_dynamic_ratio',static_tour_length/dynamic_tour_length)
		print('static_suboptimal_tour_loss',static_suboptimial_tour_length/static_tour_length)
		print('dynamic_suboptimal_tour_loss',dynamic_suboptimial_tour_length/dynamic_tour_length)
		print('***')'''

	f, axarr = plt.subplots(3)
	num_bins = 20
	axarr[0].set_title('optimal_tour_static_dynamic_ratios')
	axarr[1].set_title('static_suboptimal_tour_losses')
	axarr[2].set_title('dynamic_suboptimal_tour_losses')
	axarr[0].hist(optimal_tour_static_dynamic_ratios,bins=num_bins,normed=True)
	axarr[1].hist(static_suboptimal_tour_losses,bins=num_bins,normed=True)
	axarr[2].hist(dynamic_suboptimal_tour_losses,bins=num_bins,normed=True)
	plt.show()

def ANALYSIS_traffic_no_traffic():

	data = pd.read_csv('boiler_repair.csv')
	jobs = strip_small_jobs(data, get_jobs(data))

	optimal_tour_traffic_no_traffic_ratios = []
	traffic_suboptimal_tour_losses = []
	no_traffic_suboptimal_tour_losses = []

	# compare 
	# this should be done by some multiprocessing pool
	for job in jobs:
		logging.info('Starting job:', job)

		addresses = get_addresses(job, data)
		distance_matrix_traffic = get_distance_matrix(addresses, traffic=True, departure_time=departure_time)
		distance_matrix_no_traffic = get_distance_matrix(addresses, traffic=False, departure_time=departure_time)
		traffic_tour_length, traffic_tour = get_shortest_tour(distance_matrix_traffic)
		no_traffic_tour_length, no_traffic_tour = get_shortest_tour(distance_matrix_no_traffic)

		optimal_tour_traffic_no_traffic_ratios += [optimal_tour_length_traffic / optimal_tour_length_no_traffic]
		traffic_suboptimal_tour_losses += [suboptimal_tour_length_traffic / optimal_tour_length_traffic]
		no_traffic_suboptimal_tour_losses += [suboptimal_tour_length_no_traffic / optimal_tour_length_no_traffic]

	f, axarr = plt.subplots(3)
	num_bins = 10
	axarr[0].set_title('optimal_tour_traffic_no_traffic_ratios')
	axarr[1].set_title('traffic_suboptimal_tour_losses')
	axarr[2].set_title('no_traffic_suboptimal_tour_losses')
	axarr[0].hist(optimal_tour_traffic_no_traffic_ratios, bins=num_bins)
	axarr[1].hist(traffic_suboptimal_tour_losses, bins=num_bins)
	axarr[2].hist(no_traffic_suboptimal_tour_losses, bins=num_bins)
	plt.show()

def ANALYSIS_distance_periodicity():

	departure_time = datetime.strptime('01/01/18 00:00','%d/%m/%y %H:%M')

	origin = 'W2 5NA'
	destination = 'WC1E 7HG'
	final_departure_time = departure_time + timedelta(days=1)
	time_delta = timedelta(minutes=10)

	def plot_travel_time_variation(fig, origin, destination, departure_time, final_departure_time,
		time_delta, **kwargs):
		distances = []
		times = []
		# use multiple threads to send requests
		while departure_time < final_departure_time:
			times += [departure_time]
			departure_time += time_delta
		pool = Pool()
		# pass kwargs to get_distance
		partial_get_distance = partial(get_distance, **kwargs)
		# map over all departure_times (origin/destination constant)
		# TODO this won't work right now 
		# due to kwargs stuff (need departure_time=t, just t won't work)
		args = [(origin, destination, t) for t in times]
		distances = pool.starmap(partial_get_distance, args)
		fig.plot(times, distances)

	plot_travel_time_variation(plt, origin, destination, departure_time, final_departure_time,
		time_delta, traffic_model = 'pessimistic')
	plot_travel_time_variation(plt, origin, destination, departure_time, final_departure_time,
		time_delta, traffic_model = 'best_guess')
	plot_travel_time_variation(plt, origin, destination, departure_time, final_departure_time,
		time_delta, traffic_model = 'optimistic')
	plot_travel_time_variation(plt, origin, destination, departure_time, final_departure_time,
		time_delta, traffic = False)
	plt.show()

def process_row(row):
	print(row[0])
	row = row[1]
	origin = row['origin']
	destination = row['destination']
	duration = row['duration']
	departure_time = row['tpep_pickup_datetime']
	if duration != 0 and origin != destination and origin != (0,0) and destination != (0,0):
		departure_time = departure_time.replace(year = departure_time.year + 2)
		optimistic = get_distance(origin,destination,departure_time=departure_time,traffic_model='optimistic')
		best_guess = get_distance(origin,destination,departure_time=departure_time,traffic_model='best_guess')
		pessimistic = get_distance(origin,destination,departure_time=departure_time,traffic_model='pessimistic')
		return (optimistic,best_guess,pessimistic,duration)
	return None

@memoize('load_taxi_data')
def load_taxi_data(filepath):
	chunksize = 10 ** 6
	results = []
	pool = ThreadPool(processes=50)
	for chunk in pd.read_csv(filepath, chunksize=chunksize):
		# we have ~11,000,000 rows total
		# this would be too many requests; in a day can only make 65,000 requests right now
		# so only do 1 every 1000 rows
		chunk = chunk.iloc[::1000,:]

		# process dates etc
		chunk['tpep_pickup_datetime'] = pd.to_datetime(
			chunk['tpep_pickup_datetime'], 
			format='%Y-%m-%d %H:%M:%S')
		chunk['tpep_dropoff_datetime'] = pd.to_datetime(
			chunk['tpep_dropoff_datetime'], 
			format='%Y-%m-%d %H:%M:%S')
		chunk['duration'] = \
			(chunk['tpep_dropoff_datetime'] - chunk['tpep_pickup_datetime']).dt.total_seconds()
		chunk['origin'] = list(zip(chunk['pickup_latitude'],chunk['pickup_longitude']))
		chunk['destination'] = list(zip(chunk['dropoff_latitude'],chunk['dropoff_longitude']))

		# send google requests
		results += pool.map(process_row, chunk.iterrows())
	return results

def outliers_idx(data, m = 5):
	d = np.abs(data - np.median(data))
	mdev = np.median(d)
	s = d/mdev if mdev else 0
	return np.where(s>=m)

def main():
	# load data
	results = load_taxi_data('nyc_taxi/yellow_tripdata_2016-01.csv')

	# process results
	results = [r for r in results if r is not None]
	print(len(results),'results')

	optimistic,best_guess,pessimistic,duration = zip(*results)
	optimistic = np.array(optimistic)
	best_guess = np.array(best_guess)
	pessimistic = np.array(pessimistic)
	duration = np.array(duration)

	percentile_o = 100*(optimistic < duration).sum() / len(results)
	percentile_b = 100*(best_guess < duration).sum() / len(results)
	percentile_p = 100*(pessimistic < duration).sum() / len(results)
	print('Percentiles:')
	print('\tOptimistic:',percentile_o,'%')
	print('\tBest Guess:',percentile_b,'%')
	print('\tPessimistic:',percentile_p,'%')

	# VISUALISING
	# scale all to a standard scale
	# divide all by best_guess
	f, axarr = plt.subplots(3)

	def plot_hist(i,f,title):
		print(title)

		# transform data
		d = (f(duration)-f(best_guess)) / (f(pessimistic)+f(optimistic))
		o = (f(optimistic)-f(best_guess)) / (f(pessimistic)+f(optimistic))
		p = (f(pessimistic)-f(best_guess)) / (f(pessimistic)+f(optimistic))

		# remove nan, inf and outliers
		idx = np.unique(np.concatenate((
			np.where(~np.isfinite(d)),
			np.where(~np.isfinite(o)),
			np.where(~np.isfinite(p))),axis=1))
		d = np.delete(d,idx)
		o = np.delete(o,idx)
		p = np.delete(p,idx)
		idx = np.unique(np.concatenate((
			outliers_idx(d),
			outliers_idx(o),
			outliers_idx(p)),axis=1))
		d = np.delete(d,idx)
		o = np.delete(o,idx)
		p = np.delete(p,idx)

		# test skewness of transformed pessimistic,best_guess,optimistic
		obp = np.stack([o,np.zeros_like(o),p])
		skew = scipy.stats.skew(obp.flatten())
		print("gmaps skew:",skew)
		stat,p_val = scipy.stats.skewtest(obp.flatten())
		print("gmaps skewtest:",stat,p_val)

		# test normality of transformed data
		z_val,p_val = scipy.stats.normaltest(d)
		print("normaltest:",z_val,p_val)
		jb_val,p_val = scipy.stats.jarque_bera(d)
		print("jarque bera:",jb_val,p_val)

		mu, std = scipy.stats.norm.fit(d)
		x = np.linspace(np.min(d), np.max(d), 100)
		y = scipy.stats.norm.pdf(x, mu, std)

		axarr[i].set_title(title + ', p=' + str(np.around(p_val,6)))
		axarr[i].plot(x, y, 'k', linewidth=2)
		axarr[i].hist(d,bins=20,normed=True)
		axarr[i].scatter([0,np.average(o),np.average(p)],[0,0,0],c='k',marker='+',s=300)

	plot_hist(0,lambda x:x,'No transform')
	plot_hist(1,lambda x:np.log(x),'Log transform')
	plot_hist(2,lambda x:np.log(np.log(x)),'LogLog transform')

	plt.show()
	

'''
get dm and get distance:
	If departure_time in kwargs and mode=driving or mode not in kwargs:
		train a gaussian process on a whole week of data
		convert departure time to somewhere inside this week
			here we're using the weekly periodicity
			and preventing issues due to m.optimize() getting slightly wrong periodicity
		query gp to get mean and stdev
		depending on traffic model (if present):
			if pessimistic: return mean-stdev*1.35/2
			if best_guess: return mean
			if optimistic: return mean+stdev*1.35/2
'''

#@memoize('get_week_distance_matrices')
def get_week_distance_matrices(origins, destinations):
	# get 1 week of travel time data from google
	# for optimistic, pessimistic and best_guess
	# X = datetimes
	x = datetime(2018,1,1)
	xend = x + timedelta(days=7)
	delta = timedelta(hours=3.5,minutes=0) #3.5
	X = []
	while x < xend:
		X += [x]
		x += delta
	partial_get_best_guess = \
		lambda t: get_distance(origins, destinations, departure_time=t, traffic_model='best_guess')
	partial_get_optimistic = \
		lambda t: get_distance(origins, destinations, departure_time=t, traffic_model='optimistic')
	partial_get_pessimistic = \
		lambda t: get_distance(origins, destinations, departure_time=t, traffic_model='pessimistic')
	pool = ThreadPool(processes=50)
	best_guess = pool.map(partial_get_best_guess, X)
	optimistic = pool.map(partial_get_optimistic, X)
	pessimistic = pool.map(partial_get_pessimistic, X)

	X = np.reshape([(x.timestamp()/(60*60*24))%7 for x in X], (-1,1))
	idx = np.argsort(X,axis=0)
	X = np.reshape(X[idx], (-1,1))
	best_guess = np.reshape(np.reshape(best_guess, (-1,1))[idx], (-1,1))
	optimistic = np.reshape(np.reshape(optimistic, (-1,1))[idx], (-1,1))
	pessimistic = np.reshape(np.reshape(pessimistic, (-1,1))[idx], (-1,1))

	return (X, optimistic, best_guess, pessimistic)

#@memoize('get_gp')
def get_gp(origin, destination):
	X, optimistic, best_guess, pessimistic = get_week_distance_matrices(origin, destination)

	f = lambda x: np.log(np.log(x))
	y_mean = np.reshape(f(best_guess), (-1,1))
	y_stdev = np.reshape((f(pessimistic) - f(optimistic)) / 1.35, (-1,1))

	k = GPflow.kernels.PeriodicKernel(1,variance=0.001,lengthscales=0.2,period=1) + \
		GPflow.kernels.PeriodicKernel(1,variance=3.5,lengthscales=1,period=7) + \
		GPflow.kernels.White(1, variance=[1])
	k = 1*ExpSineSquared(periodicity=1) + 1*ExpSineSquared(periodicity=7)

	return train_gp(X, y_mean, y_stdev, k)

def train_gp(X, y_mean, y_stdev, kernel):
	def plot(m, X, Y, stdev):
		xx = np.linspace(0, 7, 500)[:,None]

		#mean, var = m.predict_y(xx)
		mean, gp_stdev = m.predict(xx, return_std=True)

		plt.figure(figsize=(12, 6))
		plt.plot(X, Y, 'k', lw=1)
		plt.fill_between(X[:,0], Y[:,0] - stdev[:,0], Y[:,0] + stdev[:,0], color='k', alpha=0.2)
		#X_model = m.X.value
		#Y_model = m.Y.value
		X_model = m.X_train_
		Y_model = m.y_train_
		plt.plot(X_model, Y_model, 'bx', mew=2)
		plt.plot(xx, mean, 'b', lw=2)

		#plt.fill_between(xx[:,0], mean[:,0] - np.sqrt(var[:,0]), mean[:,0] + np.sqrt(var[:,0]), color='blue', alpha=0.2)
		#plt.fill_between(xx[:,0], mean - gp_stdev, mean + gp_stdev, color='blue', alpha=0.2)

	x_samples = np.append(X,X)
	y_samples = np.append(y_mean + 1*y_stdev, y_mean - 1*y_stdev)

	#m = GPflow.gpr.GPR(np.reshape(x_samples, (-1,1)), np.reshape(y_samples, (-1,1)), kern=kernel)
	#m.optimize()
	print(np.shape(y_mean),np.shape(y_stdev))
	m = GaussianProcessRegressor(kernel=kernel,alpha=y_stdev)
	m.fit(np.reshape(X, (-1,1)), np.reshape(y_mean, (-1,1)))

	if True:
		plot(m, X, y_mean, y_stdev)
		#print(m.kern)
		print(m.kernel_)
		plt.show()
	return m

if __name__ == '__main__':
	logging.basicConfig(filename='main.log',level=logging.DEBUG)
	#ANALYSIS_static_dynamic()

	# analyse taxi data + transforms of
	#main()

	# GP train on a pair of addresses
	get_gp('W2 5NA', 'WC1E 6BT')
