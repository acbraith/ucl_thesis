import pandas as pd 
import numpy as np
import scipy.stats, logging, scipy.optimize, itertools
from matplotlib import pyplot as plt
from datetime import datetime, timedelta
from concurrent.futures import ProcessPoolExecutor
from multiprocessing.pool import ThreadPool
from persistentdict import persistent_memoize
from prettytable import PrettyTable
from sklearn import manifold
import networkx as nx

from gmaps_queries import get_distance_matrix, get_distance
from solve_tsp import get_shortest_tour, get_shortest_tour_dynamic_tsp, \
	check_tour_length, check_real_tour_length, \
	solve_static_deterministic_tsp, solve_deterministic_tsp, solve_static_tsp, solve_tsp, \
	get_tour_length_distribution

from gp_active_learning import get_distance_matrix_gps, \
	read_entry_gp_distance_matrix, read_gp_distance_matrix, sample_distance_matrix
from gp_active_learning import get_distance_matrix as gp_get_distance_matrix
from gp_active_learning import get_distance_matrix_element as gp_get_distance_matrix_element

def visualise_distance_matrix(dm):
	# make distance matrix symmetrical (simple averaging edge lengths)


	# build edges
	edges = []
	for i,row in enumerate(dm):
		for j,col in enumerate(row):
			print(i,j,col)
			edges += [[i,j,col]]

	# use T-SNE embedding to position nodes
	tsne = manifold.TSNE(n_components=2, metric='precomputed')
	results = tsne.fit(dm)
	coords = results.embedding_
	pos = {}
	for i,v in enumerate(coords):
		pos[i] = np.array(v)

	# draw graph
	G = nx.DiGraph()
	max_weight = np.max(np.array(edges)[:,2])
	for edge in edges:
		print(edge[2])
		print((1-edge[2]/max_weight)**5)
		nx.draw_networkx_edges(G, pos, 
			edgelist=[edge],
			alpha=(1-edge[2]/max_weight)**5,
			width=1,
			#cmap=plt.get_cmap('Blues'),
			edge_color='b')
	nx.draw_networkx_nodes(G, pos, 
		nodelist=list(range(len(dm))), 
		node_size = 100, node_color='r', linewidths=10)
	labels = {}
	for i in range(len(dm)):
		labels[i] = i
	nx.draw_networkx_labels(G,pos,labels,font_size=14)
	plt.show()

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
	addresses = list(set(engineer_day_jobs['start_address'])) + \
		list(engineer_day_jobs['job_address'])
	#return addresses
	# use already-trained GPs (get_dm.cache.OLD file)
	return sorted(set(addresses))

def forwards_approx_num_destinations_tours(origin, destinations, departure_time, **kwargs):
	'''
	Plot a graph of tour length vs number of destinations visited, and save 
	html files containing Google Maps for each of the tours with each number
	of destinations.
	(Not related to project)
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

@persistent_memoize('ANALYSIS_distance_periodicity')
def ANALYSIS_distance_periodicity(origin, destination, departure_time, final_departure_time, time_delta):

	def get_times(traffic_model):
		distances = []
		times = []
		# use multiple threads to send requests
		current_time = departure_time
		while current_time < final_departure_time:
			times += [current_time]
			current_time += time_delta
		pool = ThreadPool(processes=50)
		# pass origin, destination and kwargs to get_distance
		partial_get_distance = lambda t: \
			get_distance(origin, destination, departure_time=t, traffic_model=traffic_model)
		distances = pool.map(partial_get_distance, times)
		return (times, distances)

	times, o = get_times('optimistic')
	times, b = get_times('best_guess')
	times, p = get_times('pessimistic')
	return (times, o,b,p)

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
		minimum = get_distance(origin,destination,departure_time=datetime(2018,1,8,5,0,0),traffic_model='optimistic')
		return (minimum,optimistic,best_guess,pessimistic,duration)
	return None

@persistent_memoize('load_taxi_data')
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

def ANALYSIS_taxi_data_transform_normality():
	# load data
	results = load_taxi_data('nyc_taxi/yellow_tripdata_2016-01.csv')

	# process results
	results = [r for r in results if r is not None]

	minimum,optimistic,best_guess,pessimistic,duration = zip(*results)
	minimum = np.array(minimum)
	optimistic = np.array(optimistic)
	best_guess = np.array(best_guess)
	pessimistic = np.array(pessimistic)
	duration = np.array(duration)

	percentile_o = 100*(optimistic > duration).sum() / len(results)
	percentile_b = 100*(best_guess > duration).sum() / len(results)
	percentile_p = 100*(pessimistic > duration).sum() / len(results)
	print('Percentiles:')
	print('\tOptimistic:',percentile_o,'%')
	print('\tBest Guess:',percentile_b,'%')
	print('\tPessimistic:',percentile_p,'%')
	# These look like quartiles (when shifted such that best_guess is at median)
	# assuming google isn't trying to manipulate people by not having best_guess = median
	# however it looks like google is overestimating in general, which I think may drive customers away
	# or maybe they want people to be pleasantly surprised at the end of their trips?

	# VISUALISING
	# scale all to a standard scale
	# divide all by best_guess
	pltsize=1
	fig, axarr = plt.subplots(3,4)
	fig.set_size_inches(6*pltsize,4*pltsize)

	table = pd.DataFrame(columns=['Title','alpha','beta','gamma',
		'Gmaps Skew',
		'Duration Skew','Duration Kurtosis',
		'Duration Normal Test p-value','Duration Jarque Bera p-value'])

	def transform_data(duration,optimistic,best_guess,pessimistic,f):
		# initial transform
		d = f(duration);o = f(optimistic);b = f(best_guess);p = f(pessimistic)
		# remove nan and inf
		idx = np.unique(np.concatenate((
			np.where(~np.isfinite(d)),np.where(~np.isfinite(o)),
			np.where(~np.isfinite(b)),np.where(~np.isfinite(p))),axis=1))
		d = np.delete(d,idx);o = np.delete(o,idx);b = np.delete(b,idx);p = np.delete(p,idx)

		# fit normal distributions to d
		def mu_sigma_error(arg):
			beta,gamma = arg
			mu = beta * b
			sigma = gamma * (p - o)
			error_o = scipy.stats.norm.cdf(o, mu, sigma) - percentile_o/100
			error_b = scipy.stats.norm.cdf(b, mu, sigma) - percentile_b/100
			error_p = scipy.stats.norm.cdf(p, mu, sigma) - percentile_p/100
			res = (error_o**2 + error_b**2 + error_p**2)
			res = res[np.isfinite(res)]
			return np.sum(res)
		res = scipy.optimize.basinhopping(mu_sigma_error, x0=[1,1/1.35], 
			minimizer_kwargs={'tol':0.1, 'bounds':[(0.5,2),(0.01,10)]})
		beta = res.x[0]
		gamma = res.x[1]
		mu = beta * b
		sigma = gamma * (p-o)

		# shift data to 0 mean, unit variance
		d = (d-mu) / sigma;o = (o-mu) / sigma;b = (b-mu) / sigma;p = (p-mu) / sigma

		# remove nan, inf and outliers
		idx = np.unique(np.concatenate((
			np.where(~np.isfinite(d)),np.where(~np.isfinite(o)),
			np.where(~np.isfinite(b)),np.where(~np.isfinite(p))),axis=1))
		d = np.delete(d,idx);o = np.delete(o,idx);b = np.delete(b,idx);p = np.delete(p,idx)
		idx = np.unique(np.concatenate((
			outliers_idx(d),outliers_idx(o),
			outliers_idx(b),outliers_idx(p)),axis=1))
		d = np.delete(d,idx);o = np.delete(o,idx);b = np.delete(b,idx);p = np.delete(p,idx)

		return (d,o,b,p,beta,gamma)

	def get_test_results(d,o,b,p):
		# test skewness of transformed pessimistic,best_guess,optimistic
		obp = np.stack([o,b,p])
		gmap_skew = scipy.stats.skew(obp.flatten())
		skew_stat,gmap_skewtest_p_val = scipy.stats.skewtest(obp.flatten())

		duration_skew = scipy.stats.skew(d)
		duration_kurtosis = scipy.stats.kurtosis(d)

		# test normality of transformed data
		normaltest_z_val,normaltest_p_val = scipy.stats.normaltest(d)
		jb_val,jarque_bera_p_val = scipy.stats.jarque_bera(d)

		return (skew_stat,gmap_skew, duration_skew,duration_kurtosis,
			normaltest_z_val,normaltest_p_val, jb_val,jarque_bera_p_val)

	def plot_hist(i,j,f,title,alpha):

		print(title)

		d,o,b,p,beta,gamma = \
			transform_data(duration,optimistic,best_guess,pessimistic,f)

		_, gmap_skew, duration_skew, duration_kurtosis, _, normaltest_p_val, _, jarque_bera_p_val = \
			get_test_results(d,o,b,p)

		# add row to table
		nonlocal table
		table = table.append(
			pd.Series([title,alpha,beta,gamma,
				gmap_skew,
				duration_skew, duration_kurtosis,
				normaltest_p_val,jarque_bera_p_val],
				index=['Title','alpha','beta','gamma',
				'Gmaps Skew',
				'Duration Skew','Duration Kurtosis',
				'Duration Normal Test p-value','Duration Jarque Bera p-value']), 
			ignore_index=True)
		#print(table)

		mu, std = scipy.stats.norm.fit(d)
		x = np.linspace(np.min(d), np.max(d), 100)
		y = scipy.stats.norm.pdf(x, mu, std)

		#fig.set_title(title + ", alpha=" + str(alpha))
		fig,ax = plt.subplots()
		ax.plot(x, y, 'k', linewidth=2)
		ax.hist(d,bins=20,normed=True)
		#locs = plt.get_yticks()
		locs = [0,.5,1,1.5]
		ax.boxplot(o, sym='', whis=[5,95], vert=False, positions=[0])
		ax.boxplot(b, sym='', whis=[5,95], vert=False, positions=[0])
		ax.boxplot(p, sym='', whis=[5,95], vert=False, positions=[0])
		ax.set_yticks(locs)
		ax.set_yticklabels(locs)
		ax.set_ylim([-.1,1])
		fig.savefig('figs/' + title + ", alpha=" + str(alpha) +'.png')
		plt.close()

		#extent = axarr[i,j].get_window_extent().transformed(fig.dpi_scale_trans.inverted())
		#plt.savefig(, 
		#	bbox_inches=extent.expanded(1.1, 1.2))

	def get_alpha_p_score(alpha,f):
		d,_,_,_,_,_ = \
			transform_data(duration,optimistic,best_guess,pessimistic,f)
		jb_val,jarque_bera_p_val = scipy.stats.jarque_bera(d)
		#print("Tested",alpha,", JB:",jb_val)
		return jb_val

	no_get_alpha_p_score = \
		lambda a: get_alpha_p_score(a, lambda x: x - a*minimum)
	log_get_alpha_p_score = \
		lambda a: get_alpha_p_score(a, lambda x: np.log(x - a*minimum))
	loglog_get_alpha_p_score = \
		lambda a: get_alpha_p_score(a, lambda x: np.log(np.log(x - a*minimum)))

	# find optimal alphas for each transform
	'''
	‘Nelder-Mead’ (see here)	slowish but sensible
	‘Powell’ (see here)			cant do bounds
	‘CG’ (see here)				cant do bounds
	‘BFGS’ (see here)			cant do bounds
	‘Newton-CG’ (see here)		cant do
	‘L-BFGS-B’ (see here)		repeats itself
	‘TNC’ (see here)			
	‘COBYLA’ (see here)			
	‘SLSQP’ (see here)			
	‘dogleg’ (see here)			
	‘trust-ncg’ (see here)		
	'''
	alphas = np.linspace(0,1,21)
	print(alphas)
	no_transform_alpha = 1#alphas[np.argmin(list(map(no_get_alpha_p_score, alphas)))]
	print(no_transform_alpha)

	log_transform_alpha = .7#alphas[np.argmin(list(map(log_get_alpha_p_score, alphas)))]
	print(log_transform_alpha)

	loglog_transform_alpha = .6#alphas[np.argmin(list(map(loglog_get_alpha_p_score, alphas)))]
	print(loglog_transform_alpha)

	alpha=0.0
	plot_hist(0,0,lambda x:x - alpha*minimum,'No transform',alpha)
	plot_hist(1,0,lambda x:np.log(x - alpha*minimum),'Log transform',alpha)
	plot_hist(2,0,lambda x:np.log(np.log(x - alpha*minimum)),'LogLog transform',alpha)
	alpha=no_transform_alpha
	plot_hist(0,1,lambda x:x - alpha*minimum,'No transform',alpha)
	plot_hist(1,1,lambda x:np.log(x - alpha*minimum),'Log transform',alpha)
	plot_hist(2,1,lambda x:np.log(np.log(x - alpha*minimum)),'LogLog transform',alpha)
	alpha=log_transform_alpha
	plot_hist(0,2,lambda x:x - alpha*minimum,'No transform',alpha)
	plot_hist(1,2,lambda x:np.log(x - alpha*minimum),'Log transform',alpha)
	plot_hist(2,2,lambda x:np.log(np.log(x - alpha*minimum)),'LogLog transform',alpha)
	alpha=loglog_transform_alpha
	plot_hist(0,3,lambda x:x - alpha*minimum,'No transform',alpha)
	plot_hist(1,3,lambda x:np.log(x - alpha*minimum),'Log transform',alpha)
	plot_hist(2,3,lambda x:np.log(np.log(x - alpha*minimum)),'LogLog transform',alpha)

	print(table.to_latex(columns=['Title','alpha','beta','gamma',
				'Duration Skew','Duration Kurtosis',
				'Duration Normal Test p-value']))
	print(table)
	plt.show()
	
def ANALYSIS_traffic_no_traffic():

	data = pd.read_csv('boiler_repair.csv')
	jobs = strip_small_jobs(data, get_jobs(data))

	optimal_tour_traffic_no_traffic_ratios = []
	traffic_suboptimal_tour_losses = []
	no_traffic_suboptimal_tour_losses = []


	# compare 
	# this should be done by some multiprocessing pool
	departure_times = [
		datetime.strptime('01/01/18 03:00','%d/%m/%y %H:%M'),
		datetime.strptime('01/01/18 09:00','%d/%m/%y %H:%M'),
		datetime.strptime('01/01/18 15:00','%d/%m/%y %H:%M'),
		datetime.strptime('01/01/18 21:00','%d/%m/%y %H:%M')]
	for departure_time in departure_times:
		for i,job in enumerate(jobs):
			print('Starting job:', job)
			print('\tDeparture time:', departure_time)
			print('\t', i+1, '/', len(jobs)+1)

			addresses = get_addresses(job, data)
			distance_matrix_traffic = gp_get_distance_matrix(addresses, departure_time)
			distance_matrix_no_traffic = get_distance_matrix(addresses, traffic=False, departure_time=departure_time)

			traffic_tour, optimal_tour_length_traffic = get_shortest_tour(distance_matrix_traffic)
			no_traffic_tour, optimal_tour_length_no_traffic = get_shortest_tour(distance_matrix_no_traffic)
			suboptimal_tour_length_traffic = check_tour_length(distance_matrix_traffic, no_traffic_tour)
			suboptimal_tour_length_no_traffic = check_tour_length(distance_matrix_no_traffic, traffic_tour)

			optimal_tour_traffic_no_traffic_ratios += [optimal_tour_length_traffic / optimal_tour_length_no_traffic]
			traffic_suboptimal_tour_losses += [suboptimal_tour_length_traffic / optimal_tour_length_traffic]
			no_traffic_suboptimal_tour_losses += [suboptimal_tour_length_no_traffic / optimal_tour_length_no_traffic]

	f, axarr = plt.subplots(3)
	num_bins = 10
	axarr[0].set_title('optimal_tour_traffic_no_traffic_ratios')
	axarr[1].set_title('traffic_suboptimal_tour_losses')
	axarr[2].set_title('no_traffic_suboptimal_tour_losses')
	axarr[0].hist(optimal_tour_traffic_no_traffic_ratios, bins='auto')
	axarr[1].hist(traffic_suboptimal_tour_losses, bins='auto')
	axarr[2].hist(no_traffic_suboptimal_tour_losses, bins='auto')
	plt.show()

def do_job(data, job, departure_time):
	# used in ANALYSIS_static_dynamic
	print('Starting job:', job)
	print('Departure time:', departure_time)
	logging.info('Starting job: %s', job)
	addresses = get_addresses(job, data)
	distance_matrix = gp_get_distance_matrix(addresses, departure_time=departure_time)

	static_tour, static_tour_length = get_shortest_tour(
		distance_matrix)
	# enforce tour begins at 0
	idx = static_tour.index(0)
	static_tour = static_tour[idx:] + static_tour[:idx]

	dynamic_tour, dynamic_tour_length = get_shortest_tour_dynamic_tsp(
		addresses, departure_time=departure_time)
	
	static_suboptimial_tour_length = check_tour_length(distance_matrix, dynamic_tour)
	dynamic_suboptimial_tour_length = check_real_tour_length(addresses, static_tour, departure_time=departure_time)

	optimal_tour_static_dynamic_ratio = static_tour_length/dynamic_tour_length
	static_suboptimal_tour_loss = static_suboptimial_tour_length/static_tour_length
	dynamic_suboptimal_tour_loss = dynamic_suboptimial_tour_length/dynamic_tour_length

	print("Finished Job:", job)
	print('\toptimal_tour_static_dynamic_ratio:',optimal_tour_static_dynamic_ratio)
	print('\tstatic_suboptimal_tour_loss:',static_suboptimal_tour_loss)
	print('\tdynamic_suboptimal_tour_loss:',dynamic_suboptimal_tour_loss)

	return (optimal_tour_static_dynamic_ratio,static_suboptimal_tour_loss,dynamic_suboptimal_tour_loss)

def ANALYSIS_static_dynamic():
	data = pd.read_csv('boiler_repair.csv')
	jobs = strip_small_jobs(data, get_jobs(data))

	optimal_tour_static_dynamic_ratios = []
	static_suboptimal_tour_losses = []
	dynamic_suboptimal_tour_losses = []

	pool = ThreadPool(processes=8)
	#pool = ProcessPoolExecutor(max_workers=6)
	departure_time = datetime.strptime('01/01/18 03:00','%d/%m/%y %H:%M')
	partial_do_job = lambda j: do_job(data, j, departure_time)
	results = list(pool.map(partial_do_job, jobs))

	departure_time = datetime.strptime('01/01/18 09:00','%d/%m/%y %H:%M')
	partial_do_job = lambda j: do_job(data, j, departure_time)
	results += list(pool.map(partial_do_job, jobs))

	departure_time = datetime.strptime('01/01/18 15:00','%d/%m/%y %H:%M')
	partial_do_job = lambda j: do_job(data, j, departure_time)
	results += list(pool.map(partial_do_job, jobs))

	departure_time = datetime.strptime('01/01/18 21:00','%d/%m/%y %H:%M')
	partial_do_job = lambda j: do_job(data, j, departure_time)
	results += list(pool.map(partial_do_job, jobs))

	optimal_tour_static_dynamic_ratios,static_suboptimal_tour_losses,dynamic_suboptimal_tour_losses = \
		zip(*results)
	#print(results)
	print(optimal_tour_static_dynamic_ratios)
	print(static_suboptimal_tour_losses)
	print(dynamic_suboptimal_tour_losses)

	f, axarr = plt.subplots(3)
	num_bins = 20
	axarr[0].set_title('optimal_tour_static_dynamic_ratios')
	axarr[1].set_title('static_suboptimal_tour_losses')
	axarr[2].set_title('dynamic_suboptimal_tour_losses')
	axarr[0].hist(optimal_tour_static_dynamic_ratios,bins='auto')
	axarr[1].hist(static_suboptimal_tour_losses,bins='auto')
	axarr[2].hist(dynamic_suboptimal_tour_losses,bins='auto')
	plt.show()

if __name__ == '__main__':
	logging.basicConfig(filename='main.log',level=logging.DEBUG)

	# ANALYSIS
	# First, look for periodicity in Google travel times

	# Analyse how travel time varies over different periods of time
	if False:
		print("Distance Periodicity Analysis")
		origin = 'W2 5NA'
		destination = 'WC1E 7HG'
		title = '1day10minplot'
		departure_time = datetime.strptime('01/01/18 00:00','%d/%m/%y %H:%M')
		final_departure_time = departure_time + timedelta(days=7)
		time_delta = timedelta(minutes=10)
		times, o, b, p = ANALYSIS_distance_periodicity(
			origin, destination, departure_time, 
			final_departure_time, time_delta)
		m = min(o)
		alpha = .7; beta = .961; gamma = .659
		mu = beta * np.log(np.array(b) - alpha*m)
		sigma = gamma * (np.log(np.array(p) - alpha*m) - np.log(np.array(o) - alpha*m))

		import time
		datetimes = times
		times = [time.mktime(t.timetuple()) for t in times]

		intervals = np.linspace(1,6*12,6*12)
		print(intervals)
		mu_errors = []
		sigma_errors = []
		for interval in intervals:
			interval = int(interval)
			sigma_partial = sigma[::interval]
			mu_interp = np.interp(times, times[::interval], mu[::interval])
			sigma_interp = np.interp(times, times[::interval], sigma[::interval])
			mu_error = np.mean(np.abs(mu_interp - mu))
			sigma_error = np.mean(np.abs(sigma_interp - sigma))
			print(interval,mu_error,sigma_error)
			mu_errors += [mu_error]
			sigma_errors += [sigma_error]


		intervals = intervals / 6
		plt.plot(intervals, mu_errors, 'r', label='mu average error')
		plt.plot(intervals, sigma_errors, 'b', label='sigma average error')
		plt.plot(intervals, np.array(sigma_errors)+np.array(mu_errors), 'k', label='Total average error')
		plt.legend()
		plt.xlabel('Time interval / hours')
		plt.show()

		interval = 11
		mu_interp = np.interp(times, times[::interval], mu[::interval])
		sigma_interp = np.interp(times, times[::interval], sigma[::interval])
		fig,ax = plt.subplots()
		plt.plot(datetimes, mu, 'k', lw=1)
		plt.fill_between(datetimes, mu+sigma, mu-sigma, color='k', alpha=0.2)
		plt.plot(datetimes, mu_interp, 'b', lw=1)
		plt.fill_between(datetimes, mu_interp+sigma_interp, mu_interp-sigma_interp, color='b', alpha=0.2)
		plt.xticks(rotation=15)
		plt.show()
		#fig.savefig('figs/distance_periodicity/' + title+'.png')
		#plt.close()

	# Next, analyse how to make Google traffic models more normally distributed
	# Here we can look at the skew of optimistic and pessimistic around best_guess
	# as well as the normality of shifted and scaled real data from taxi drivers
	# We can also infer the percentiles optimistic and pessimistic are meant to represent

	# analyse taxi data + transforms of
	if False:
		print("Taxi data transforms Analysis")
		ANALYSIS_taxi_data_transform_normality()

	# Now it's time to train Gaussian Processes on the travel times between locations
	# utilising the data transforms we used before
	# First, experiment with GP active learning methods and kernel designs

	# GP train on a pair of addresses, plot graphs, compare Gmaps vs GP
	if False:
		addresses = ['W2 5NA', 'WC1E 7HG']
		# now plot a week of travel time
		def gmap_get_travel_times(traffic_model):
			times = []
			departure_time = datetime(2018,1,1)
			final_departure_time = datetime(2018,1,8)
			time_delta = timedelta(hours=2)
			while departure_time < final_departure_time:
				times += [departure_time]
				departure_time += time_delta
			partial_get_distance = lambda t: \
				get_distance_matrix(addresses,
					departure_time=t, traffic_model=traffic_model)[0,1]
			pool = ThreadPool(processes=50)
			distances = list(map(partial_get_distance, times))
			return (times, distances)

		def gp_get_travel_times(traffic_model):
			times = []
			departure_time = datetime(2018,1,1)
			final_departure_time = datetime(2018,1,8)
			time_delta = timedelta(hours=2)
			while departure_time < final_departure_time:
				times += [departure_time]
				departure_time += time_delta
			partial_get_distance = lambda t: \
				gp_get_distance_matrix_element(addresses, 0, 1, 
					departure_time=t, traffic_model=traffic_model)
			pool = ThreadPool(processes=6)
			distances = list(map(partial_get_distance, times))
			return (times, distances)
		'''print("Querying Gmaps: optimistic")
		times, o = gmap_get_travel_times('optimistic')
		print("Querying Gmaps: best_guess")
		times, b = gmap_get_travel_times('best_guess')
		print("Querying Gmaps: pessimistic")
		times, p = gmap_get_travel_times('pessimistic')
		plt.plot(times, b, 'k', lw=1)
		plt.fill_between(times, o, p, color='k', alpha=0.2)'''
		print("Querying GPs: optimistic")
		times, o = gp_get_travel_times('optimistic')
		raise Exception()
		print("Querying GPs: best_guess")
		times, b = gp_get_travel_times('best_guess')
		print("Querying GPs: pessimistic")
		times, p = gp_get_travel_times('pessimistic')
		plt.plot(times, b, 'b', lw=2)
		plt.fill_between(times, o, p, color='b', alpha=0.2)
		plt.show()

	# Next, need to build an interface to use these GPs to mimic the Google Maps get_distance_matrix
	# First, train all the GPs if needed, and cache them
	# Next, sample from the GPs at the time requested (default to datetime.now())

	# Build and read a full distance matrix of GPs
	if False:
		gps = get_distance_matrix_gps(['W2 5NA', 'WC1E 7HG'], hours=24)
		print(gps)
		print(read_entry_gp_distance_matrix(gps, 0, 1))
		dm = read_gp_distance_matrix(gps)
		
		optimistic = gp_get_distance_matrix(['W2 5NA', 'WC1E 7HG'], traffic_model='optimistic')
		best_guess = gp_get_distance_matrix(['W2 5NA', 'WC1E 7HG'], traffic_model='best_guess')
		pessimistic = gp_get_distance_matrix(['W2 5NA', 'WC1E 7HG'], traffic_model='pessimistic')
		print(optimistic)
		print(best_guess)
		print(pessimistic)

	# From now on, only use GPs

	# Test out solve_*_tsp methods
	if False:
		data = pd.read_csv('boiler_repair.csv')
		jobs = strip_small_jobs(data, get_jobs(data))
		addresses = get_addresses(jobs[0], data)
		departure_time = datetime.strptime('01/01/18 08:00','%d/%m/%y %H:%M')

		'''
		# visualise TSPs
		dmb = gp_get_distance_matrix(addresses, departure_time)
		dm1 = sample_distance_matrix(addresses, departure_time)
		dm2 = sample_distance_matrix(addresses, departure_time)
		tour_b, tour_length_bb = get_shortest_tour(dmb)
		tour_1, tour_length_11 = get_shortest_tour(dm1)
		tour_2, tour_length_22 = get_shortest_tour(dm2)
		tour_length_b1 = check_tour_length(dm1,tour_b)
		tour_length_b2 = check_tour_length(dm2,tour_b)
		tour_length_1b = check_tour_length(dmb,tour_1)
		tour_length_12 = check_tour_length(dm2,tour_1)
		tour_length_2b = check_tour_length(dmb,tour_2)
		tour_length_21 = check_tour_length(dm1,tour_2)
		print(tour_b, tour_length_bb,tour_length_b1,tour_length_b2)
		print(tour_1, tour_length_1b,tour_length_11,tour_length_12)
		print(tour_2, tour_length_2b,tour_length_21,tour_length_22)

		print(" \\\\\n".join([" & ".join(map(str,line)) for line in dmb]))
		print()
		print(" \\\\\n".join([" & ".join(map(str,line)) for line in dm1]))
		print()
		print(" \\\\\n".join([" & ".join(map(str,line)) for line in dm2]))
		visualise_distance_matrix(dm)
		raise Exception()'''
		
		tour_distribution_iter = 5

		def print_info(addresses, departure_time, tour, title):
			print("\tTour:", tour)
			tour_lengths = get_tour_length_distribution(addresses, tour, departure_time, num_iter=tour_distribution_iter)
			print("\t\tTour Lengths:", [int(t) for t in tour_lengths])
			print("\t\t\tMean:", np.mean(tour_lengths))
			print("\t\t\tStdev:", np.std(tour_lengths))
			print("\t\t\tQuartiles:", np.percentile(tour_lengths, [25.,50.,75.]))
			plt.hist(tour_lengths, normed=True, cumulative=True, 
				label=title, histtype='step', bins = tour_distribution_iter,
				alpha=0.8)

		print("Planning static, deterministic tour")
		tour = solve_static_deterministic_tsp(addresses, departure_time)
		print_info(addresses, departure_time, tour, 'Static Deterministic')

		print("Planning static, probabilistic tour")
		tour = solve_static_tsp(addresses, departure_time)
		print_info(addresses, departure_time, tour, 'Static Probabilistic')

		print("Planning dynamic, deterministic tour")
		tour = solve_deterministic_tsp(addresses, departure_time)
		print_info(addresses, departure_time, tour, 'Dynamic Deterministic')

		print("Planning dynamic, probabilistic tour")
		# this one is special; we get tour lengths not by testing one tour multiple times,
		# but by generating multiple tours on-line
		'''args = [(addresses, departure_time) for _ in range(tour_distribution_iter)]
		res = list(itertools.starmap(solve_tsp, args))
		tours = list(zip(*res))[0]
		tour_lengths = list(zip(*res))[1]
		print("\t\tTour Lengths:", [int(t) for t in tour_lengths])
		print("\t\t( Tours:", tours, ")")
		print("\t\t\tMean:", np.mean(tour_lengths))
		print("\t\t\tStdev:", np.std(tour_lengths))
		print("\t\t\tQuartiles:", np.percentile(tour_lengths, [25.,50.,75.]))
		plt.hist(tour_lengths, normed=True, cumulative=True, 
			label='Dynamic Probabilistic', histtype='step', bins = tour_distribution_iter,
			alpha=0.8)'''

		plt.legend()
		plt.show()

	if False:
		# TODO test this
		# will take a while first run; need to build all the GP Distance Matrices
		print("Traffic Analysis")
		ANALYSIS_traffic_no_traffic()
	if True:
		# TODO test this
		print("Static Dynamic comparison")
		ANALYSIS_static_dynamic()


