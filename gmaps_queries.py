import numpy as np
import googlemaps, random, threading, time, logging, math
from persistentdict import persistent_memoize, PersistentDict
from multiprocessing import Lock
from multiprocessing.pool import ThreadPool
from datetime import date, datetime, timedelta

key_list = [
	'AIzaSyB4QGLWlDDhzWwbcvwXaEYw4RWnzAnR5dc',
	'AIzaSyDKVWMVqJZmdpbtrAYpNcSls57ril2jQyk',
	'AIzaSyB8XNz5x9aCrVof0O8mr0TcOrzhFlGwYfE',
	'AIzaSyBJGPn837UCoO41-GmYJ8qsl6HELk876Yg',
	'AIzaSyCPV0Vt7B5pgu-tZhNWMK0iH1FmmamQT8k',
	'AIzaSyAddqHFPfBPyaDIlUrirh1zyMHKeZUmxvs',
	'AIzaSyB1_B-BDqYD5gKoYelh7nWZpgN08VgvJcU',
	'AIzaSyCwW5glqclKE-ScAySWAXklm1M7GsRcvOs',
	'AIzaSyBvc9mGvfPkSMCgddH8VwmsJRHFglm8aAs',
	'AIzaSyAvHGivHVew20740v2-2rx-EKU0RXti9qM',
	'AIzaSyCDP-i3yAilCDwSUc2gW9QWH1efbGOR_po',
	'AIzaSyC-cWVFSjnAlCqEm01zubqnxHE6XDiHEs4',
	\
	'AIzaSyA7qn4BOTA75G21-H0z6dvjRBW-6-u7gN8',
	'AIzaSyDTyhhcVujWh58VX9jpmLoeSDTd-UoxShc',
	'AIzaSyAsajHpyWmI3ZF4tjyfBRojJ8x4mTiKXzk',
	'AIzaSyBDAD4XbSkznbTeA8l4H7C6Jp4S_Kt9gIM',
	'AIzaSyCAHyCsPH7lL3NShMN-xJSUQm20rtqwQ28',
	'AIzaSyC4QCdST6Ds-YoIBPQi4d1iS6qxZCswQww',
	'AIzaSyDCKhVrWuh4RkBwvx4L5Rpgb9sp_FGeVck',
	'AIzaSyAf5rKKBuloMB61WVLdirYcaUthlSO3WaI',
	'AIzaSyDjTu67Pfpx7Mu0VSj2s74_dY0W0wlCA88',
	'AIzaSyC6YeE9zakNTyUKXCGhW56FSVkZU-5i0zs',
	'AIzaSyAXWyQp2frbgggNOHpVH8YULobXGoLBz_A',
	'AIzaSyAN3iIBhvm7dpA_d2hELC2-DXi134CZ45U']
KEYS = dict.fromkeys(key_list, True)

# keep track of keys where quota not reached today
valid_keys = PersistentDict('API_keys')
if 'timestamp' not in valid_keys or valid_keys['timestamp'] != date.today():
	valid_keys.clear()
	valid_keys['timestamp'] = date.today()
	valid_keys.update(**KEYS)
	valid_keys[""] = False

set_lock = threading.Lock()

@persistent_memoize('gmaps_distance_matrix')
def gmaps_distance_matrix(*args, **kwargs):
	'''

	Cached version of gmaps.distance_matrix
	Documentation at 
	https://googlemaps.github.io/google-maps-services-python/docs/2.4.6/#googlemaps.Client.distance_matrix
	On timeout, use the next API key in list

	'''
	#print(args, kwargs)
	if 'origins' not in kwargs:
		raise Exception('Invalid request. Missing the \'origins\' parameter.')
	elif 'destinations' not in kwargs:
		raise Exception('Invalid request. Missing the \'destinations\' parameter.')
	elif 'departure_time' in kwargs and 'arrival_time' in kwargs:
		raise Exception('Invalid request. Can only have one of \'departure_time\' and \'arrival_time\'.')
	elif 'traffic_mode' in kwargs and 'departure_time' not in kwargs:
		raise Exception('Invalid request. Missing the \'departure_time\' parameter.')
	elif 'transit_mode' in kwargs and ('mode' not in kwargs or kwargs['mode'] != 'transit'):
		raise Exception('Invalid request. \'transit_mode\' only permissed for mode = transit.')
	elif 'transit_routing_preference' in kwargs and ('mode' not in kwargs or kwargs['mode'] != 'transit'):
		raise Exception('Invalid request. \'transit_routing_preference\' only permissed for mode = transit.')
	query = None
	key = get_next_key("")
	while query is None:
		try:
			query = googlemaps.Client(key = key).distance_matrix(*args, **kwargs)
		except googlemaps.exceptions.Timeout:
			logging.info("Google Maps Timeout on key: %s", key)
			print("Google Maps Timeout on key:", key)
			key = get_next_key(key)
		except Exception as e:
			logging.warning("Google Maps Exception: %s", str(e))
			print("Google Maps Exception:", str(e))
			print(args)
			print(kwargs)
			query = None
			time.sleep(30)
		# check query results
		try:
			for i,row in enumerate(query['rows']):
				for j,col in enumerate(row['elements']):
					if col['status'] != 'OK':
						debug_info = 'Status not OK'
						debug_info += '\nElement: ' + str(col)
						debug_info += '\nFrom ' + str(i) + ' to ' + str(j)
						debug_info += '\nargs,kwargs: ' + str(args) + ', ' + str(kwargs)
						debug_info += '\nQuery result: ' + str(query)
						logging.info('Status not OK')
						logging.debug(debug_info)
						print(debug_info)
						raise Exception('Status not OK')
					elif 'duration' not in col:
						debug_info = 'No duration'
						debug_info += '\nElement: ' + str(col)
						debug_info += '\nFrom ' + str(i) + ' to ' + str(j)
						debug_info += '\nargs,kwargs: ' + str(args) + ', ' + str(kwargs)
						debug_info += '\nQuery result: ' + str(query)
						logging.info('No duration')
						logging.debug(debug_info)
						print(debug_info)
						raise Exception('No duration')
					elif ((('mode' in kwargs and \
						kwargs['mode'] == 'driving') \
						or 'mode' not in kwargs) and \
						'duration_in_traffic' not in col):
						debug_info = 'No duration_in_traffic'
						debug_info += '\nElement: ' + str(col)
						debug_info += '\nFrom ' + str(i) + ' to ' + str(j)
						debug_info += '\nargs,kwargs: ' + str(args) + ', ' + str(kwargs)
						debug_info += '\nQuery result: ' + str(query)
						logging.info('No duration_in_traffic')
						logging.debug(debug_info)
						print(debug_info)
						raise Exception('No duration_in_traffic')
		except Exception:
			query = None
			time.sleep(5)
	return query

def get_next_key(prev_key):
	global valid_keys
	try:
		with set_lock:
			valid_keys[prev_key] = False
			keys = [k for k in valid_keys if valid_keys[k] is True]
			return random.choice(keys)
	except Exception:
		raise Exception("Out of API keys")


@persistent_memoize('get_distance_matrix')
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

@persistent_memoize('get_week_distance_matrices')
def get_week_distance_matrices(addresses, hours=3.5, **kwargs):
	# get 1 week of travel time data from google
	# for optimistic, pessimistic and best_guess
	# X = datetimes
	x = datetime(2018,1,1)
	xend = x + timedelta(days=7)
	delta = timedelta(hours=hours)
	X = []
	# generate datetimes to query
	while x < xend:
		X += [x]
		x += delta
	partial_get_distance_matrix = \
		lambda t: get_distance_matrix(addresses, departure_time=t, **kwargs)
	pool = ThreadPool(processes=50)
	# query gmaps
	dms = np.array(pool.map(partial_get_distance_matrix, X))
	# sort in date order
	X = np.reshape([(x.timestamp()/(60*60*24))%7 for x in X], (-1,1))
	idx = np.argsort(X,axis=0)
	X = np.reshape(X[idx], (-1,1))
	dms = np.squeeze(dms[idx])

	return (X, dms)

@persistent_memoize('get_week_distances')
def get_week_distances(origins, destinations, hours=3.5, **kwargs):
	# get 1 week of travel time data from google
	# for optimistic, pessimistic and best_guess
	# X = datetimes
	x = datetime(2018,1,1)
	xend = x + timedelta(days=7)
	delta = timedelta(hours=hours)
	X = []
	while x < xend:
		X += [x]
		x += delta
	partial_get_distance = \
		lambda t: get_distance(origins, destinations, departure_time=t, **kwargs)
	pool = ThreadPool(processes=50)
	distances = pool.map(partial_get_distance, X)

	X = np.reshape([(x.timestamp()/(60*60*24))%7 for x in X], (-1,1))
	idx = np.argsort(X,axis=0)
	X = np.reshape(X[idx], (-1,1))
	distances = np.reshape(np.reshape(distances, (-1,1))[idx], (-1,1))

	return (X, distances)
