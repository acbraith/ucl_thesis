import GPflow
import numpy as np
import matplotlib.pyplot as plt
from itertools import repeat
from scipy.optimize import minimize
from scipy.stats import norm
from datetime import datetime

from gmaps_queries import get_week_distance_matrices, get_week_distances
from persistentdict import persistent_memoize, memoize

def gaussian_kl_divergence(mu_p,sigma_p,mu_q,sigma_q):
	'''
	KL(p||q) for p~N(mu_p,sigma_p^2), q~N(mu_q,sigma_q^2)
	'''
	sigma_p = max(sigma_p, .1)
	sigma_q = max(sigma_q, .1)
	return np.log(sigma_q / sigma_p) + (sigma_p**2 + (mu_p-mu_q)**2)/(2*sigma_q**2) - 1/2

def gp_point_kl_divergence(m, x, true_mean,true_stdev):
	'''
	Return KL divergence between GP and true distribution at given point.
	'''
	pred_mean,pred_var = m.predict_y(np.reshape(x, (-1,1)))
	pred_stdev = np.sqrt(pred_var)
	return gaussian_kl_divergence(pred_mean[0,0],pred_stdev[0,0],true_mean[0],true_stdev[0])

def get_kl_divergences(m, X, y_mean, y_stdev):
	return list(map(gp_point_kl_divergence, repeat(m), X, y_mean, y_stdev))

def gp_total_kl_divergence(m, X, y_mean, y_stdev):
	'''
	Return total KL divergence summed over all points with known distributions.
	'''
	kl_divergences = get_kl_divergences(m, X, y_mean, y_stdev)
	return sum(kl_divergences)

def gp_add_point(m, point):
	'''
	Add a point to the GPs training set.
	'''
	x,y = point
	X_new = np.append(m.X.value, np.reshape(x,(-1,1)), axis=0)
	Y_new = np.append(m.Y.value, np.reshape(y,(-1,1)), axis=0)
	m = GPflow.gpr.GPR(X_new, Y_new, kern=m.kern, mean_function=m.mean_function)
	m.optimize()
	return m

def new_gp_point_kl_divergence(m, x, true_mean, true_stdev, point):
	return gp_point_kl_divergence(gp_add_point(m,point), x, true_mean, true_stdev)

def new_gp_total_kl_divergence(m, X, y_mean, y_stdev, point):
	return gp_total_kl_divergence(gp_add_point(m,point), X, y_mean, y_stdev)

def plot(m, X, Y, stdev):
    xx = np.linspace(0, 7, 500)[:,None]
    mean, var = m.predict_y(xx)
    plt.figure(figsize=(12, 6))
    plt.plot(X, Y, 'k', lw=1)
    plt.fill_between(X[:,0], Y[:,0] - stdev[:,0], Y[:,0] + stdev[:,0], color='k', alpha=0.2)
    X_model = m.X.value
    Y_model = m.Y.value
    plt.plot(X_model, Y_model, 'bx', mew=2)
    plt.plot(xx, mean, 'b', lw=2)
    plt.fill_between(xx[:,0], mean[:,0] - np.sqrt(var[:,0]), mean[:,0] + np.sqrt(var[:,0]), color='blue', alpha=0.2)
    #plt.xlim(-0.1, 1.1)


# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # 
# Front end: Gmaps DM interface
# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # 
def get_distance_matrix(addresses, 
	departure_time, traffic_model='best_guess', **kwargs):

	if traffic_model not in ['pessimistic','best_guess','optimistic']:
		raise Exception("Invalid traffic model: %s" % traffic_model)

	gp_dm = get_distance_matrix_gps(addresses, **kwargs)
	dms = read_gp_distance_matrix(gp_dm, departure_time)
	idx = 0 if traffic_model == 'optimistic' else 1 if traffic_model == 'best_guess' else 2
	return dms[idx]

def get_distance_matrix_element(addresses, i, j, 
	departure_time, traffic_model='best_guess', **kwargs):

	if traffic_model not in ['pessimistic','best_guess','optimistic']:
		raise Exception("Invalid traffic model: %s" % traffic_model)

	gp_dm_element = get_distance_matrix_gp_element(addresses, i, j, **kwargs)
	distances = read_entry_gp_distance_matrix(gp_dm_element, departure_time)
	idx = 0 if traffic_model == 'optimistic' else 1 if traffic_model == 'best_guess' else 2
	return distances[idx]

# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # 
# Front end: Sample edges from DM
# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # 
def sample_distance_matrix(addresses,
	departure_time, **kwargs):
	gp_dm = get_distance_matrix_gps(addresses, **kwargs)
	return sample_gp_distance_matrix(gp_dm, departure_time)

def sample_distance_matrix_element(addresses, i, j,
	departure_time, **kwargs):
	gp_dm_element = get_distance_matrix_gp_element(addresses, i, j, **kwargs)
	return sample_entry_gp_distance_matrix(gp_dm_element, departure_time)

# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # 
# Initial GP training
# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # 
@persistent_memoize('get_distance_matrix_gps')
def get_distance_matrix_gps(addresses, hours=2, **kwargs):
	X, optimistic = \
		get_week_distance_matrices(addresses, hours=hours, traffic_model='optimistic', **kwargs)
	X, best_guess = \
		get_week_distance_matrices(addresses, hours=hours, traffic_model='best_guess', **kwargs)
	X, pessimistic = \
		get_week_distance_matrices(addresses, hours=hours, traffic_model='pessimistic', **kwargs)

	f = lambda x: np.log(x)

	gp_dm = np.array([[None for x in range(len(addresses))] for y in range(len(addresses))])
	for i in range(len(addresses)):
		for j in range(len(addresses)):
			if i != j:
				alpha = 0.7
				beta = 1#0.961
				gamma = 1/1.35#0.659
				min_travel_time = np.min(optimistic[:,i,j]) * alpha
				optimistic[:,i,j] = optimistic[:,i,j] - min_travel_time
				best_guess[:,i,j] = best_guess[:,i,j] - min_travel_time
				pessimistic[:,i,j] = pessimistic[:,i,j] - min_travel_time

				y_mean = np.reshape(f(best_guess[:,i,j]) * beta, (-1,1))
				y_stdev = np.reshape((f(pessimistic[:,i,j]) - f(optimistic[:,i,j])) * gamma, (-1,1))

				k = GPflow.kernels.PeriodicKernel(1,variance=0.0001,lengthscales=1,period=7) + \
					GPflow.kernels.PeriodicKernel(1,variance=1,lengthscales=7,period=1) + \
					GPflow.kernels.PeriodicKernel(1,variance=0.0008,lengthscales=1,period=7) * \
					GPflow.kernels.PeriodicKernel(1,variance=0.0008,lengthscales=7,period=1)
				k.periodickernel_1.period.fixed = True
				k.periodickernel_2.period.fixed = True
				k.prod.periodickernel_1.period.fixed = True
				k.prod.periodickernel_2.period.fixed = True

				meanf = GPflow.mean_functions.Constant(0)
				meanf = GPflow.mean_functions.Zero()
				gp_dm[i,j] = (train_gp(X, y_mean, y_stdev, k, meanf), min_travel_time)
	return gp_dm

@persistent_memoize('get_distance_matrix_gp_element')
def get_distance_matrix_gp_element(addresses, i, j, **kwargs):
	gp_dm = get_distance_matrix_gps(addresses, **kwargs)
	return gp_dm[i,j]

# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # 
# Get mean/var from GPs
# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # 
#@memoize
def get_entry_gp_dm_element(gp_dm_element, departure_time):
	if not(type(gp_dm_element) is tuple or type(gp_dm_element) is list):
		return (0,0,0)
	gp = gp_dm_element[0]
	min_travel_time = gp_dm_element[1]
	x = np.reshape((departure_time.timestamp()/(60*60*24))%7, (-1,1))
	mean, var = gp.predict_y(x)
	mean = mean[0,0]; stdev = np.sqrt(var)[0,0]
	return (mean, stdev, min_travel_time)

def get_entry_gp_distance_matrix(gp_dm_element, departure_time):
	if gp_dm_element == 0:
		return (0,0,0)
	else:
		return get_entry_gp_dm_element(gp_dm_element, departure_time)

def get_gp_dm_mean_stdev(gp_dm, departure_time):
	means = np.zeros_like(gp_dm)
	stdevs = np.zeros_like(gp_dm)

	x = np.reshape((departure_time.timestamp()/(60*60*24))%7, (-1,1))

	for i in range(len(gp_dm)):
		for j in range(len(gp_dm)):
			if i != j:
				mean, var = gp_dm[i,j][0].predict_y(x)
				mean = mean[0,0]; stdev = np.sqrt(var)[0,0]
				means[i,j] = mean
				stdevs[i,j] = stdev

	return means, stdevs

# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # 
# Sample from GPs
# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # 
def sample_entry_gp_distance_matrix(gp_dm_element, departure_time):
	f_inv = lambda x: np.exp(x)
	if gp_dm_element == 0:
		return 0
	else:
		mean, stdev, min_travel_time = get_entry_gp_distance_matrix(
			gp_dm_element, departure_time)
		val = np.random.normal(loc=mean, scale=stdev)
		trip_time = f_inv(val) + min_travel_time
		return trip_time

def sample_gp_distance_matrix(gp_dm, departure_time):
	trip_durations = np.zeros_like(gp_dm, dtype=np.int32)
	for i in range(np.shape(gp_dm)[0]):
		for j in range(np.shape(gp_dm)[0]):
				if i != j:
					trip_durations[i,j] = \
						sample_entry_gp_distance_matrix(gp_dm[i,j],departure_time)
	return trip_durations

# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # 
# Get optimistic/pessimistic/best_guess from GPs
# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # 
def read_entry_gp_distance_matrix(gp_dm_element, departure_time):
	f_inv = lambda x: np.exp(x)
	if not(type(gp_dm_element) is tuple or type(gp_dm_element) is list):
		return (0,0,0)
	else:
		mean, stdev, min_travel_time = get_entry_gp_distance_matrix(
			gp_dm_element, departure_time)

		best_guess = f_inv(mean) + min_travel_time
		# abbreviate f(X-min_travel_time) to g(X), optimistic=o, best_guess=b, pessimistic=p
		# assume g(p) and g(o) symmetric about g(b)
		# => g(o) = g(b - 1.35/2 * stdev)
		#    g(p) = g(b + 1.35/2 * stdev)
		optimistic = f_inv(mean - 1.35/2*stdev) + min_travel_time
		pessimistic = f_inv(mean + 1.35/2*stdev) + min_travel_time

		return (round(optimistic),round(best_guess),round(pessimistic))

def read_gp_distance_matrix(gp_dm, departure_time):
	optimistic = np.zeros_like(gp_dm, dtype=np.int32)
	best_guess = np.zeros_like(gp_dm, dtype=np.int32)
	pessimistic = np.zeros_like(gp_dm, dtype=np.int32)
	for i in range(np.shape(gp_dm)[0]):
		for j in range(np.shape(gp_dm)[0]):
				if i != j:
					gp = gp_dm[i,j]
					o,b,p = \
						read_entry_gp_distance_matrix(gp_dm[i,j],departure_time)
					optimistic[i,j] = o
					best_guess[i,j] = b
					pessimistic[i,j] = p
	return (optimistic, best_guess, pessimistic)

# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # 
# GP Active Learning
# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # 
def train_gp(X, y_mean, y_stdev, kernel, meanf=None, printinfo=True):

	weeks = 1

	def plot(m, X, Y, stdev):
		fig,ax = plt.subplots()

		x = y = z = np.zeros((1,1))
		# plot heatmap
		'''ylim = (np.min(Y-2*stdev),np.max(Y+2*stdev))
		xlim = (np.min(X)+np.random.rand(), np.max(X)-np.random.rand())
		xlim = (np.min(X), np.max(X))
		nx = 7*6; ny=1+8*2
		# get list of xy points to plot
		xy = np.mgrid[
			xlim[0]:xlim[1]:(nx*1j), ylim[0]:ylim[1]:(ny*1j)].reshape(2,-1).T
		#print(xy.shape)
		args = [(m, X, Y, stdev, point) for point in xy]
		temp_new_gp_total_kl_divergence = lambda x: \
			new_gp_total_kl_divergence(x[0], x[1], x[2], x[3], x[4])
		z = list(map(temp_new_gp_total_kl_divergence, args))
		#print(xy)
		#print(z)
		z = np.array(z).reshape(nx,ny).T
		x,y = np.meshgrid(np.linspace(xlim[0],xlim[1],nx),
			np.linspace(ylim[0],ylim[1],ny))
		#raise Exception()
		ax.contourf(x,y,np.log(z),alpha=0.5,cmap=plt.get_cmap('hot'))'''

		# plot true
		ax.plot(X, Y, 'k', lw=1)
		ax.fill_between(X[:,0], 
			Y[:,0] - stdev[:,0], 
			Y[:,0] + stdev[:,0], color='k', alpha=0.2)

		# plot GP
		xx = np.linspace(0, 7, 500)[:,None]
		mean, var = m.predict_y(xx)
		ax.plot(xx, mean, 'b', lw=2)
		ax.fill_between(xx[:,0], 
			mean[:,0] - np.sqrt(var[:,0]), 
			mean[:,0] + np.sqrt(var[:,0]), color='blue', alpha=0.2)
		# plot training points
		ax.scatter(m.X.value, m.Y.value, c='b')

		# plot kl divergences
		kl_divergences = get_kl_divergences(m, X, Y, stdev)
		ax2 = ax.twinx()
		ax2.plot(X, kl_divergences, 'k--', alpha=.5)
		ax2.set_ylim([ax2.get_ylim()[0], ax2.get_ylim()[1]*4])

		return (x,y,z)

	# look for large kl divergence
	# if mean far off or stdev too big, add point at mean
	# if stdev too small, add points at +-stdev
	intervals = 1
	x_samples = X[::intervals]
	y_samples = y_mean[::intervals]
	for w in range(1,weeks):
		x_samples = np.append(x_samples, max(X)*w+X[w::intervals])
		y_samples = np.append(y_samples, y_mean[w::intervals])
	#x_samples = np.zeros((0,1))
	#y_samples = np.zeros((0,1))
	m = GPflow.gpr.GPR(np.reshape(x_samples, (-1,1)), np.reshape(y_samples, (-1,1)), \
		kern=kernel, mean_function=meanf)
	m.likelihood.variance = 0.1
	iters = 40#(12*7)//2
	for i in range(iters):
		m.optimize()
		gptotdiv = gp_total_kl_divergence(m, X, y_mean, y_stdev)

		if printinfo:
			argmin_point = (0,0)
			if i % 1 == 0:
				x,y,z = plot(m,X,y_mean,y_stdev)
				idx = np.unravel_index(np.argmin(z),z.shape)
				argmin_point = (x[idx],y[idx])
				int_totdiv = "NaN" if not(np.isfinite(gptotdiv)) else str(int(gptotdiv))
				plt.savefig('figs/gp_active_learning/'+\
					str(int(datetime.now().timestamp()))+\
					'-iter'+str(i)+',kl'+int_totdiv+'.png')
				plt.close()

			print('#######################################')
			print("Iteration",i)
			print("Total KL Divergence:",gptotdiv)
			print("Argmin new point:",argmin_point)

		kl_divergences = get_kl_divergences(m, X, y_mean, y_stdev)

		# choose x and minimise its local KL divergence
		'''idx = np.argmax(kl_divergences)
		idx = np.random.choice(range(len(kl_divergences)), p=kl_divergences/sum(kl_divergences))
		ys = np.linspace(
			y_mean[idx]-3*y_stdev[idx],
			y_mean[idx]+3*y_stdev[idx],
			1+8*2)
		temp_new_gp_point_kl_divergence = lambda y: \
			new_gp_point_kl_divergence(m, X[idx], y_mean[idx], y_stdev[idx], (X[idx], y))
		temp_new_gp_total_kl_divergence = lambda y: \
			new_gp_total_kl_divergence(m, X, y_mean, y_stdev, (X[idx], y))
		z = list(map(temp_new_gp_point_kl_divergence, ys))
		local_argmin_point = (X[idx], ys[np.argmin(z)])
		print("Local argmin new point:", local_argmin_point)'''

		kl_divergences = np.array(kl_divergences) ** 1
		kl_divergences /= np.sum(kl_divergences)

		# randomly select a point, weighted by squared KL divergence
		# better than choosing max KL divergence; this reduces chance of getting stuck
		idx = np.random.choice(range(len(kl_divergences)), p=kl_divergences)
		mean, var = m.predict_y(np.reshape(X[idx],(-1,1)))

		# find y using some simple rules of thumb
		mean_diff = (mean-y_mean[idx]) / y_mean[idx]
		stdev_diff = (np.sqrt(var)-y_stdev[idx]) / y_stdev[idx]
		sigma_p = np.sqrt(var); sigma_q = y_stdev[idx]
		mu_p = mean; mu_q = y_mean[idx]
		# find which of mean or stdev contributes more to KL divergence
		# then add a point to reduce the error for either of these
		# do this instead of a 1D argmin; much faster
		px = X[idx] + np.random.randint(0,weeks)*7
		if (np.log(sigma_q/sigma_p) + sigma_p**2/(2*sigma_q**2)) < ((mu_p-mu_q)**2/(2*sigma_q**2)):
			heuristic_point = (px, y_mean[idx])
		else:
			#alpha = np.logspace(np.log10(5),-1,num=iters+1)[i]
			alpha = np.linspace(2,0,num=iters+1)[i]

			if sigma_p > sigma_q:
				heuristic_point = (px, y_mean[idx])
			elif mu_p > mu_q:
				heuristic_point = (px, y_mean[idx]-alpha*y_stdev[idx])
			else:
				heuristic_point = (px, y_mean[idx]+alpha*y_stdev[idx])

		print("Heuristic new point:", heuristic_point)
		m = gp_add_point(m, heuristic_point)

	return m

'''
N = 200
X = np.random.rand(N,1)
X = np.reshape(np.linspace(0,7,N), (-1,1))
y_mean = 3+np.sin(X*(2*np.pi)/1) + 0.1*np.cos(X*(2*np.pi)/7)
y_stdev = .1*(1+0.2*np.random.rand(N,1)*y_mean)

X_samples = np.zeros((0,1))
y_samples = np.zeros((0,1))

# generate initial model
k = GPflow.kernels.PeriodicKernel(1, lengthscales=5, period=1) + \
	GPflow.kernels.PeriodicKernel(1, lengthscales=8, period=7)
k = GPflow.kernels.Matern52(1,lengthscales=1)
m = GPflow.gpr.GPR(X_samples,y_samples,kern=k)
'''

'''
Method 1: do a 2D argmin of new_gp_total_kl_divergence
	very slow
	seems to have an odd error where minimize only passes 1 argument instead of 2 in tuple
'''
'''for i in range(2):
	# find point of max kl_divergence
	idx = gp_argmax_kl_divergence(m, X, y_mean, y_stdev)
	x_new = X[idx]
	# use this as initial guess for best point
	partial_new_gp_total_kl_divergence = partial(new_gp_total_kl_divergence, m, X, y_mean, y_stdev)
	optimal_sample = minimize(partial_new_gp_total_kl_divergence, x0=(X[idx],y_mean[idx]))
	print(optimal_sample.x)
	m = gp_add_point(m, optimal_sample.x)
	plot(m, X, y_mean)
	plt.show()'''

'''
Method 2: do a 1D argmin of new_gp_total_kl_divergence for a fixed x
	faster, but seems like it may get stuck on a point sometimes due to effect of other points
'''
'''
Method 3: do a 1D argmin of new_gp_point_kl_divergence for a fixed x
	still not too fast
	seems to work better tho
	problem:
		iniiallty we always throw first point way above (or maybe below) to try and get stdev correct

'''
'''for i in range(10):
	idx = gp_argmax_kl_divergence(m, X, y_mean, y_stdev)
	print(idx)
	x_new = X[idx]
	# minimise KL divergence, but penalise unlikely points
	partial_new_gp_point_kl_divergence = \
		lambda y: new_gp_point_kl_divergence(m, X[idx], y_mean[idx], y_stdev[idx], (X[idx],y)) * \
			norm.pdf(y, y_mean[idx], y_stdev[idx])
	partial_new_gp_total_kl_divergence = \
		lambda y: new_gp_total_kl_divergence(m, X, y_mean, y_stdev, (X[idx],y)) * \
			norm.pdf(y, y_mean[idx], y_stdev[idx])
	# find y to minimize this
	optimal_y = minimize(partial_new_gp_total_kl_divergence, x0=y_mean[idx])
	m = gp_add_point(m, (x_new, optimal_y.x))
plot(m, X, y_mean)
plt.show()'''

'''
Method 4: add points iteratively, but also allow movement of previous points
'''
def get_optimal_new_point(m, X, y_mean, y_stdev):
	kl_divergences = get_kl_divergences(m, X, y_mean, y_stdev)
	idx = np.argmax(kl_divergences)
	partial_new_gp_total_kl_divergence = \
		lambda point: new_gp_total_kl_divergence(m, X, y_mean, y_stdev, point)
	optimal_sample = basinhopping(partial_new_gp_total_kl_divergence, x0=(X[idx],y_mean[idx]), niter_success=2)
	return optimal_sample.x

'''for i in range(10):
	print("previous",i, gp_total_kl_divergence(m, X, y_mean, y_stdev))
	# add a point at optimal location
	point = get_optimal_new_point(m, X, y_mean, y_stdev)
	m = gp_add_point(m, point)
	print("pseudo-optimal",i, gp_total_kl_divergence(m, X, y_mean, y_stdev))
	# optimize all points, using current points as initial guess
	def partial_new_gp_total_kl_divergence(points):
		xs = points[:len(points)//2]
		ys = points[len(points)//2:]
		return gp_total_kl_divergence(\
			GPflow.gpr.GPR(
				np.reshape(xs,(-1,1)), np.reshape(ys,(-1,1)), kern=m.kern), 
				X, y_mean, y_stdev)
	initial_guess = (m.X.value, m.Y.value)
	optimal_samples = basinhopping(partial_new_gp_total_kl_divergence, x0=initial_guess, niter_success=2)
	print(optimal_samples.x)
	X_samples = optimal_samples.x[:len(optimal_samples.x)//2]
	Y_samples = optimal_samples.x[len(optimal_samples.x)//2:]
	m = GPflow.gpr.GPR(np.reshape(X_samples,(-1,1)), np.reshape(Y_samples,(-1,1)), kern=m.kern)
	print("optimal",i, gp_total_kl_divergence(m, X, y_mean, y_stdev))'''

'''
Method 5: 2 samples per known distribution point
	At mean +/- alpha*stdev
	Need to find a good alpha
	This works
	Can we make it more efficient? Merge method 5 and 6 together...
'''
'''alpha = 1
idxs = []
for i in range(1):
	x_samples = np.reshape(np.append(X,X), (-1,1))
	y_samples = np.reshape(np.append(y_mean + y_stdev*alpha, y_mean - y_stdev*alpha), (-1,1))
	m = GPflow.gpr.GPR(x_samples, y_samples, kern=k)
	print(i)
	m.optimize()
	print(gp_total_kl_divergence(m,X,y_mean,y_stdev))
	plot(m, X, y_mean, y_stdev)
	print(m.kern)
	plt.show()'''
'''
Method 6: Sample from distribution at point with max KL divergence
'''
'''for i in range(20):
	kl_divergences = get_kl_divergences(m, X, y_mean, y_stdev)
	kl_divergences = np.array(kl_divergences) / np.sum(kl_divergences)
	for j in range(5):
		idx = np.random.choice(range(len(kl_divergences)), p=kl_divergences)
		x = X[idx]
		y = np.random.normal(y_mean[idx], y_stdev[idx])
		point = (x,y)
		m = gp_add_point(m, point)
	m.optimize()
	print(i)
	print(gp_total_kl_divergence(m,X,y_mean,y_stdev))
print(m.kern)
plot(m, X, y_mean, y_stdev)
plt.show()'''