import numpy as np




def sigmoid_normalize(data_arr, given_scaler = None, scaled = True):
	if scaled == True:

		# This is implementing 1 / (1 + e^((- 4/col_max) * x)) where scale is a scaler to prevent convergence 
		# to 0 or 1 for output values.
		data_normed = np.copy(data_arr)
		if given_scaler != None:
			scale = given_scaler
		else:
			scale = (4. / (data_arr.max()))

		data_normed = (1 / (1 + np.exp( - scale * data_arr))) 

		return data_normed, scale

	else:
		return (1 / (1 + np.exp(- data_arr)))


def logistic_unnormalize(data_arr, scale_term, scaled = True):

	data_unnormed = np.zeros(data_arr.shape)

	if scaled == True:
			scale = scale_term
			data_unnormed = - (np.log((1 / data_arr) - 1)) / (scale) 
	else: 
		data_unnormed = - np.log((1 / data_arr) - 1) 

	return data_unnormed




def tanh_normalize(data_arr, given_scaler = None, scaled = True):
	if scaled == True:
		data_normed = np.copy(data_arr)
		if given_scaler != None:
			scale_term = given_scaler
		else:
			scale = (4.0 / (data_arr.max()))
		
		data_normed = np.tanh(scale * data_arr)
		return data_normed, scale
	else: 
		return np.tanh(data_arr)

def arctanh_unnormalize(data_arr, scale_term, scaled = True):

	data_unnormed = np.zeros(data_arr.shape)

	if scaled == True:
		scale = scale_term
		data_unnormed = (1 / scale) * np.arctanh(data_arr)
	else:
		data_unnormed = np.arctanh(data_arr)

	return data_unnormed



def uniform_range_normalize(data_arr, given_scalers = None, low = 0., high = 1.):
	data_normed = np.copy(data_arr)
	if given_scalers != None:
		mini = given_scalers[0]
		maxi = given_scalers[1]
		data_normed[:,col] = (high - low)* ( data_arr - mini) / (maxi - mini) + low
	else:
		mini = data_arr.min()
		maxi = data_arr.max()
		data_normed = (high - low)* ( data_arr - mini) / (maxi - mini) + low
	return data_normed, [mini,maxi]

def uniform_range_unnormalize(data_arr, scale_terms, low = 0.0, high = 1.0):
	data_unnormed = np.zeros(data_arr.shape)
	mini = scale_terms[0]
	maxi = scale_terms[1]

	data_unnormed = ((data_arr - low) * (maxi - mini)  / (high - low)) + mini

	return data_unnormed




def normalize(data_arr, norm_scale = 'unif'):

	#print norm_scale

	if (norm_scale == 'unif'):
		data_normed, scale = uniform_range_normalize(data_arr)
	elif (norm_scale == '-1to1'):
		data_normed, scale = uniform_range_normalize(data_arr, low=-1.,high=1.)
	elif norm_scale == 'tanh':
		data_normed, scale = tanh_normalize(data_arr)
	elif norm_scale == 'sigmoid':
		data_normed, scale = sigmoid_normalize(data_arr)
	elif norm_scale == 'none':
		data_normed = data_arr
		scale = [1,1]

	return data_normed, scale


def unnormalize(data_arr, scale, norm_scale = 'unif'):

	#print norm_scale
	if norm_scale == 'unif':
		data_unnormed = uniform_range_unnormalize(data_arr, scale)
	elif norm_scale == '-1to1':
		data_unnormed = uniform_range_unnormalize(data_arr, scale, low = -1., high =1.)
	elif norm_scale == 'tanh':
		data_unnormed = arctanh_unnormalize(data_arr, scale)
	elif norm_scale == 'sigmoid':
		data_unnormed = logistic_unnormalize(data_arr, scale)
	elif norm_scale == 'none':
		data_unnormed = data_arr

	return data_unnormed