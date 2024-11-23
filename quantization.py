import numpy as np

def quantize_regular(array, method, data_type):

	min_ = np.percentile(array, 1)
	max_ = np.percentile(array, 100-1)
	
	if method == 'quantize':

		if data_type == 'int8':
			return ((254 * ((array - min_) / (max_ - min_)) ) - 127).round()
		elif data_type == 'int4':
			return ((15 * ((array - min_) / (max_ - min_)) ) - 7).round()
		elif data_type == 'int2':
			return ((2 * ((array - min_) / (max_ - min_)) ) - 1).round()
		elif data_type == 'binary':
			return ((array - min_) / (max_ - min_) > 0.5)
		
	if method == 'dequantize':

		if data_type == 'int8':
			return ((array + 127) / 254) * (max_ - min_) + min_
		elif data_type == 'int4':
			return ((array + 7) / 15) * (max_ - min_) + min_
		elif data_type == 'int2':
			return ((array + 1) / 2) * (max_ - min_) + min_
		elif data_type == 'binary':
			return (array * (max_ - min_)) + min_


def quantize_dequantize(array, data_type):
	array_quantized = quantize_regular(array, 'quantize', data_type)
	array_dequantized = quantize_regular(array_quantized, 'dequantize', data_type)
	return array_dequantized


def feature_mapping(array, method, data_type):
	# this maps each element to the actual distribution
	hist, bins = np.histogram(array, bins=1000, range=(np.percentile(array, .5), np.percentile(array, 100-.5)))

	if method == 'density':
		hist_cumsum = hist.cumsum()
	elif method == 'linear':
		hist_cumsum = np.array([x for x in range(len(hist))])/len(array)
		
	if data_type == 'int8':
		quantization_interval = (-128, 127)
	elif data_type == 'int4':
		quantization_interval = (-8, 7)
	elif data_type == 'int2':
		quantization_interval = (-1, 1)
	elif data_type == 'binary':
		quantization_interval = (0, 1)

	# scaled_arr = np.interp(hist_cumsum, (hist_cumsum.min(), hist_cumsum.max()), (array.min(), array.max()))
	# scaled_arr_q = np.interp(hist_cumsum, (hist_cumsum.min(), hist_cumsum.max()), (-128, 127))
	scaled_arr = np.interp(hist_cumsum, (np.percentile(hist_cumsum, .5), np.percentile(hist_cumsum, 100-.5)), (np.percentile(array, .5), np.percentile(array, 100-.5)))
	scaled_arr_q = np.interp(hist_cumsum, (np.percentile(hist_cumsum, .5), np.percentile(hist_cumsum, 100-.5)), quantization_interval)
	return scaled_arr, scaled_arr_q


def quantize_feature(x, scaled_arr, scaled_arr_q):
	# map each value in the vector to the closest element in the interval
	mapped_vector = np.array(np.searchsorted(scaled_arr, x))
	# prevent reaching the maximum array index
	mapped_vector[mapped_vector >= 1000] = 999
	return scaled_arr_q[mapped_vector]


def dequantize_feature(x, scaled_arr, scaled_arr_q):
	mapped_vector = np.array(np.searchsorted(scaled_arr_q, x))
	# prevent reaching the maximum array index
	mapped_vector[mapped_vector >= 1000] = 999
	return scaled_arr[mapped_vector]


def quantize_vector(sample_vectors, quantizer_list):
	# we encode one feature at a time in a numpy array: much faster
	quantized_features = list()
	for feature_index in range(sample_vectors.shape[1]):
		scaled_arr = quantizer_list[feature_index]['scaled_arr']
		scaled_arr_q = quantizer_list[feature_index]['scaled_arr_q']
		quantized_ft = quantize_feature(sample_vectors[:, feature_index], scaled_arr, scaled_arr_q)
		quantized_features.append(quantized_ft)
	return np.array(quantized_features).T


def dequantize_vector(sample_vectors, quantizer_list):
	# we encode one feature at a time in a numpy array: much faster
	dequantized_features = list()
	for feature_index in range(sample_vectors.shape[1]):
		scaled_arr = quantizer_list[feature_index]['scaled_arr']
		scaled_arr_q = quantizer_list[feature_index]['scaled_arr_q']
		quantized_ft = dequantize_feature(sample_vectors[:, feature_index], scaled_arr, scaled_arr_q)
		dequantized_features.append(quantized_ft)
	return np.array(dequantized_features).T