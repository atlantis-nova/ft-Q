# Please cite
Algorithm: **feature-level quantization (ft-Q)**<br>
Author: **Michelangiolo Mazzeschi**<br>
Published: **24th November 2024**

# implementation of feature-level quantization (ft-Q)

The notebook **feature-level-quantization.ipynb** showcases the experiment outlined in the article (*WIP).

### Usage

To use the ft-Q quantization functions, you only need numpy installed. You will just need to import the functions contained in quantization.py.

```
from quantization import *

# prepare ft-Q quantizer, this only has to run once
interval_mapping_method = 'linear' # supported: 'linear', 'density'
quantization_metod = 'binary' # supported: 'int8', 'int4', 'int2', 'binary'

quantizer_list = list()
for feature_n in range(array.shape[-1]):
	array = array[:, feature_n].copy()
	scaled_arr, scaled_arr_q = feature_mapping(array, method=interval_mapping_method, data_type=quantization_metod)
	quantizer_list.append({'scaled_arr': scaled_arr, 'scaled_arr_q': scaled_arr_q})

# quantize array, output is in float32
quantized_tfQ = quantize_vector(array, quantizer_list).round()
```

Note that the output array (quantized_tf8) is still in **float32**, it is up to the user to cast it to the correct datatype (ex. **int8**, or **bool**). This hs been done to allow for maximum flexibility.