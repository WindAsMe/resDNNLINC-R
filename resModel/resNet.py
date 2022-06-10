"""
ResNet model for regression of Keras.
Optimal model for the paper

"Chen, D.; Hu, F.; Nian, G.; Yang, T. Deep Residual Learning for Nonlinear Regression. Entropy 2020, 22, 193."

Depth:28
Width:16
"""
from tensorflow.keras import layers, models


def identity_block(input_tensor, units):
	"""The identity block is the block that has no conv layer at shortcut.
	# Arguments
		input_tensor: input tensor
		units:output shape
	# Returns
		Output tensor for the block.
	"""
	x = layers.Dense(units)(input_tensor)
	x = layers.BatchNormalization()(x)
	x = layers.Activation('relu')(x)

	x = layers.Dense(units)(x)
	x = layers.BatchNormalization()(x)
	x = layers.Activation('relu')(x)

	x = layers.Dense(units)(x)
	x = layers.BatchNormalization()(x)

	x = layers.add([x, input_tensor])
	x = layers.Activation('relu')(x)

	return x


def dens_block(input_tensor, units):
	"""A block that has a dense layer at shortcut.
	# Arguments
		input_tensor: input tensor
		unit: output tensor shape
	# Returns
		Output tensor for the block.
	"""

	x = layers.Dense(units)(input_tensor)
	x = layers.BatchNormalization()(x)
	x = layers.Activation('relu')(x)

	x = layers.Dense(units)(x)
	x = layers.BatchNormalization()(x)
	x = layers.Activation('relu')(x)

	x = layers.Dense(units)(x)
	x = layers.BatchNormalization()(x)

	shortcut = layers.Dense(units)(input_tensor)
	shortcut = layers.BatchNormalization()(shortcut)

	x = layers.add([x, shortcut])
	x = layers.Activation('relu')(x)
	return x


def ResNet50Regression(Dim):
	"""Instantiates the ResNet50 architecture.
	# Arguments        
		input_tensor: optional Keras tensor (i.e. output of `layers.Input()`)
			to use as input for the model.        
	# Returns
		A Keras model instance.
	"""
	Res_input = layers.Input(shape=(Dim,))

	width = Dim * 1.5

	x = dens_block(Res_input, width)
	x = identity_block(x, width)
	x = identity_block(x, width)

	x = dens_block(x, width)
	x = identity_block(x, width)
	x = identity_block(x, width)
	
	x = dens_block(x, width)
	x = identity_block(x, width)
	x = identity_block(x, width)

	x = layers.BatchNormalization()(x)
	x = layers.Dense(1, activation='linear')(x)
	model = models.Model(inputs=Res_input, outputs=x)

	return model


