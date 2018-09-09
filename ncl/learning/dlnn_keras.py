from keras.layers import Acrivation, Dense
from keras.models import Sequential

from ncl import settings, utils

num_categories = len(settings.CATEGORIES)
model = Sequential()
model.add(Dense(settings.NUM_UNITS1,
                activation='relu',
                input_shape=(settings.PCA_DIMENSION,)))

model.add(Dense(settings.NUM_UNITS1, activation='relu'))
model.add(Dense(num_categories, activation='sigmoid'))
