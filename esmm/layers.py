from tensorflow import keras


class MLP(keras.layers.Layer):
    def __init__(self, units, last_activation):
        super().__init__()
        self.mlp = [keras.layers.Dense(unit, activation='relu') for unit in units[:-1]]
        self.mlp.append(keras.layers.Dense(units[-1], activation=last_activation))

    def call(self, inputs, **kwargs):
        for fc in self.mlp:
            inputs = fc(inputs)
        return inputs
