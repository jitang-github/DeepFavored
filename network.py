# ========================================
# Codes for computational graph
# ========================================
import tensorflow as tf


def WrapperActFunc(actFuncName, Input, nodeNum=None):
    Output = ''
    if actFuncName in ['relu', 'sigmoid', 'tanh', 'softmax']:
        Output = eval('tf.keras.activations.' + actFuncName)(Input)
    if actFuncName in ['LeakyReLU']:
        Output = eval('tf.keras.layers.' + actFuncName)().call(Input)
    if actFuncName in ['PReLU']:
        ActObj = eval('tf.keras.layers.' + actFuncName)()
        ActObj.build(input_shape=(None, nodeNum))
        Output = ActObj.call(Input)
    return Output


class MLP:
    def __init__(self, featuresNum, batchNormHiddenLayer, init, hiddenLayerActFunc, outputLayerActFunc=None):
        self.featuresNum = featuresNum
        self.batchNormHiddenLayer = batchNormHiddenLayer
        self.hiddenLayerActFunc = hiddenLayerActFunc
        self.outputLayerActFunc = outputLayerActFunc
        tf.compat.v1.set_random_seed(1)
        self.initializer = ''
        if init == 'glorot_uniform':
            self.initializer = tf.compat.v1.keras.initializers.glorot_normal()

    def input_layer(self):
        layer = tf.compat.v1.placeholder("float", [None, self.featuresNum], name='X')
        return layer

    def hidden_layers(self, prelayer, prelayer_node_num, layers, layer_name):
        layer = prelayer
        lastLayerNodeNum = prelayer_node_num
        if layers in [[0], None]:
            layersNum = 0
        else:
            layersNum = len(layers)
        for idx in range(layersNum):
            node_num = layers[idx]
            bias = tf.Variable(self.initializer(shape=(1, node_num)), name=layer_name + "_b_" + str(idx),
                               dtype="float32")
            if idx == 0:
                weights = tf.Variable(self.initializer(shape=(prelayer_node_num, node_num)),
                                      name=layer_name + "_W_" + str(idx), dtype="float32")
                Input = tf.matmul(prelayer, weights) + bias
            else:
                weights = tf.Variable(self.initializer(shape=(layers[idx - 1], node_num)),
                                      name=layer_name + "_W_" + str(idx), dtype="float32")
                Input = tf.matmul(layer, weights) + bias
            layer = WrapperActFunc(self.hiddenLayerActFunc, Input, node_num)
            if self.batchNormHiddenLayer:
                layer = tf.keras.layers.BatchNormalization(center=True, scale=False)(layer)
            lastLayerNodeNum = node_num
        return layer, lastLayerNodeNum

    def output_layer(self, prelayer, prelayer_node_num, node_num, act_func, layer_name):
        bias = tf.Variable(self.initializer(shape=(1, node_num)), name=layer_name + "_outputLayer_b", dtype="float32")
        weights = tf.Variable(self.initializer(shape=(prelayer_node_num, node_num)), name=layer_name + "_outputLayer_W",
                              dtype="float32")
        Input = tf.matmul(prelayer, weights) + bias
        return WrapperActFunc(act_func, Input)

    def build(self):
        pass


class MLPtwoOut(MLP):
    """
    MLP neural network with two output branches
    """

    def __init__(self, sharedLayers, Y1_hiddenLayers, Y2_hiddenLayers, Y1_outputLayerNodeNum,
                 Y1_outputLayerActFunc, Y2_outputLayerNodeNum, Y2_outputLayerActFunc,
                 **kwargs):
        super(MLPtwoOut, self).__init__(**kwargs)
        self.sharedLayers = sharedLayers
        self.Y1_hiddenLayers = Y1_hiddenLayers
        self.Y1_outputLayerNodeNum = Y1_outputLayerNodeNum
        self.Y1_outputLayerActFunc = Y1_outputLayerActFunc
        self.Y2_hiddenLayers = Y2_hiddenLayers
        self.Y2_outputLayerNodeNum = Y2_outputLayerNodeNum
        self.Y2_outputLayerActFunc = Y2_outputLayerActFunc

    def build(self):
        inputLayer = self.input_layer()

        # shared layers
        shared_layer, lastSharedLayerNodeNum = self.hidden_layers(inputLayer, self.featuresNum, self.sharedLayers,
                                                                  'shared')

        # output branch y1
        Y1_hiddenLayer, last_Y1_hiddenLayerNodeNum = self.hidden_layers(shared_layer, lastSharedLayerNodeNum,
                                                                        self.Y1_hiddenLayers, 'Y2_hidden')
        Y1_output = self.output_layer(Y1_hiddenLayer, last_Y1_hiddenLayerNodeNum, self.Y1_outputLayerNodeNum,
                                      self.Y1_outputLayerActFunc, 'Y1'
                                      )

        # output branch y2
        Y2_hiddenLayer, last_Y2_hiddenLayerNodeNum = self.hidden_layers(shared_layer, lastSharedLayerNodeNum,
                                                                        self.Y2_hiddenLayers, 'Y2_hidden')
        Y2_output = self.output_layer(Y2_hiddenLayer, last_Y2_hiddenLayerNodeNum, self.Y2_outputLayerNodeNum,
                                      self.Y2_outputLayerActFunc, 'Y2'
                                      )

        Y1_output = tf.compat.v1.identity(Y1_output, name='Y1_pred')
        Y2_output = tf.compat.v1.identity(Y2_output, name='Y2_pred')
        return inputLayer, Y1_output, Y2_output
