from keras.backend import sigmoid
def swish(x, beta = 1):
    return (x * sigmoid(beta * x))

from keras.utils.generic_utils import get_custom_objects
from keras.layers import Activation
get_custom_objects().update({'swish': Activation(swish)})
from tensorflow.keras import Model
from tensorflow.keras.layers import Dense, Flatten,Embedding,Concatenate,Dropout
from tensorflow.keras.models import clone_model 
hidden_units = (128,64,32)
stock_embedding_size = 50

cat_data = train['stock_id']


import tensorflow as tf
tf.random.set_seed(42)
from tensorflow import keras
import numpy as np
from keras import backend as K
def root_mean_squared_per_error(y_true, y_pred):
    return K.sqrt(K.mean(K.square( (y_true - y_pred)/ y_true ) ))
    
es = tf.keras.callbacks.EarlyStopping(
    monitor='val_loss', patience=10, verbose=0,
    mode='min',restore_best_weights=True)

plateau = tf.keras.callbacks.ReduceLROnPlateau(
    monitor='val_loss', factor=0.2, patience=7, verbose=0,
    mode='min')

class RMSPE_MEAN(tf.keras.metrics.Metric):

    def __init__(self, name='rsmpe_mean', **kwargs):
        super(RMSPE_MEAN, self).__init__(name=name, **kwargs)
        self.res_sum = self.add_weight(name='res_sum', initializer='zeros')
        self.tot_len = self.add_weight(name='len', initializer='zeros')
    def update_state(self, y, y_pred, sample_weight=None):
        self.res_sum.assign_add( tf.reduce_sum( tf.square( tf.divide(y-y_pred,y) ) ) )
        self.tot_len.assign_add(tf.cast(tf.size(y),tf.float32))
    def reset_states(self):
        self.res_sum.assign(0)
        self.tot_len.assign(0)

    def result(self):
      return tf.sqrt(tf.divide(self.res_sum,self.tot_len))

def recursive_eval(model,x,training_mode=False):
    features=None
    if not model.start_net:
        features,out = recursive_eval(model.parent_model,x,training_mode=training_mode)
    new_features,new_out = model(x,features,training=training_mode)
    if model.start_net:
        out = model.c0 +model.alpha* new_out
    else:
        out = out + model.alpha* new_out
    return new_features,out

def set_all_parent_models_trainability(model,option):
    if model.parent_model is not None:
        for layer in model.parent_model.layers:
            layer.trainable = option
        set_all_parent_models_trainability(model.parent_model,option)

class Base_Model(Model):
    def __init__(self,stage,alpha=1.0,c0=None,dropout_rate=0.0,parent_model=None,**kwargs):
        super(Base_Model, self).__init__( **kwargs)
        self.stage = stage
        self.start_net = stage==0
        initializer = tf.zeros_initializer()
        if not stage%2:
            initializer = tf.keras.initializers.GlorotNormal()
        

        self.embedding = Embedding(max(cat_data)+1, stock_embedding_size, 
                                              input_length=1, name='stock_embedding',embeddings_initializer=initializer)
        self.concat = Concatenate()
        self.flatten = Flatten()
        self.dropout = Dropout(dropout_rate)
        self.d0 = Dense(256, activation='swish',name='d1',kernel_initializer=initializer)
        self.d1 = Dense(128, activation='swish',name='d1',kernel_initializer=initializer)
        self.d2 = Dense(64, activation='swish',name='d2',kernel_initializer=initializer)
        self.d3 = Dense(32, activation='swish',name='d3',kernel_initializer=initializer)
        self.out_layer = Dense(1,name='out',kernel_initializer=initializer)
        self.parent_model = parent_model
        self.single_network_training = True
        self.alpha = tf.Variable(alpha,trainable=True,name='alpha')
        
        if not self.start_net:
            assert self.parent_model is not None
        self.c0 = None
        if self.start_net:
            self.c0 = tf.constant(c0,name='Const')
            

    def call(self, x,features=None):
        #print(x)
        cat,num = x[:,0],x[:,1:]
        embed = self.flatten(self.embedding(cat))
        if self.start_net:
            x = self.concat([embed,num])
        else:
            x = self.concat([embed,num,features])
        #x = self.d0(x)
        # x = self.d1(x)
        # x = self.d2(x)
        x = self.dropout(self.d2(x))
        out = self.out_layer(x)
        return x,out
    

    def train_step(self, data):
        x,y = data
        features = None
        # if not self.start_net and self.single_network_training:
        #     features,prev_out = recursive_eval(self.parent_model,x)
            
        with tf.GradientTape() as tape:
          # training=True is only needed if there are layers with different
          # behavior during training versus inference (e.g. Dropout).
            if not self.start_net:
                #if not self.single_network_training:
                features,prev_out = recursive_eval(self.parent_model,x,training_mode=True)

            _,y_pred = self(x,features=features, training=True)
            y_pred = self.alpha*y_pred
            if not self.start_net:
                y_pred =y_pred + prev_out
            if self.start_net:
                y_pred = y_pred + self.c0
                #y = y - prev_out
            loss = self.compiled_loss(y, y_pred, regularization_losses=self.losses)

        # Compute gradients
        trainable_vars = self.trainable_variables
        if self.single_network_training:
            trainable_vars = [x for x in trainable_vars if 'alpha' not in x.name]
        gradients = tape.gradient(loss, trainable_vars)
        # Update weights
        self.optimizer.apply_gradients(zip(gradients, trainable_vars))
        # Update metrics (includes the metric that tracks the loss)
        
        self.compiled_metrics.update_state(y, y_pred)
        # Return a dict mapping metric names to current value
        return {m.name: m.result() for m in self.metrics}
    def test_step(self, data):
        # Unpack the data
        x, y = data
        # Compute predictions
        features=None
        if not self.start_net:
            features,prev_out = recursive_eval(self.parent_model,x,training_mode=False)
        _,y_pred = self(x,features, training=False)
        
        y_pred = self.alpha*y_pred
        if not self.start_net:
            y_pred = y_pred + prev_out
        
        # Updates the metrics tracking the loss
        self.compiled_loss(y, y_pred, regularization_losses=self.losses)
        # Update the metrics.
        

        self.compiled_metrics.update_state(y, y_pred)
        # Return a dict mapping metric names to current value.
        # Note that it will include the loss (tracked in self.metrics).
        return {m.name: m.result() for m in self.metrics}
    
    def predict_step(self,data):
        x = data
        features = None
        if not self.start_net:
            features,prev_out = recursive_eval(self.parent_model,x,training_mode=False)
        _,y_pred = self(x,features, training=False)
        y_pred = self.alpha*y_pred
        if not self.start_net:
            y_pred = y_pred + prev_out
        return y_pred
       

    # def get_config(self):
    #     #config = super(Base_Model, self).get_config()
    #     config = {"stage": self.stage,
    #     'start_net':self.start_net,
    #     'alpha':self.alpha,
    #     'parent_model':self.parent_model,
    #     'c0':self.c0}
    #     return config
        

def copy_model(model,sample_x,sample_y):
    parent_model = None
    if model.parent_model is not None:
        parent_model = copy_model(model.parent_model,sample_x,sample_y)
    m = Base_Model(stage = model.stage,alpha = model.alpha,c0=model.c0,parent_model = parent_model)
    m.compile(optimizer=keras.optimizers.Adam(learning_rate=1e-5), loss=root_mean_squared_per_error)
    m.evaluate(sample_x,sample_y)
    m.set_weights(model.get_weights())
    return m

def boost_train(x_train,y_train,x_val,y_val,num_boost_rounds=10,lr=1e-2,boost_patience=3):
    num_boost_rounds = num_boost_rounds
    c0 = 0.0#np.mean(y_train)
    model = None
    lr = lr
    print('started training')
    start_net = True
    train_errors = []
    val_errors = []
    best_stage =-1
    best_err = np.inf
    best_model = None
    boost_patience = boost_patience
    for boost_round in range(num_boost_rounds):
        #boosting stage
        model = Base_Model(stage=boost_round,alpha=1.0,c0=c0,dropout_rate=0.0,parent_model=model)
        if c0 is not None:
            c0 = None
        model.compile(optimizer=keras.optimizers.Adam(learning_rate=lr), loss=root_mean_squared_per_error)
        model.fit(x_train, 
            y_train,               
            batch_size=1024*2,
            epochs=1000,
            validation_data=(x_val, y_val),
            callbacks=[es, plateau],
            validation_batch_size=len(y_val),
            shuffle=True,
            verbose = 0)
        gc.collect()
        print("#"*100)
        print(f'boosting stage is {boost_round}')
        print(f'training error is :{model.evaluate(x_train,y_train)}')
        print(f'validation error is :{model.evaluate(x_val,y_val)}')
        print("#"*100)
        #corrective step
        model.single_network_training = False
        #set all parent models to training
        set_all_parent_models_trainability(model,True)
        
        model.compile(optimizer=keras.optimizers.Adam(learning_rate=1e-5), loss=root_mean_squared_per_error)
        model.fit(x_train, 
            y_train,               
            batch_size=1024*2,
            epochs=1000,
            validation_data=(x_val, y_val),
            callbacks=[es, plateau],
            validation_batch_size=len(y_val),
            shuffle=True,
            verbose = 0)
        gc.collect()
        #lr/=10
        print("#"*100)
        print(f'correction stage is {boost_round}')
        train_error = model.evaluate(x_train,y_train)
        val_error = model.evaluate(x_val,y_val)
        train_errors.append(train_error)
        val_errors.append(val_error)
        
        if val_error < best_err:
            best_err = val_error
            best_stage = boost_round
            best_model = copy_model(model,x_val.iloc[:2],y_val.iloc[:2])
            # model.save('net_'+str(boost_round),save_format='tf')
            # # model.save('net',save_format='tf')
            # best_model = tf.keras.models.load_model('net_'+str(boost_round), compile=True,custom_objects={'root_mean_squared_per_error': root_mean_squared_per_error,'swish':swish})

        # print(f'training error is :{train_error}.Best training error is {best_model.evaluate(x_train,y_train)} ')
        # print(f'validation error is :{val_error}.Best validation error is {best_model.evaluate(x_val,y_val)}')
        # print("#"*100)
        #freezing all parent models 
        set_all_parent_models_trainability(model,False)
        for layer in model.layers:
            layer.trainable = False
        
        if boost_round > best_stage + boost_patience:
            break
    return best_model  

