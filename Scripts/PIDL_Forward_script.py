
import tensorflow as tf
import numpy as np
import scipy.io
import time
import sys

from utilities import neural_net, Navier_Stokes_2D, \
                      tf_session, mean_squared_error, relative_error

class HFM(object):

    
    def __init__(self, t_data, x_data, y_data, c_data,
                       t_eqns, x_eqns, y_eqns,
                       t_inlet, x_inlet, y_inlet, u_inlet, v_inlet,
                       t_left, x_left, y_left, u_left, v_left,
                       t_right, x_right, y_right, u_right, v_right,
                       t_bottom, x_bottom, y_bottom, u_bottom, v_bottom,
                       layers, batch_size,
                       Pec, Rey):
        
        # specs
        self.layers = layers
        self.batch_size = batch_size
        
        # flow properties
        self.Pec = Pec
        self.Rey = Rey
        
        # data
        [self.t_data, self.x_data, self.y_data, self.c_data] = [t_data, x_data, y_data, c_data]
        [self.t_eqns, self.x_eqns, self.y_eqns] = [t_eqns, x_eqns, y_eqns]
        [self.t_inlet, self.x_inlet, self.y_inlet, self.u_inlet, self.v_inlet] = [t_inlet, x_inlet, y_inlet, u_inlet, v_inlet]
        [self.t_left, self.x_left, self.y_left, self.u_left, self.v_left] = [t_left, x_left, y_left, u_left, v_left]
        [self.t_right, self.x_right, self.y_right, self.u_right, self.v_right] = [t_right, x_right, y_right, u_right, v_right]
        [self.t_bottom, self.x_bottom, self.y_bottom, self.u_bottom, self.v_bottom] = [t_bottom, x_bottom, y_bottom, u_bottom, v_bottom]
        
        # placeholders
        [self.t_data_tf, self.x_data_tf, self.y_data_tf, self.c_data_tf] = [tf.placeholder(tf.float32, shape=[None, 1]) for _ in range(4)]
        [self.t_eqns_tf, self.x_eqns_tf, self.y_eqns_tf] = [tf.placeholder(tf.float32, shape=[None, 1]) for _ in range(3)]
        [self.t_inlet_tf, self.x_inlet_tf, self.y_inlet_tf, self.u_inlet_tf, self.v_inlet_tf] = [tf.placeholder(tf.float32, shape=[None, 1]) for _ in range(5)]
        [self.t_left_tf, self.x_left_tf, self.y_left_tf, self.u_left_tf, self.v_left_tf] = [tf.placeholder(tf.float32, shape=[None, 1]) for _ in range(5)]
        [self.t_right_tf, self.x_right_tf, self.y_right_tf, self.u_right_tf, self.v_right_tf] = [tf.placeholder(tf.float32, shape=[None, 1]) for _ in range(5)]
        [self.t_bottom_tf, self.x_bottom_tf, self.y_bottom_tf, self.u_bottom_tf, self.v_bottom_tf] = [tf.placeholder(tf.float32, shape=[None, 1]) for _ in range(5)]
        
        # physics "uninformed" neural networks
        self.net_cuvp = neural_net(self.t_data, self.x_data, self.y_data, layers = self.layers)
        
        [self.c_data_pred,
         self.u_data_pred,
         self.v_data_pred,
         self.p_data_pred] = self.net_cuvp(self.t_data_tf,
                                           self.x_data_tf,
                                           self.y_data_tf)
        
        # physics "uninformed" neural networks (data at the inlet)
        [_,
         self.u_inlet_pred,
         self.v_inlet_pred,
         _] = self.net_cuvp(self.t_inlet_tf,
                            self.x_inlet_tf,
                            self.y_inlet_tf)
        
        # physics "uninformed" neural networks (data on the left boundary)
        [_,
         self.u_left_pred,
         self.v_left_pred,
         _] = self.net_cuvp(self.t_left_tf,
                            self.x_left_tf,
                            self.y_left_tf)
        
        # physics "uninformed" neural networks (data on the right boundary)
        [_,
         self.u_right_pred,
         self.v_right_pred,
         _] = self.net_cuvp(self.t_right_tf,
                            self.x_right_tf,
                            self.y_right_tf)
        
        # physics "uninformed" neural networks (data on the bottom boundary)
        [_,
         self.u_bottom_pred,
         self.v_bottom_pred,
         _] = self.net_cuvp(self.t_bottom_tf,
                            self.x_bottom_tf,
                            self.y_bottom_tf)
        
        # physics "informed" neural networks
        [self.c_eqns_pred,
         self.u_eqns_pred,
         self.v_eqns_pred,
         self.p_eqns_pred] = self.net_cuvp(self.t_eqns_tf,
                                           self.x_eqns_tf,
                                           self.y_eqns_tf)
        
        [self.e1_eqns_pred,
         self.e2_eqns_pred,
         self.e3_eqns_pred,
         self.e4_eqns_pred] = Navier_Stokes_2D(self.c_eqns_pred,
                                               self.u_eqns_pred,
                                               self.v_eqns_pred,
                                               self.p_eqns_pred,
                                               self.t_eqns_tf,
                                               self.x_eqns_tf,
                                               self.y_eqns_tf,
                                               self.Pec,
                                               self.Rey)
        

        # loss
        self.loss = mean_squared_error(self.c_data_pred, self.c_data_tf) + \
                    mean_squared_error(self.u_inlet_pred, self.u_inlet_tf) + \
                    mean_squared_error(self.v_inlet_pred, self.v_inlet_tf) + \
                    mean_squared_error(self.u_left_pred, self.u_left_tf) + \
                    mean_squared_error(self.v_left_pred, self.v_left_tf) + \
                    mean_squared_error(self.u_right_pred, self.u_right_tf) + \
                    mean_squared_error(self.v_right_pred, self.v_right_tf) + \
                    mean_squared_error(self.u_bottom_pred, self.u_bottom_tf) + \
                    mean_squared_error(self.v_bottom_pred, self.v_bottom_tf) + \
                    mean_squared_error(self.e1_eqns_pred, 0.0) + \
                    mean_squared_error(self.e2_eqns_pred, 0.0) + \
                    mean_squared_error(self.e3_eqns_pred, 0.0) + \
                    mean_squared_error(self.e4_eqns_pred, 0.0)
        
        # optimizers
        self.learning_rate = tf.placeholder(tf.float32, shape=[])
        self.optimizer = tf.train.AdamOptimizer(learning_rate = self.learning_rate)
        self.train_op = self.optimizer.minimize(self.loss)
        
        self.sess = tf_session()
    
    def train(self, total_time, learning_rate):
        
        N_data = self.t_data.shape[0]
        N_eqns = self.t_eqns.shape[0]
        
        start_time = time.time()
        running_time = 0
        it = 0
        while running_time < total_time:
            
            idx_data = np.random.choice(N_data, min(self.batch_size, N_data))
            idx_eqns = np.random.choice(N_eqns, self.batch_size)
            
            (t_data_batch,
             x_data_batch,
             y_data_batch,
             c_data_batch) = (self.t_data[idx_data,:],
                              self.x_data[idx_data,:],
                              self.y_data[idx_data,:],
                              self.c_data[idx_data,:])

            (t_eqns_batch,
             x_eqns_batch,
             y_eqns_batch) = (self.t_eqns[idx_eqns,:],
                              self.x_eqns[idx_eqns,:],
                              self.y_eqns[idx_eqns,:])


            tf_dict = {self.t_data_tf: t_data_batch,
                       self.x_data_tf: x_data_batch,
                       self.y_data_tf: y_data_batch,
                       self.c_data_tf: c_data_batch,
                       self.t_eqns_tf: t_eqns_batch,
                       self.x_eqns_tf: x_eqns_batch,
                       self.y_eqns_tf: y_eqns_batch,
                       self.t_inlet_tf: self.t_inlet,
                       self.x_inlet_tf: self.x_inlet,
                       self.y_inlet_tf: self.y_inlet,
                       self.u_inlet_tf: self.u_inlet,
                       self.v_inlet_tf: self.v_inlet,
                       self.t_left_tf: self.t_left,
                       self.x_left_tf: self.x_left,
                       self.y_left_tf: self.y_left,
                       self.u_left_tf: self.u_left,
                       self.v_left_tf: self.v_left,
                       self.t_right_tf: self.t_right,
                       self.x_right_tf: self.x_right,
                       self.y_right_tf: self.y_right,
                       self.u_right_tf: self.u_right,
                       self.v_right_tf: self.v_right,
                       self.t_bottom_tf: self.t_bottom,
                       self.x_bottom_tf: self.x_bottom,
                       self.y_bottom_tf: self.y_bottom,
                       self.u_bottom_tf: self.u_bottom,
                       self.v_bottom_tf: self.v_bottom,
                       self.learning_rate: learning_rate}
            
            self.sess.run([self.train_op], tf_dict)
            
            # Print
            if it % 10 == 0:
                elapsed = time.time() - start_time
                running_time += elapsed/3600.0
                [loss_value,
                 learning_rate_value] = self.sess.run([self.loss,
                                                       self.learning_rate], tf_dict)
                print('It: %d, Loss: %.3e, Time: %.2fs, Running Time: %.2fh, Learning Rate: %.1e'
                      %(it, loss_value, elapsed, running_time, learning_rate_value))
                sys.stdout.flush()
                start_time = time.time()
            it += 1
    
    def predict(self, t_star, x_star, y_star):
        
        tf_dict = {self.t_data_tf: t_star, self.x_data_tf: x_star, self.y_data_tf: y_star}
        
        c_star = self.sess.run(self.c_data_pred, tf_dict)
        u_star = self.sess.run(self.u_data_pred, tf_dict)
        v_star = self.sess.run(self.v_data_pred, tf_dict)
        p_star = self.sess.run(self.p_data_pred, tf_dict)
        
        return c_star, u_star, v_star, p_star
    
    
if __name__ == "__main__":
    
    batch_size = 10000
    
    layers = [3] + 10*[4*50] + [4]
    
    # Load Data
    data = scipy.io.loadmat('PIDL_Input_100x100_10TP.mat')
    
    t_star = data['t_star'] # T x 1
    x_star = data['x_star'] # N x 1
    y_star = data['y_star'] # N x 1
    
    T = t_star.shape[0]
    N = x_star.shape[0]
    
    U_star = data['U_star'] # N x T
    V_star = data['V_star'] # N x T
    P_star = data['P_star'] # N x T
    C_star = data['C_star'] # N x T
    
    # Rearrange Data 
    T_star = np.tile(t_star, (1,N)).T # N x T
    X_star = np.tile(x_star, (1,T)) # N x T
    Y_star = np.tile(y_star, (1,T)) # N x T
    
    t = T_star.flatten()[:,None] # NT x 1
    x = X_star.flatten()[:,None] # NT x 1
    y = Y_star.flatten()[:,None] # NT x 1
    u = U_star.flatten()[:,None] # NT x 1
    v = V_star.flatten()[:,None] # NT x 1
    p = P_star.flatten()[:,None] # NT x 1
    c = C_star.flatten()[:,None] # NT x 1
    
    ######################################################################
    ######################## Training Data ###############################
    ######################################################################
    
    T_data = T # int(sys.argv[1])
    N_data = N # int(sys.argv[2])
    idx_t = np.concatenate([np.array([0]), np.random.choice(T-2, T_data-2, replace=False)+1, np.array([T-1])] )
    idx_x = np.random.choice(N, N_data, replace=False)
    t_data = T_star[:, idx_t][idx_x,:].flatten()[:,None]
    x_data = X_star[:, idx_t][idx_x,:].flatten()[:,None]
    y_data = Y_star[:, idx_t][idx_x,:].flatten()[:,None]
    c_data = C_star[:, idx_t][idx_x,:].flatten()[:,None]
        
    T_eqns = T
    N_eqns = N
    idx_t = np.concatenate([np.array([0]), np.random.choice(T-2, T_eqns-2, replace=False)+1, np.array([T-1])] )
    idx_x = np.random.choice(N, N_eqns, replace=False)
    t_eqns = T_star[:, idx_t][idx_x,:].flatten()[:,None]
    x_eqns = X_star[:, idx_t][idx_x,:].flatten()[:,None]
    y_eqns = Y_star[:, idx_t][idx_x,:].flatten()[:,None]
    
    # Training Data on velocity (top)
    t_inlet = t[y == y.max()][:,None]
    x_inlet = x[y == y.max()][:,None]
    y_inlet = y[y == y.max()][:,None]
    u_inlet = u[y == y.max()][:,None]
    v_inlet = v[y == y.max()][:,None]
    
    # Training Data on velocity (left)
    t_left = t[x == x.min()][:,None]
    x_left = x[x == x.min()][:,None]
    y_left = y[x == x.min()][:,None]
    u_left = u[x == x.min()][:,None]
    v_left = v[x == x.min()][:,None]
    
    # Training Data on velocity (right)
    t_right = t[x == x.max()][:,None]
    x_right = x[x == x.max()][:,None]
    y_right = y[x == x.max()][:,None]
    u_right = u[x == x.max()][:,None]
    v_right = v[x == x.max()][:,None]
    
    # Training Data on velocity (bottom)
    t_bottom = t[y == y.min()][:,None]
    x_bottom = x[y == y.min()][:,None]
    y_bottom = y[y == y.min()][:,None]
    u_bottom = u[y == y.min()][:,None]
    v_bottom = v[y == y.min()][:,None]
    
    # Training
    model = HFM(t_data, x_data, y_data, c_data,
                t_eqns, x_eqns, y_eqns,
                t_inlet, x_inlet, y_inlet, u_inlet, v_inlet,
                t_left, x_left, y_left, u_left, v_left,
                t_right, x_right, y_right, u_right, v_right,
                t_bottom, x_bottom, y_bottom, u_bottom, v_bottom,
                layers, batch_size,
                Pec = 100, Rey = 100)
    
    model.train(total_time = 1, learning_rate=1e-3)

    # Test Data
    snap = np.array([9])    # change it every time
    t_test = T_star[:,snap]
    x_test = X_star[:,snap]
    y_test = Y_star[:,snap]
    
    c_test = C_star[:,snap]
    u_test = U_star[:,snap]
    v_test = V_star[:,snap]
    p_test = P_star[:,snap]
    
    # Prediction
    c_pred, u_pred, v_pred, p_pred = model.predict(t_test, x_test, y_test)
    
    # Error
    error_c = relative_error(c_pred, c_test)
    error_u = relative_error(u_pred, u_test)
    error_v = relative_error(v_pred, v_test)
    error_p = relative_error(p_pred - np.mean(p_pred), p_test - np.mean(p_test))

    print('Error c: %e' % (error_c))
    print('Error u: %e' % (error_u))
    print('Error v: %e' % (error_v))
    print('Error p: %e' % (error_p))
################# Save Data ###########################

C_pred = 0*C_star
U_pred = 0*U_star
V_pred = 0*V_star
P_pred = 0*P_star
for snap in range(0,t_star.shape[0]):
    t_test = T_star[:,snap:snap+1]
    x_test = X_star[:,snap:snap+1]
    y_test = Y_star[:,snap:snap+1]

    c_test = C_star[:,snap:snap+1]
    u_test = U_star[:,snap:snap+1]
    v_test = V_star[:,snap:snap+1]
    p_test = P_star[:,snap:snap+1]

    # Prediction
    c_pred, u_pred, v_pred, p_pred = model.predict(t_test, x_test, y_test)

    C_pred[:,snap:snap+1] = c_pred
    U_pred[:,snap:snap+1] = u_pred
    V_pred[:,snap:snap+1] = v_pred
    P_pred[:,snap:snap+1] = p_pred

    # Error
    error_c = relative_error(c_pred, c_test)
    error_u = relative_error(u_pred, u_test)
    error_v = relative_error(v_pred, v_test)
    error_p = relative_error(p_pred - np.mean(p_pred, axis=0, keepdims=True), p_test - np.mean(p_test, axis=0, keepdims=True))

    print('Error c: %e' % (error_c))
    print('Error u: %e' % (error_u))
    print('Error v: %e' % (error_v))
    print('Error p: %e' % (error_p))


scipy.io.savemat('PIDL_Pred_100x100_10TP_10HL200N.mat',
                 {'C_pred': C_pred, 'U_pred': U_pred, 'V_pred': V_pred, 'P_pred': P_pred})


