# -*- coding: utf-8 -*-

"""
Authors: Zheng Wang
input: emp sc emp fc
outputs: sim fc and esc(fitted)
"""

### Linear Model fitting based on relatioship SC FC and Q

class LinearFit:
    def __init__(self, sc, fc):
        if sc.shape == fc.shape:
            self.sc = sc
            self.fc = fc
            self.num_node = sc.shape[0]


    def ldm_2nd_model_fitting(self, std, num_converge):
        
        # eigen decomposition on sc
        u, d, v = np.linalg.svd(self.sc)

        if np.abs(d).max()>= 1.0:
            print("unstable sc")
            print(np.abs(d).max())
        else:
            

            # design Q noise correlation
            
            q = std*np.eye(self.num_node)
            

            # transform fc and Q to sc eigen space  
            fc_t = (u.T.dot(self.fc)).dot(v.T)
            q_t = (u.T.dot(q)).dot(v.T)

            
            def myFunction(z):
                F = (1-z)**3 *np.diag(fc_t)-std*z-std
      
                return F
            zGuess = 0.2*np.ones((self.num_node,))
            for i in range(num_converge):
                
                k = fsolve(myFunction,zGuess)
                zGuess = k
            
            # based new eigenvalues k to get new sc
            self.eSC= u.dot(np.diag(k)).dot(v)
            
            
            
            # get est of fc in sc eigen space
            q_new = q_t+ 0.5*np.diag(k/(1-k)).dot(q_t)+0.5*q_t.dot(np.diag(k/(1-k)))#+ np.diag(k**2).dot(q_t)+q_t.dot(np.diag(k**2))
            
            C_t = np.zeros_like(self.sc)
            
            for i in range(self.num_node):
              for j in range(self.num_node):
                C_t[i,j] = q_new[i,j]/(1-k[i]*k[j])
                
            # convert est fc to orginal space
            self.simFC= (u.dot(C_t)).dot(v)