from utils import feature_averages

class RelEntIRL:
    def __init__(self,expert_demos,nonoptimal_demos):
        self.expert_demos = expert_demos
        self.nonoptimal_demos = nonoptimal_demos
        self.num_features = len(self.expert_demos[0][0])
        self.weights = np.zeros((self.num_features,))
        
    def calculate_objective(self):
        '''For the partition function Z($\theta$), we just sum over all the exponents of their rewards, similar to
        the equation above equation (6) in the original paper.'''
        objective = np.dot(self.expert_feature,self.weights)
        for i in range(self.nonoptimal_demos.shape[0]):
            objective -= np.exp(np.dot(self.policy_features[i],self.weights))
        return objective
    
    def calculate_expert_feature(self):
        self.expert_feature = np.zeros_like(self.weights)
        for i in range(len(self.expert_demos)):
            self.expert_feature += feature_averages(self.expert_demos[i])
        self.expert_feature /= len(self.expert_demos)
        return self.expert_feature
    
    def train(self,step_size=1e-4,num_iters=50000,print_every=5000):
        self.calculate_expert_feature()
        self.policy_features = np.zeros((len(self.nonoptimal_demos),self.num_features))
        for i in range(len(self.nonoptimal_demos)):
            self.policy_features[i] = feature_averages(self.nonoptimal_demos[i])
            
        importance_sampling = np.zeros((len(self.nonoptimal_demos),))
        for i in range(num_iters):
            update = np.zeros_like(self.weights)
            for j in range(len(self.nonoptimal_demos)):
                importance_sampling[j] = np.exp(np.dot(self.policy_features[j],self.weights))
            importance_sampling /= np.sum(importance_sampling,axis=0)
            weighted_sum = np.sum(np.multiply(np.array([importance_sampling,]*self.policy_features.shape[1]).T,\
                                              self.policy_features),axis=0)
            self.weights += step_size*(self.expert_feature - weighted_sum)
            # One weird trick to ensure that the weights don't blow up the objective.
            self.weights = self.weights/np.linalg.norm(self.weights,keepdims=True)
            if i%print_every == 0:
                print "Value of objective is: " + str(self.calculate_objective())