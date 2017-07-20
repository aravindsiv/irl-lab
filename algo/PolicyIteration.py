class PolicyIteration:
    def __init__(self,env):
        self.env = env
        self.values = np.zeros((self.env.num_states,))
        self.policy = np.random.choice(self.env.actions,size=(self.env.num_states))
    
    def policy_evaluation(self,num_iters=10,gamma=0.99):
        for i in range(num_iters):
            transition_probs = np.zeros((self.env.num_states,self.env.num_states))
            for j in range(self.env.num_states):
                transition_probs[j] = self.env.get_transition_probabilities(j,self.policy[j])
            self.values = self.env.get_rewards() + gamma*np.dot(transition_probs,self.values)
    
    def policy_iteration(self,num_iters=10):
        for i in range(num_iters):
            self.policy_evaluation()
            self.policy = self.env.take_greedy_action(self.values)
        return self.policy