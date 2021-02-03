import numpy as np 
import abc

class Agent(abc.ABC):
    """
    Agent's Model 
    """
    def __init__(self, N, epsilon=0):
        # print("N: {} | eps: {} ".format(N, epsilon))
        self.epsilon = epsilon # if epsilon equals 0 => Greedy agent, Agent agent otherwise
        self.N = int(N)
        self.reset()

    def step(self):
        # Step in time (choose an action)
        prob = np.random.uniform()
        if prob < self.epsilon:
            # Exploratory move
            N = self.Q.size
            self.last_action = np.random.choice(np.array(range(N), dtype=int))
        else:
            # Eploitation move
            max_actions = np.argwhere(self.Q == np.amax(self.Q)).flatten() # greedy actions (max value)
            if len(max_actions) == 1:
                self.last_action = max_actions
            else:
                self.last_action = np.random.choice(max_actions)

        return self.last_action

    def update(self, reward):
        """
        Update Value function depending on reward obtained
        """
        self.r_sum[self.last_action] += reward
        self.action_count[self.last_action] += 1
        
        # Calculate alpha depending on action-value method chosen
        alpha = self.__getAlpha__()
        
        self.Q[self.last_action] += alpha*(reward - self.Q[self.last_action]) 

    def reset(self):
        # Reinitialise Agent values
        self.Q = np.zeros(self.N)
        self.r_sum = np.zeros(self.N) # Reward Sum for each action
        self.action_count = np.zeros(self.N) # Nb of times each action was taken
        self.last_action = None

    @abc.abstractmethod
    def __getAlpha__(self):
        """
        Abstract method to calculate the value of alpha (it depends on the type of agent we're using)
        :rtype: float
        """
        pass

class SAAgent(Agent):
    """
    Sample-Averages agent.
    Update rule:
        Q(a)_n+1 = Q(a)_n + alpha*(R(a) - Q(a)_n) with alpha = 1/n 
    """
    def __getAlpha__(self):
        return 1/self.action_count[self.last_action]

    # Return string for graph legend
    def __str__(self):
        if self.epsilon == 0:
            return "SAAgent"
        else:
            return "SAAgent (epsilon : " + str(self.epsilon) + ")"

class WAAgent(Agent):
    """
    Weighted-Averages agent.
    Update rule:
        Q(a)_n+1 = Q(a)_n + alpha*(R(a) - Q(a)_n) with 0 < alpha <= 1

    with alpha being constant we get:
        Q(a)_n+q = ((1-alpha)^n)*Q(a)_n + sum(alpha*((1-alpha)^(n-i))*Ri) 
    """
    def __init__(self, alpha, *args):
        super(WAAgent, self).__init__(*args)
        self.alpha = alpha

    def __getAlpha__(self):
        return self.alpha

    # Return string for graph legend
    def __str__(self):
        if self.epsilon == 0:
            return "WAAgent"
        else:
            return "WAAgent (epsilon : " + str(self.epsilon) + ")"

class UCB_SAAgent(SAAgent):
    """
    Upper-Confident-Bound Action selection Agent
    """
    def __init__(self, c, N, epsilon=0):
        super(UCB_SAAgent, self).__init__(N, epsilon)
        self.c = c

    def step(self):
        """
        It selects an action according to their potential for actually being optimal, taking
        into account the uncertainties of each action by the rule:
        At = argmax(Qt + c*sqrt(ln(t)/N(a)))
        """
        # No need for epsilon as exploration is controlled by c


        # Step in time (choose an action)        
        n = np.sum(self.action_count)
        if n > 0: # Condition to evaluate on first iteration, because np.log(0) = -inf
            mask = self.action_count > 0 # Mask to avoid division by 0 on the formula for upper confidence uncertainties
            uncertainties = np.zeros(self.action_count.shape)
            uncertainties[mask] = self.c*np.sqrt(np.log(n)/self.action_count[mask])
            uncertainties[~mask] = float('inf') # We increment uncertainty of actions we've never chosen
        else:
            uncertainties = np.array(np.repeat(float('inf'), len(self.action_count))) 
        optimals = self.Q + uncertainties # Uncertainty rises the value of less chosen actions, hence promoting exploration
        
        max_actions = np.argwhere(optimals == np.amax(optimals)).flatten() # greedy actions (max value)
        if len(max_actions) == 1:
            self.last_action = max_actions
        else:
            self.last_action = np.random.choice(max_actions)

        return self.last_action

    # Return string for graph legend
    def __str__(self):
        return "UCB SAAgent"