import numpy as np 
import abc

class Agent(abc.ABC):
    def __init__(self, N):
        self.N = N # Number of bandits

    @abc.abstractmethod
    def step(self):
        """
        Makes the agent take an step (select an action) based on its current information.

        Return:
        -------
        action(int): Action chosen by the agent 
        """
        raise NotImplementedError

    @abc.abstractmethod
    def update(self, reward):
        """
        Updates agents information after taking an action.

        Arguments:
        ----------
        reward(float): Reward given by the enviroment after agent's behaviour (action taken) 
        """
        raise NotImplementedError
    
    @abc.abstractmethod
    def reset(self):
        """
        Resets agent's information to run a new simulation.
        """
        raise NotImplementedError

class SAAgent(Agent):
    """
    Sample-Averages e-Greedy agent.
    """

    def __init__(self, epsilon, N):
        super(SAAgent, self).__init__(N)
        self.epsilon = epsilon # if epsilon equals 0 => Greedy agent, e-Greedy agent otherwise
        self.reset()

    def reset(self):
        # Reinitialise Agent's information
        self.Q = np.zeros(self.N)
        self.r_sum = np.zeros(self.N) # Reward Sum for each action
        self.action_count = np.zeros(self.N) # Nb of times each action was taken
        self.last_action = None

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
        Update Value function depending on reward obtained using the following update rule:
        Q(a)_n+1 = Q(a)_n + alpha*(R(a) - Q(a)_n) with alpha = 1/n 
        """
        self.r_sum[self.last_action] += reward
        self.action_count[self.last_action] += 1
        
        # Calculate alpha
        alpha = 1/self.action_count[self.last_action]
        
        self.Q[self.last_action] += alpha*(reward - self.Q[self.last_action]) 

    # Return string for graph legend
    def __str__(self):
        if self.epsilon == 0:
            return "SAAgent"
        else:
            return "SAAgent (epsilon : " + str(self.epsilon) + ")"

class WAAgent(Agent):
    """
    Weighted-Averages agent.
    """
    def __init__(self, alpha, epsilon, N):
        super(WAAgent, self).__init__(N)
        self.alpha = alpha
        self.epsilon = epsilon
        self.reset()

    def reset(self):
        # Reinitialise Agent's information
        self.Q = np.zeros(self.N)
        self.r_sum = np.zeros(self.N) # Reward Sum for each action
        self.action_count = np.zeros(self.N) # Nb of times each action was taken
        self.last_action = None

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
        Update Value function depending on reward obtained using the following update rule:
        Update rule:
        Q(a)_n+1 = Q(a)_n + alpha*(R(a) - Q(a)_n) with 0 < alpha <= 1

        with alpha being constant we get:
        Q(a)_n+q = ((1-alpha)^n)*Q(a)_n + sum(alpha*((1-alpha)^(n-i))*Ri) 
        """
        self.r_sum[self.last_action] += reward
        self.action_count[self.last_action] += 1
                
        self.Q[self.last_action] += self.alpha*(reward - self.Q[self.last_action])

    # Return string for graph legend
    def __str__(self):
        if self.epsilon == 0:
            return "WAAgent (alpha : " + str(self.alpha) + ")"
        else:
            return "WAAgent (epsilon : " + str(self.epsilon) + ", alpha:" + str(self.alpha) + ")"

class UCB_SAAgent(SAAgent):
    """
    Upper-Confident-Bound Action selection Agent
    """
    def __init__(self, c, N):
        super(UCB_SAAgent, self).__init__(None, N) # No need for epsilon
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
        return "UCB SAAgent (c : " + str(self.c) + ")"

class GAgent(Agent):
    """
    Gradient Bandit Agent
    """
    def __init__(self, alpha, N, **kwargs):
        super(GAgent, self).__init__(N)
        self.alpha = alpha
        self.baseline = kwargs.get('baseline', True) # Default algorithm is with baseline

    def reset(self):
        # Reinitialise Agent values
        self.H = np.zeros(self.N, dtype=float) # We use preferences instead of estimates
        self.r_sum = np.zeros(self.N, dtype=float) # Reward Sum for each action
        self.action_count = np.zeros(self.N, dtype=int) # Nb of times each action was taken
        self.last_pi = np.zeros(self.N, dtype=float) # Last probabilities for each action
        self.last_action = None

    def step(self):
        e_H = np.exp(self.H - np.max(self.H))
        policy = e_H / np.sum(e_H, axis=0) # Softmax (prob of taking each action)
        self.last_action = np.random.choice(len(self.H), p=policy)
        self.last_pi = policy
        return self.last_action

    def update(self, reward):
        """
        Update preferences depending on reward obtained

        The update rule for this algorithm is the folowing:
        H_t+1[A_t] = H_t[A_t] + alpha(R_t - avg_R_t)*(1 - pi_t[A_t])
        H_t+1[a] = H_t[A] - alpha(R_t - avg_R_t)*pi[a] for all a != A_t
        """
        self.r_sum[self.last_action] += reward
        self.action_count[self.last_action] += 1

        n = np.sum(self.action_count, axis=0)
        
        # If a baseline is used, it should be the average reward obtained on all actions chosen
        # not the average on the last action chosen (np.sum(self.r_sum)/n instead of 
        # self.r_sum[self.last_action]/n )
        reward_avg = np.sum(self.r_sum, axis=0)/n if self.baseline else 0 

        # Vectorization of update rules into a single one
        mask = np.zeros(len(self.H))
        mask[self.last_action] = 1 # mask is equivalent to 1x==a

        # We update preferences, we increase (or decrease) the last action taken based on the reward obtained
        # compared to the average reward obtained and modify the rest of the action in the opposite direction
        self.H += self.alpha*(reward - reward_avg)*(mask - self.last_pi)

    # Return string for graph legend
    def __str__(self):
        baseline = "with" if self.baseline else "without"
        return "GAgent (alpha : " + str(self.alpha) + ") {} baseline".format(baseline)