import numpy as np 

# Setup for simulation

class SingleThreadOnlineSimulator:
    """
    Simulates online learning scenario - generators for input (x) and its 
    label (y) such that y can be only received after obtaining x
    """
    def __init__(self, gen_x, gen_y):
        """
        Parameters
        ----------
        gen_x: generator
               receive input (x)
        gen_y: generator
               receive label(y)
        
        Attributes
        ----------
        pending_labels : int
                         available labels
        receive_x : get input x from gen_x and increment pending label
        receive_y : get label y from gen_y if there is a pending label 
        """
        def x_decorator(generator):
            """
            decorate generator of input to get the next available input (raises 
            StopIteration if all input exhausted), also updates count of pending 
            labels so that corresponding number of labels can be retrieved
            """
            g = generator()
            def wrapper():
                self.pending_labels += 1
                return next(g)
            return wrapper
            
        def y_decorator(generator):
            """
            decorate generator of label to get the next available label (only if  
            an input has already been generated (raises StopIteration if all 
            labels exhausted)
            """
            g = generator()
            def wrapper():
                if self.pending_labels > 0:
                    self.pending_labels -= 1
                    self.round += 1
                    return next(g)
            return wrapper
            
        self.pending_labels = 0
        self.round = 0
        self.receive_x = x_decorator(gen_x)
        self.receive_y = y_decorator(gen_y)
        
class Expert:
    """
    Abstract expert with the ability to predict
    """
    def predict(self, x):
        raise NotImplementedError
        
# End simulation setup    


# Online learning algorithms

def halving_algorithm(online_data, experts):
    """
    The halving algorithm - 0/1 loss; realizable i.e. one expert predicts 
    everything correctly
    """
    # maintain best experts
    best_experts = [e for e in experts]
    # loss of experts
    loss_experts = [0 for e in experts]
    # Mistakes by algorithm
    aggregate_loss = 0
    # Simulate online scenario
    while True:
        try:
            # Receive x
            x = online_data.receive_x()
            # Get expert predictions
            y_preds = [e.predict(x) for e in best_experts]
            # Predict based on majority vote
            y_pred = 1 if 2*y_preds.count(1) > len(best_experts) else 0
            # Receive y
            y = online_data.receive_y()
            # Incorrect prediction
            if y != y_pred:
                aggregate_loss += 1
                # Retain only correct experts
                best_experts = [best_experts[i] for i in range(len(y_preds)) \
                if y_preds[i] == y]
        except StopIteration:
            break
    # retrieve best experts and aggregate mistakes
    return best_experts, aggregate_loss

def wm_algorithm(online_data, experts, beta=0.75):
    """
    The weighted majority algorithm - 0/1 loss; non-realizable i.e. no expert 
    predicts everything correctly
    """
    # maintain best experts - closer to 1 the better
    weight_experts = [1 for e in experts]
    # maintain aggregate loss of experts
    aggregate_loss_experts = [0 for e in experts]
    # Mistakes by algorithm
    aggregate_loss = 0
    # Simulate online scenario
    while True:
        try:
            # Receive x
            x = online_data.receive_x()
            # Get expert predictions
            y_preds = [e.predict(x) for e in experts]
            # Predict based on weighted majority vote
            y_pred = 1 if \
                     2*sum([w*y_p for w, y_p in zip(weight_experts, y_preds)]) \
                     > sum(weight_experts) else 0
            # Receive y
            y = online_data.receive_y()
            # Compute aggregate loss of experts
            aggregate_loss_experts = [aggregate_loss_experts[i] + 1 if \
            y_preds[i] != y else aggregate_loss_experts[i] \
            for i in range(len(y_preds))]
            # Incorrect prediction 
            if y != y_pred:
                aggregate_loss += 1
                # Reduce weight of experts with wrong prediction
                weight_experts = [beta*weight_experts[i] if y_preds[i] != y \
                else weight_experts[i] for i in range(len(y_preds)) ]
        except StopIteration:
            break
        best_expert_aggregate_loss = min(aggregate_loss_experts)
        Regret = aggregate_loss - best_expert_aggregate_loss
    return weight_experts, aggregate_loss_experts, aggregate_loss, \
    best_expert_aggregate_loss, Regret

def ewm_algorithm(online_data, experts, loss, eta=1):
    """
    The exponential weighted majority algorithm - convex loss
    """
    # maintain best experts - closer to 1 the better
    weight_experts = [1 for e in experts]
    # maintain aggregate loss of experts
    aggregate_loss_experts = [0 for e in experts]
    # Mistakes by algorithm
    aggregate_loss = 0
    # Simulate online scenario
    while True:
        try:
            # Receive x
            x = online_data.receive_x()
            # Get expert predictions
            y_preds = [e.predict(x) for e in experts]
            # Predict based on weighted majority ratio
            y_pred = sum([w*y_p for w, y_p in zip(weight_experts, y_preds)]) \
                     / sum(weight_experts)
            # Receive y
            y = online_data.receive_y()
            # Compute aggregate loss of algorithm
            aggregate_loss += loss(y_pred, y)
            # Compute aggregate loss of experts
            aggregate_loss_experts = [aggregate_loss_experts[i] + \
            loss(y_preds[i], y) if y_preds[i] != y else  aggregate_loss_experts[i] \
            for i in range(len(y_preds))]
            # Reduce weight of experts with wrong prediction
            weight_experts = [weight_experts[i] * np.exp((-loss(y_preds[i], y))) \
            for i in range(len(y_preds))]
        except StopIteration:
            break
        best_expert_aggregate_loss = min(aggregate_loss_experts)
        Regret = aggregate_loss - best_expert_aggregate_loss
    return weight_experts, aggregate_loss_experts, aggregate_loss, \
    best_expert_aggregate_loss, Regret

# End online learning algorithms    

if __name__ == '__main__':
    
    # Testing simulations
    
    def truth(samples=10):
        """
        Generators for input and labels
        Input domain x_i \in [0, 1) x [0, 1) x [0, 1)
        Label = 1 if input is within unit sphere centered at origin else 0
        this input is reused for EWM also (eventhough it can handle non binary
        labels)
        """
        x = np.random.rand(samples, 3)
        y = ((x[:,0]**2 + x[:,1]**2 + x[:,2]**2) < 1).astype(int)
        def gen_x():
            for x_i in x:
                yield x_i
        def gen_y():
            for y_i in y:
                yield y_i
        # return generators of inputs and labels                
        return (gen_x, gen_y)
        
    """
    Define experts - predicts 0/1 but is reused for generic loss in EWM also
    """
    
    # Always 1
    class Expert1(Expert):
        def predict(self, x):
            return 1
    
    # Always 0
    class Expert2(Expert):
        def predict(self, x):
            return 0      
    
    # Ideal expert
    class IdealExpert(Expert):
        def predict(self, x):
            return  1 if (x[0]**2 + x[1]**2 + x[2]**2) < 1 else 0
            
    # Good expert
    class GoodExpert(Expert):
        def predict(self, x):
            return  1 if (x[0]**2 + x[1]**2 + x[2]**2) < 0.75 else 0
    
    # Input within circle along 1st and 2nd dimension
    class Expert5(Expert):
        def predict(self, x):
            return 1 if x[0]**2 + x[1]**2 < 1 else 0
    
    # Input within circle along 2nd and 3rd dimension
    class Expert6(Expert):
        def predict(self, x):
            return 1 if x[1]**2 + x[2]**2 < 1 else 0 
    
    # Input within circle along 1st and 3rd dimension        
    class Expert7(Expert):
        def predict(self, x):
            return 1 if x[0]**2 + x[2]**2 < 1 else 0
            
            
    """Halving algorithm test"""
    print('-------------------------------------------------------------------')
    print('Test halving algorithm')
    stolsim1 = SingleThreadOnlineSimulator(*truth(100))
    experts1 = [Expert1(), Expert2(), IdealExpert(), GoodExpert(), Expert5(), 
               Expert6(), Expert7()]
    halving_algorithm_results = halving_algorithm(stolsim1, experts1)
    print(f'best experts: {halving_algorithm_results[0]}')
    print(f'aggregate (0/1) loss of algo: {halving_algorithm_results[1]}')
    print(experts1[2] in halving_algorithm_results[0])
    print('-------------------------------------------------------------------')
    
    """WM algorithm test"""
    print('-------------------------------------------------------------------')
    print('Test WM algorithm')
    stolsim2 = SingleThreadOnlineSimulator(*truth(100))
    experts2 = [Expert1(), Expert2(), GoodExpert(), Expert5(), 
               Expert6(), Expert7()]  
    wm_algorithm_results = wm_algorithm(stolsim2, experts2, beta=0.8)
    print(f'weight of experts: {wm_algorithm_results[0]}')
    print(f'aggregate (0/1) loss of algo: {wm_algorithm_results[2]}')
    print(f'aggregate loss of experts: {wm_algorithm_results[1]}')
    print(f'best expert aggregate loss: {wm_algorithm_results[3]}')
    print(f'Regret of not choosing best expert: {wm_algorithm_results[4]}')
    print('-------------------------------------------------------------------')
    
    """
    def loss function - RMSE
    """
    def rmse(y_pred, y):
        return ((y_pred - y)**2)**0.5
    
    """EWM algorithm test"""
    print('-------------------------------------------------------------------')
    print('Test EWM algorithm')
    stolsim3 = SingleThreadOnlineSimulator(*truth(100))
    experts3 = [Expert1(), Expert2(), GoodExpert(), Expert5(), 
               Expert6(), Expert7()]  
    ewm_algorithm_results = ewm_algorithm(stolsim3, experts3, rmse, eta=0.8)
    print(f'weight of experts: {ewm_algorithm_results[0]}')
    print(f'aggregate RMSE loss of algo: {ewm_algorithm_results[2]}')
    print(f'aggregate loss of experts: {ewm_algorithm_results[1]}')
    print(f'best expert aggregate loss: {ewm_algorithm_results[3]}')
    print(f'Regret of not choosing best expert: {ewm_algorithm_results[4]}')
    print('-------------------------------------------------------------------')
    
    
    

    
    

