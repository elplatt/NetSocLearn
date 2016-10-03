import numpy as np
import numpy.random as nprand
import elp_networks as enet

def sample_pure(n, d=3):
    '''Choose <n> pure strategies uniformly at random.'''
    states = []
    for i in range(n):
        v = np.zeros(shape=(d,))
        v[nprand.randint(0, d)] = 1.0
        states.append(v)
    return states

def mean_state(network, states):
    '''Calculate the mean state over the entire network.'''
    state_sums = np.sum([s for s in states.itervalues()], axis=0)
    state = state_sums / np.sum(state_sums)
    return state

def mean_neighbor_state(network, states, node):
    '''Calculate the mean state over self and neighbors'''
    # Find mean state of neighbors
    state_sums = np.sum([states[v] for v in network.neighbors(node)], axis=0)
    state_sums += states[node]
    state_frac = state_sums / np.sum(state_sums)
    return state_frac

def best_response(payoff, neighbor_state):
    '''Return the pure best response to neighbor_state w/ given payoffs.'''
    # Find dimension
    d = len(neighbor_state)
    # Convert to column vector
    state_vec = np.matrix(neighbor_state).transpose()
    # Find maximal payoffs
    # Convert column vector to row, then list of rows, take first row
    choice_payoff = (payoff * state_vec).transpose().tolist()[0]
    best_p = max(choice_payoff)
    best_i_list = [i for i, p in enumerate(choice_payoff) if p == best_p]
    # Chose randomly between maximal payoffs
    best_i = nprand.choice(best_i_list)
    # Save next state for later
    new_state = np.zeros(shape=(d,))
    new_state[best_i] = 1.0
    return new_state

def learn(payoff, network, initial, iter_rule, iter_count):
    '''Iteratively learn strategies for each node.
       Return list of weights for each strategy.'''
    states = dict(initial)
    d = payoff.shape[0]
    # Create list to track state history, and initialize
    state_frac = mean_state(network, states)
    history = list()
    for i in range(d):
        history.append([state_frac[i]])
    # Step through time
    for t in range(iter_count):
        new_states = dict(states)
        for node in network.nodes:
            neighbor_state = mean_neighbor_state(network, states, node)
            new_states[node] = iter_rule(payoff, neighbor_state)
        # Update states simultaneously
        states = new_states
        # Update history
        state_frac = mean_state(network, states)
        for i in range(d):
            history[i].append(state_frac[i])
    return history
