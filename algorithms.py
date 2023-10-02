import torch
import numpy as np

def select_pruned_neurons(nr_neurons, ratio):
    nr_discarded = int(nr_neurons * (1-ratio))
    return np.random.choice(nr_neurons,nr_discarded,replace=False)

def select_pruned_synapses(nr_neurons, nr_synapses, ratio):
    nr_discarded = int(nr_synapses * (1-ratio))
    mask = np.ones((nr_neurons,nr_synapses))
    for i in range(nr_neurons):
        x = np.random.choice(nr_synapses,nr_discarded,replace=False)
        mask[i,x] = 0
    return torch.FloatTensor(mask)

def select_pruned_neurons_efficient(nr_neurons, nr_synapses, plastics):
    nr_plastic = plastics
    device = 'cpu'   
    if torch.backends.mps.is_available():
        device = 'mps'
    if torch.cuda.is_available():
        device = 'cuda:0' 
    mask = torch.zeros(nr_neurons,device=device)
    if nr_plastic > 0:
        x = torch.multinomial(torch.ones(nr_neurons),nr_plastic,replacement=False).to(device)
        mask[x] = 1.0
    return mask

def mute_first_layer(nr_neurons,nr_synapses):
    mask = np.zeros([nr_neurons,nr_synapses])
    return torch.FloatTensor(mask)

def select_pruned_neurons_refractory(nr_neurons, nr_synapses, ratio, refractory_time, refracted_neurons):
    nr_discarded = int(nr_neurons * (1-ratio))
    nr_refracted = np.count_nonzero(refracted_neurons)
    if nr_neurons - nr_refracted < nr_discarded:
        nr_discarded = max(nr_neurons - nr_refracted,0)

    selected_mask = np.ones(nr_neurons)
    mask = np.ones([nr_neurons,nr_synapses])
    nonrefracted_indices = np.where(refracted_neurons == 0)[0]
    x = np.random.choice(nonrefracted_indices,nr_discarded,replace=False)
    mask[x] = 0
    selected_mask[x] = 0

    for i in range(nr_neurons):
        refracted_neurons[i] = max(refracted_neurons[i]-1, 0)
        
    refracted_neurons[np.nonzero(selected_mask)] += refractory_time
    return torch.FloatTensor(mask),refracted_neurons


def smart_select_nonrefractory_pruned_neurons(nr_neurons, ratio, refracted_neurons, digit_grads, current_digit):
    nr_discarded = int(nr_neurons * (1-ratio))
    nr_tweaked = int(1.0 * nr_neurons / 10)
    tweak_magnitude = 0.1
    available_neurons = [i for i in range(nr_neurons) if refracted_neurons[i] == 0]
    uniform_probs = [1.0 / nr_neurons for i in range(nr_neurons)]
    last_grad = digit_grads[current_digit]
    if last_grad != []:
        v = last_grad[0].argsort()
        highest_probs = v[:nr_tweaked]
        for i in highest_probs:
            uniform_probs[i] += tweak_magnitude

    tweaked_probs = [i / sum(uniform_probs) for i in uniform_probs]
    probs = [tweaked_probs[i] for i in available_neurons]
    norm_probs = [i / sum(probs) for i in probs]
    return np.random.choice(available_neurons,nr_discarded,p=norm_probs,replace=False), refracted_neurons, digit_grads, current_digit