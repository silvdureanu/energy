def prune_hook_factory(noncompetitive_neurons):
    def prune_hook(grad):
        grad_clone = grad.clone()
        grad_clone[0] = grad_clone[0] * noncompetitive_neurons
        return grad_clone
    return prune_hook

def prune_synapses_hook_factory(noncompetitive_synapses):
    def prune_hook(grad):
        grad_clone = grad.clone()
        grad_clone = grad_clone * noncompetitive_synapses
        return grad_clone
    return prune_hook

def prune_hook_random(grad, noncompetitive_neurons):
    grad_clone = grad.clone()
    for i in noncompetitive_neurons:
        grad_clone[0,i]=0
    return grad_clone

def prune_hook_random_refractory(grad, noncompetitive_neurons, nr_hidden, refracted_neurons, refractory_time):
    grad_clone = grad.clone()
    for i in noncompetitive_neurons:
        grad_clone[0,i]=0
    refracted_neurons = [i-1 for i in refracted_neurons]
    refracted_neurons = [i if i>=0 else 0 for i in refracted_neurons]
    for i in range(nr_hidden):
        if not i in noncompetitive_neurons:
            refracted_neurons[i] += refractory_time
    return grad_clone, refracted_neurons

def smart_prune_hook_random_refractory(grad, noncompetitive_neurons, nr_hidden, refracted_neurons, refractory_time, digit_grads, current_digit):
    grad_clone = grad.clone()
    grad_clone_2 = grad.clone()
    digit_grads[current_digit] = grad_clone_2
    for i in noncompetitive_neurons:
        grad_clone[0,i]=0
    refracted_neurons = [i-1 for i in refracted_neurons]
    refracted_neurons = [i if i>=0 else 0 for i in refracted_neurons]
    for i in range(nr_hidden):
        if not i in noncompetitive_neurons:
            refracted_neurons[i] += refractory_time
    return grad_clone, refracted_neurons

def prune_worst_hook(grad, activation_heatmap, noncompetitive_neurons, nr_hidden, competitive_ratio):
    grad_clone = grad.clone()
    gradarray = grad_clone.clone().data.detach().numpy()
    v = gradarray[0].argsort()
    nr_discarded = int(nr_hidden * (1-competitive_ratio))
    noncompetitive_neurons = v[:nr_discarded]
    selected_neurons = [1 for i in range(nr_hidden)]
    noncompetitive_neurons.sort()
    for i in noncompetitive_neurons:
        grad_clone[0,i]=0
        selected_neurons[i] = 0
    activation_heatmap.append(selected_neurons)
    return grad_clone, activation_heatmap, noncompetitive_neurons