import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.datasets as datasets
from torchvision import transforms
from torch.utils.data.dataset import random_split
from torch.utils.data import TensorDataset, Dataset, DataLoader
import numpy as np
from scipy import stats
import time
import gc
from algorithms import *
from hooks import *
from utilities import *

activation_heatmap=[]
using_hooks = False
doneThresh = 0.85
failEpochThresh = 2
failThresh = 0.50
batch_size = 1
max_epochs = 1000
#For performance reasons, accuracy is computed at a set interval of training steps
check_interval = 1000
#MNIST-specific dimensions
nr_inputs = 784
nr_outputs = 10
torch.manual_seed(42)

class Energetic_Classifier(nn.Module):
    def __init__(self,nr_inputs,nr_hidden,nr_outputs,lt,hf):
        super().__init__()
        self.loss_type = lt
        self.hidden_fn = hf
        self.input2hidden = nn.Linear(nr_inputs,nr_hidden)
        self.hidden2output = nn.Linear(nr_hidden,nr_outputs)
        with torch.no_grad():
            self.l1consolidated = nn.Linear(nr_inputs,nr_hidden)
            self.l2consolidated = nn.Linear(nr_hidden,nr_outputs)
        self.l1consolidated.weight.requires_grad = False
        self.l2consolidated.weight.requires_grad = False
        if using_hooks:
            self.hook = -1
            self.hookfun = -1

    def make_hook(self,noncompetitive_neurons):
        self.hookfun = prune_hook_factory(noncompetitive_neurons)

    def dereg_hook(self):
        if self.hook !=-1:
            self.hook.remove()

    def forward(self, pn):
        relu = nn.ReLU()
        elu = nn.ELU()
        lrelu = nn.LeakyReLU()
        sig = nn.Sigmoid()
        softmax = nn.Softmax()
        logsoftmax = nn.LogSoftmax(dim=-1)
        if self.hidden_fn == "relu":
            activation = relu
        if self.hidden_fn == "elu":
            activation = elu
        if self.hidden_fn == "lrelu":
            activation = lrelu
        if self.hidden_fn == "sig":
            activation = sig
        hidden_output = self.input2hidden(pn)

        if using_hooks:
            #Handle hooks for gradient modifications
            self.dereg_hook()
            if self.hookfun != -1:
                self.hook = hidden_output.register_hook(self.hookfun)

        #hidden_activation = sig(hidden_output)
        #hidden_activation = softmax(hidden_output)
        hidden_activation = activation(hidden_output)
        outputs = self.hidden2output(hidden_activation)
        if self.loss_type == 'KL':
            outputs = logsoftmax(outputs)
        return outputs



def evaluate(model, batcher):
    nr_digits = 10
    for x,y in batcher:
        y_np = y.detach().numpy()
        x = x.view(x.shape[0], -1)
        x = x*1
        yhot = torch.zeros(len(batcher.dataset), nr_digits)
        yhot[range(y.shape[0]), y] = 1
        y = yhot
        device = 'cpu'   
        if torch.backends.mps.is_available():
            device = 'mps'
        if torch.cuda.is_available():
            device = 'cuda:0'     
        x = x.to(device)
        y = y.to(device)
        model.eval()
        y_predict = model(x)
        yp_np = y_predict.cpu().detach().numpy()
        #neglog_loss = nn.NLLLoss()
        #mse_loss = nn.MSELoss(reduction='mean')
        #cross_loss = nn.CrossEntropyLoss()
        #loss = mse_loss(y_predict,y)
        #print(loss.data)
        predicts = np.argmax(yp_np, axis=1)
        mispredicts = y_np - predicts
        acc = 1-np.count_nonzero(mispredicts) / predicts.size
        #print("ACC:", acc)
        return acc

def train_with(*args):

    print(args)    
    #gc.enable(  )
    #gc.set_debug(gc.DEBUG_LEAK)
    (pandargs,) = args
    loss_type = pandargs['loss_function']
    hidden_fn = pandargs['hidden_activation']
    lr = pandargs['learning_rate']

    nr_hidden = pandargs['hidden_size']
    plastic_percent = pandargs['plastic_percentage']
    sample_size = pandargs['sample_size']
    algorithm = pandargs['algorithm']
    plastic_fraction = plastic_percent * 1.0 / 100
    plastics = int(nr_hidden * plastic_percent * 1.0 / 100)
    nr_rec_arrays = 12
    each_energy, each_energy_l1, each_energy_l2, each_val_acc, each_nr_iterations, each_test_acc, each_ratio, each_theoretical,\
        each_time, each_energy_array, each_acc_array, iterations_per_el = (np.array([]) for i in range(nr_rec_arrays))
    panda_list=[]

    #Run each training configuration several times, record data for each experiment, and average results
    for experiment_nr in range(sample_size):
        start_time = time.process_time()
        #MNIST-specific normalisation
        transform = transforms.Compose([transforms.ToTensor(),transforms.Normalize((0.1307,), (0.3081,))])
        mnist_dataset = datasets.MNIST(root='./data',train=True,download=True,transform=transform)
        mnist_testset = datasets.MNIST(root='./data',train=False,download=True, transform=transform)
        lengths = [int(len(mnist_dataset)*0.8), int(len(mnist_dataset)*0.2)]
        mnist_trainset, mnist_valset = torch.utils.data.random_split(mnist_dataset, lengths)

        train_batcher = DataLoader(dataset=mnist_trainset, batch_size=batch_size, shuffle=True)
        val_batcher = DataLoader(dataset=mnist_valset, batch_size=12000, shuffle=False)
        test_batcher = DataLoader(dataset=mnist_testset, batch_size=10000, shuffle=False)

        device = 'cpu'   
        if torch.backends.mps.is_available():
            device = 'mps'
        if torch.cuda.is_available():
            device = 'cuda:0' 
        network = (Energetic_Classifier(nr_inputs,nr_hidden,nr_outputs,loss_type,hidden_fn)).to(device)
        
        network.apply(weightinit)
        firstWeightsL1 = network.input2hidden.weight.detach().clone()
        firstWeightsL2 = network.hidden2output.weight.detach().clone()
        firstBiasL1 = network.input2hidden.bias.detach().clone()
        firstBiasL2 = network.hidden2output.bias.detach().clone()

        neglog_loss = nn.NLLLoss()
        mse_loss = nn.MSELoss(reduction='mean')
        cross_loss = nn.CrossEntropyLoss()
        l1_loss = nn.L1Loss()
        l1_smooth_loss = nn.SmoothL1Loss()
        KL_loss = nn.KLDivLoss()
        hub_loss = nn.HuberLoss()
        loss_fun = cross_loss

        if loss_type == "KL":
            loss_fun = KL_loss
        if loss_type == "MSE":
            loss_fun = mse_loss
        if loss_type == "NLL":
            loss_fun = cross_loss
        if loss_type == "L1":
            loss_fun = l1_loss
        if loss_type == "L1SMOOTH":
            loss_fun = l1_smooth_loss
        if loss_type == "HUB":
            loss_fun = hub_loss
        optimiser = optim.SGD(network.parameters(), lr=lr)
        totenerg = 0
        l1energ = 0
        l2energ = 0
        theoretical_best = 0
        energy_array = np.array([])
        acc_array = np.array([])
        iteration_nr = 0
        l0direct_exact = 0
        overThreshold = False
        converged = True
        #Maximum number of epochs
        firstIteration = True
        pruned_neurons = []
        epoch_nr = 0
        while not overThreshold:
            print("EPOCH " + str(epoch_nr))
            network.train()
            for x, y in train_batcher:
                iteration_nr+=1
                initialWeightsL1 = network.input2hidden.weight.detach().clone()
                initialWeightsL2 = network.hidden2output.weight.detach().clone()
                initialBiasL1 = network.input2hidden.bias.detach().clone()
                initialBiasL2 = network.hidden2output.bias.detach().clone()  

                network.input2hidden.weight.requires_grad = True
                network.hidden2output.weight.requires_grad = True

                digit = y.clone().data.detach().numpy()[0]

                #Modify gradients using Pytorch hooks
                if using_hooks: #Currently: False
                    if (algorithm == 'fixed' and firstIteration) or algorithm == 'random':
                        pruned_neurons = select_pruned_neurons_efficient(nr_hidden, nr_inputs,plastics)  
                        firstIteration = False
                    network.make_hook(pruned_neurons)

                x = x.to(device)
                y = y.to(device)
                x = x*1
                x = x.view(x.shape[0], -1)
                yhot = torch.zeros(batch_size, nr_outputs)
                yhot[range(y.shape[0]), y] = 1
                yhot = yhot.to(device)
                ytag = y
                y = yhot
                y_predict = network(x)
                if loss_type == "NLL":
                    loss = loss_fun(y_predict,ytag)
                else:
                    loss = loss_fun(y_predict,yhot)

                loss.backward()
                #Modify gradients directly  
                if not using_hooks:        
                    with torch.no_grad():
                        if plastics < nr_hidden:
                            if (algorithm == 'fixed' and firstIteration) or algorithm == 'random':      
                                mask = torch.multinomial(torch.ones(nr_hidden),nr_hidden - plastics,replacement=False).to(device)
                                firstIteration = False
                            network.input2hidden.weight.grad[mask] = 0
                            network.input2hidden.bias.grad[mask] = 0
                optimiser.step()

                finalWeightsL1 = network.input2hidden.weight.detach().clone()
                finalWeightsL2 = network.hidden2output.weight.detach().clone()
                finalBiasL1 = network.input2hidden.bias.detach().clone()
                finalBiasL2 = network.hidden2output.bias.detach().clone()
                deltaWL1 = torch.abs(torch.sub(finalWeightsL1,initialWeightsL1))
                deltaWL2 = torch.abs(torch.sub(finalWeightsL2,initialWeightsL2))
                deltaB1 = torch.abs(torch.sub(finalBiasL1,initialBiasL1))
                deltaB2 = torch.abs(torch.sub(finalBiasL2,initialBiasL2))
                l0delta = torch.count_nonzero(deltaWL1) + torch.count_nonzero(deltaWL2) + torch.count_nonzero(deltaB1) + torch.count_nonzero(deltaB2)
                l0direct_exact += l0delta.item()
                costl1 = (torch.sum(deltaWL1) + torch.sum(deltaB1)).item()
                costl2 = (torch.sum(deltaWL2) + torch.sum(deltaB2)).item()
                cost = costl1 + costl2
                #Energy modelled as proportional to the absolute sum total of changes in the weights
                total_weight = (torch.sum(torch.abs(torch.sub(finalWeightsL1,firstWeightsL1))) + torch.sum(torch.abs(torch.sub(finalWeightsL2,firstWeightsL2))) + 
                                torch.sum(torch.abs(torch.sub(finalBiasL1,firstBiasL1))) + torch.sum(torch.abs(torch.sub(finalBiasL2,firstBiasL2)))).item()
                totenerg += cost
                l1energ += costl1
                l2energ += costl2

                # Regularly verify if validation accuracy is over the threshold
                if not(iteration_nr % check_interval):
                    acc = evaluate(network,val_batcher)
                    print(experiment_nr,nr_hidden,plastics,acc,iteration_nr)
                    acc_array = np.append(acc_array,acc)
                    energy_array = np.append(energy_array,totenerg)                    
                    #print(psutil.virtual_memory())
                    #cpuStats()
                    #memReport()
                    gc.collect()
                    if acc >= doneThresh:
                        overThreshold = True
                        theoretical_best = total_weight
                        break
                    #Over epoch threshold
                    if epoch_nr >= max_epochs:
                        converged = False
                        overThreshold = True
                        break
                    #Failed to achieve sufficient accuracy after a few epochs, terminating early
                    if acc < failThresh and epoch_nr >= failEpochThresh:
                        converged = False
                        overThreshold = True
                        break   
                optimiser.zero_grad()
            epoch_nr += 1

        print("DONE - final iteration_nr and validation accuracy:")
        print(args, iteration_nr, acc)
        print("final energy:")
        print(args,totenerg)
        test_acc = evaluate(network,test_batcher)

        if experiment_nr == 0:
            each_energy_array = np.zeros(len(energy_array))
            each_acc_array = np.zeros(len(acc_array))
            iterations_per_el = np.zeros(len(energy_array))
        runtime = time.process_time() - start_time
        eff_ratio = theoretical_best / totenerg
        nr_updates = (nr_hidden * (plastic_fraction * nr_inputs + 1) + nr_outputs * (nr_hidden + 1))
        l0 = nr_updates * iteration_nr
        #Compute actual fraction of plastic (nonzero gradient) units
        p_actual = plastic_fraction * (l0direct_exact / l0)

        #Storing list of results from every iteration, to be averaged
        each_energy = np.append(each_energy,totenerg)
        each_energy_l1 = np.append(each_energy_l1,l1energ)
        each_energy_l2 = np.append(each_energy_l2,l2energ)
        each_val_acc = np.append(each_val_acc,acc)
        each_nr_iterations = np.append(each_nr_iterations,iteration_nr)
        each_test_acc = np.append(each_test_acc,test_acc)
        each_ratio = np.append(each_ratio,eff_ratio)
        each_theoretical = np.append(each_theoretical,theoretical_best)
        each_time = np.append(each_time,(runtime))

        #Handle mismatched length when averaging energy and accuracy "paths"      
        latest_length = len(energy_array)
        if len(each_acc_array) < len(acc_array):
            each_acc_array.resize(len(acc_array))
            each_energy_array.resize(len(energy_array))
            iterations_per_el.resize(len(energy_array))            
        if len(acc_array) < len(each_acc_array):  
            acc_array.resize(len(each_acc_array))   
            energy_array.resize(len(each_energy_array)) 
            iterations_per_el.resize(len(each_energy_array))        
        iterations_per_el[:latest_length] += 1
        each_acc_array = np.add(each_acc_array,acc_array)
        each_energy_array = np.add(each_energy_array,energy_array)

        variable_names=[totenerg,l1energ,l2energ,iteration_nr,l0,eff_ratio,theoretical_best,runtime,acc,test_acc,l0direct_exact,p_actual]
        panda_names=["M_tot","M_L1","M_L2","iteration_nr","L0","efficiency","theoretical_best","time","val_acc","test_acc","L0_direct","p_actual"]
        panda_data=[{'p':plastic_fraction,'data_val':it1,'data_type':it2,'network_size':pandargs['hidden_size'],'algorithm':pandargs['algorithm'],'activation':pandargs['hidden_activation'],
            'loss':pandargs['loss_function'],'lr':pandargs['learning_rate'],'converged':converged} for it1,it2 in zip(variable_names,panda_names)]
        panda_list = panda_list + panda_data

    avg_totenerg = np.mean(each_energy)
    avg_el1 = np.mean(each_energy_l1)
    avg_el2 = np.mean(each_energy_l2)
    avg_val = np.mean(each_val_acc)
    avg_nr = np.mean(each_nr_iterations)
    avg_test = np.mean(each_test_acc)
    avg_ratio = np.mean(each_ratio)
    avg_best = np.mean(each_theoretical)
    avg_time = np.mean(each_time)
    avg_energy_array = each_energy_array / iterations_per_el
    avg_acc_array = each_acc_array / iterations_per_el
    #torch.save(network.state_dict(), './dendrite_competition_10K/ae_lr_'+str(lr)+'_cr_'+str(competitive_ratio)+'.params')

    return (avg_totenerg,stats.sem(each_energy)),(avg_el1,stats.sem(each_energy_l1)),(avg_el2,stats.sem(each_energy_l2)),(avg_nr,stats.sem(each_nr_iterations)),\
        (avg_val,stats.sem(each_val_acc)),(avg_test,stats.sem(each_test_acc)),(avg_ratio,stats.sem(each_ratio)),(avg_best,stats.sem(each_theoretical)),\
        (avg_time,stats.sem(each_time)),avg_energy_array,avg_acc_array,panda_list

