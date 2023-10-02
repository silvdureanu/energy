from trainergy import train_with
from multiprocessing import Pool
import pandas as pd
import json

if __name__ == '__main__':

    counter = 0
    consolidationFactor = 50  
    parameters = json.load(open('./config.json'))
    learning_rates = parameters['learning_rates']
    plastic_ratios = parameters['plastic_ratios'] 
    nr_totals = parameters['nr_totals']
    losses = parameters['losses']
    hidden_activations = parameters['hidden_activations']
    algorithms = parameters['algorithms']
    sample_size = parameters['sample_size']
    prefix = parameters['prefix']
    pandargs = [{'hidden_size':nr_total,'algorithm':algorithm,'hidden_activation':h_a,'loss_function':loss,'learning_rate':lr,'plastic_percentage':rp,
                 'sample_size':sample_size} for nr_total in nr_totals for algorithm in algorithms for h_a in hidden_activations for loss in losses\
                      for lr in learning_rates for rp in plastic_ratios]
    nr_measurements = 13
    totenergs, l1energs, l2energs, nrs, val_accs, test_accs, ratios, bests, times, res, entrajs, acctrajs, pandatable = ([] for i in range(nr_measurements)) 
    jobs = []
    for arg in pandargs:
        x = [arg]
        jobs = jobs + x
        counter += 1
        #consolidationFactor limits the size of the total workload mapped to pool, in order to alleviate memory issues in large runs
        if counter % consolidationFactor == 0:
            pool = Pool(5) # parallel jobs
            restemp = pool.map(train_with, jobs)
            res = res + restemp
            pool.close()
            jobs=[]
            
    #Train leftover network configurations
    if counter % consolidationFactor != 0 and consolidationFactor > 1:
        pool = Pool(5)
        restemp = pool.map(train_with,jobs)
        res = res + restemp    
        pool.close()

    for totenerg, l1e, l2e, nr, val_acc, test_acc, ratio, best, time, entraj, acctraj, panres in res:
        totenergs.append(totenerg)
        l1energs.append(l1e)
        l2energs.append(l2e)
        nrs.append(nr)
        val_accs.append(val_acc)
        test_accs.append(test_acc)
        ratios.append(ratio)
        bests.append(best)
        times.append(time)
        entrajs.append(entraj)
        acctrajs.append(acctraj)
        pandatable = pandatable + panres

    #Save results as pickled Pandas dataframe
    pandata=pd.DataFrame(pandatable)
    pandata.to_pickle('./fixednet_lr0p005_N1000.data')

    #Save results as text files - mean and standard deviation  
    prefixstring = prefix + 'output_'
    written_measurements = [totenergs,l1energs,l2energs,nrs,val_accs,test_accs,ratios,bests,times]
    written_filenames = ['energ.out','L1energ.out','L2energ.out','nr.out','valacc.out','testacc.out','efficiency.out','theoreticalbest.out','time.out']

    for (measurement,filename) in zip(written_measurements,written_filenames):
        s = prefixstring + filename
        out = open(s,'w')
        for (mean,std) in measurement:
            out.write('%f %f\n' %(mean,std))
        out.close()

    #Energy and accuracy trajectories
    written_traj_measurements = [entrajs,acctrajs]
    written_traj_filenames = ['trajectory_energy.out','trajectory_accuracy.out']

    for (measurement,filename) in zip(written_traj_measurements,written_traj_filenames):
        s = prefixstring + filename
        out = open(s,'w')
        for traj in measurement:
            for element in traj:
                out.write('%f ' %(element))
            out.write('\n')
        out.close()        