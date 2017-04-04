import pickle
from functions.ev_functions_BCubed import *

if __name__ == "__main__":

    dataset = pickle.load(open('data/tmp/dataset.pkl','rb'))
    event_assig = pickle.load(open('data/output/event_assignments.npy', 'rb'))

    purity_arr = np.zeros(5)
    inv_purity_arr = np.zeros(5)
    f_measure_arr = np.zeros(5)

    pr, re, f = evaluate(dataset, event_assig[0,:])
    purity_arr[0], inv_purity_arr[0], f_measure_arr[0] = pr, re, f
    print "Mcinerney & Blei model -", " Purity: ", pr, " Inv. Purity: ", re, " F-measure:", f

    pr, re, f = evaluate(dataset, event_assig[1,:])
    purity_arr[1], inv_purity_arr[1], f_measure_arr[1] = pr, re, f
    print "WARBLE model without simulatenous topic learning model -", " Inv. Purity: ", pr, " Recall: ", re, " F-measure:", f

    pr, re, f = evaluate(dataset, event_assig[2,:])
    purity_arr[2], inv_purity_arr[2], f_measure_arr[2] = pr, re, f
    print "WARBLE model without background -", " Purity: ", pr, " Inv. Purity: ", re, " F-measure:", f

    pr, re, f = evaluate(dataset, event_assig[3,:])
    purity_arr[3], inv_purity_arr[3], f_measure_arr[3] = pr, re, f
    print "WARBLE model -", " Purity: ", pr, " Inv. Purity: ", re, " F-measure:", f

    pr, re, f = evaluate(dataset, event_assig[4,:])
    purity_arr[4], inv_purity_arr[4], f_measure_arr[4] = pr, re, f
    print "Tweet-SCAN -", " Purity: ", pr, " Inv. Purity: ", re, " F-measure:", f

    np.savetxt('data/output/purity.txt', purity_arr)
    np.savetxt('data/output/inv_purity.txt', inv_purity_arr)
    np.savetxt('data/output/f_measure.txt', f_measure_arr)
