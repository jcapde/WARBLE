import pickle
from functions.ev_functions import *

if __name__ == "__main__":

    en_arr = np.load('data/output/en.npy')
    dfcomp = pickle.load(open('data/input/dfcomp.pkl','rb'))

    IND, N, K = en_arr.shape

    en_arr_est = np.zeros((5, N))
    purity_arr = np.zeros(5)
    inv_purity_arr = np.zeros(5)
    f_measure_arr = np.zeros(5)

    en_arr_est[0, :] = map_estimate(en_arr[0, :, :])
    pr, re, f = evaluate_wback(dfcomp, en_arr_est[0, :])
    purity_arr[0], inv_purity_arr[0], f_measure_arr[0] = pr, re, f
    print "Blei&Mcinerney model -", " Purity: ", pr, " Inv. Purity: ", re, " F-measure:", f

    en_arr_est[1, :] = map_estimate(en_arr[1, :, :])
    pr, re, f = evaluate_wback(dfcomp, en_arr_est[1,:])
    purity_arr[1], inv_purity_arr[1], f_measure_arr[1] = pr, re, f
    print "WARBLE model without simulatenous topic learning model -", " Inv. Purity: ", pr, " Recall: ", re, " F-measure:", f

    en_arr_est[2, :] = map_estimate(en_arr[2, :, :])
    pr, re, f = evaluate_wback(dfcomp, en_arr_est[2, :])
    purity_arr[2], inv_purity_arr[2], f_measure_arr[2] = pr, re, f
    print "WARBLE model without background -", " Purity: ", pr, " Inv. Purity: ", re, " F-measure:", f

    en_arr_est[3, :] = map_estimate(en_arr[3, :, :])
    pr, re, f = evaluate_wback(dfcomp, en_arr_est[3,:])
    purity_arr[3], inv_purity_arr[3], f_measure_arr[3] = pr, re, f
    print "WARBLE model -", " Purity: ", pr, " Inv. Purity: ", re, " F-measure:", f

    en_arr_est[4, :] = en_arr[4, :, 0] + 1
    pr, re, f = evaluate_wback(dfcomp, en_arr_est[4,:])
    purity_arr[4], inv_purity_arr[4], f_measure_arr[4] = pr, re, f
    print "Blei&Mcinerney model -", " Purity: ", pr, " Inv. Purity: ", re, " F-measure:", f

    np.save('data/output/en_est_wback.npy', en_arr_est)
    np.savetxt('data/output/purity_wback.txt', purity_arr)
    np.savetxt('data/output/inv_purity_wback.txt', inv_purity_arr)
    np.savetxt('data/output/f_measure_wback.txt', f_measure_arr)
