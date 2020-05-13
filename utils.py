def MSE(computed,real):
    s=0
    for i in range(len(computed)):
        s+=(computed[i]-real[i])**2
    return s/len(computed)