def calculateStatisticFast(matrix, pixel_threshold, diff_threshold):
    NDVIGraduation = {
        0 : 0, # <0.1
        1 : 0, # 0.1-0.2
        2 : 0, # 0.2-0.3
        3 : 0, # 0.3-0.4
        4 : 0, # 0.4-0.5
        5 : 0, # 0.5-0.6
        6 : 0, # 0.6-0.7
        7 : 0, # 0.7-0.8
        8 : 0, # 0.8-0.9
        9 : 0, # 0.9-1.00
        'diff' : 0 # Number of pixel with constrast (Forest-Desert, Forest-Cities)
    }
    shape = np.shape(matrix)
    temp = matrix/255.0
    
    for i,val in enumerate(np.histogram(temp[1:-1,1:-1], bins=[0.0,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1.0])[0]):
        NDVIGraduation[i] = val 
     
    diff_10 = np.where((temp - np.roll(temp,shift=1,axis=0)) > DIFF_THRESHOLD,1,0)
    diff_11 = np.where((temp - np.roll(temp,shift=1,axis=1)) > DIFF_THRESHOLD,1,0)
    diff_matrix_thr = np.where(temp>PIXEL_THRESHOLD/10.,diff_10+diff_11,0)
    
    NDVIGraduation['diff'] = diff_matrix_thr[1:-1,1:-1].sum()

    return NDVIGraduation
