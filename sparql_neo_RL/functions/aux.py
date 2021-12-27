def MetricTotalAccuraccy(ds_final, validation=True):
    copy_ds_final = ds_final.copy()
    copy_ds_final = copy_ds_final.reset_index(drop=True)
    bad_pred = []
    accep_pred = []
    good_pred = []
    try:
        if validation:
            for i in range(len(copy_ds_final)):
               # print(i)
                if copy_ds_final['color'][i] == 'bad prediction':
                    bad_pred.append((copy_ds_final['time'][i],copy_ds_final['y_pred'][i]))
                if copy_ds_final['color'][i] == 'aceptable prediction':
                    accep_pred.append((copy_ds_final['time'][i],copy_ds_final['y_pred'][i]))
                if copy_ds_final['color'][i] == 'good prediction':
                    good_pred.append((copy_ds_final['time'][i],copy_ds_final['y_pred'][i]))
        else:
            for i in range(len(copy_ds_final)):
               # print(i)
                if copy_ds_final['color'][i] == 'bad prediction':
                    bad_pred.append((copy_ds_final['time'][i],copy_ds_final['y_realcheck'][i]))
                if copy_ds_final['color'][i] == 'aceptable prediction':
                    accep_pred.append((copy_ds_final['time'][i],copy_ds_final['y_realcheck'][i]))
                if copy_ds_final['color'][i] == 'good prediction':
                    good_pred.append((copy_ds_final['time'][i],copy_ds_final['y_realcheck'][i]))
    except:
        print("Its not ds_final")
        return 0,0,0,0,0,0,0
    b = len(bad_pred)
    a = len(accep_pred)
    g = len(good_pred)
    tot = b+a+g
    
    bp = (b/tot)*100
    ap = (a/tot)*100
    gp = (g/tot)*100
    
    return tot,b,a,g,bp,ap,gp

