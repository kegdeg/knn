
def metrics(y_test, y_pred): 

    """
    Metric function to calculate, returns accuracy, precision, recall, f1 score

    Params
    ------------
    
    - y_test : array
    - y_pred : array


    Returns
    -------------
    - float : accuracy
    - float : precision
    - float : recall
    
    """

    #accuracy 
    total = len(y_test)
    counter = 0
    
    for i in range(total):
        if y_test[i] == y_pred[i]:
            counter += 1

    accuracy = counter/total

    confusion_matrix = conf_mat(y_test, y_pred)


    #extract TP, FP, FN from confusion matrix
    true_positives = np.diag(confusion_matrix)
    false_positives = np.sum(confusion_matrix, axis=0) - true_positives
    false_negatives = np.sum(confusion_matrix, axis=1) - true_positives

    #Calculate precision Precision = TP / TP + FP
    precision = np.nan_to_num(np.divide(true_positives, (true_positives + false_positives)))
    #Calculate recall  Recall = TP / TP + FN
    recall = np.nan_to_num(np.divide(true_positives, (true_positives + false_negatives)))
    
    return f'accuracy: {accuracy}', f'precision: {precision}', f'recall: {recall}'


#return TODO print better
