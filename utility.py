import matplotlib.pyplot as plt
import numpy as np
import h5py
from   tensorflow.keras.models import Model
import tensorflow.keras.backend as K
import sys
from sklearn.preprocessing import StandardScaler
from sklearn import svm
import tensorflow.keras as keras
from tensorflow.keras.utils import get_custom_objects
from tensorflow.keras.layers import Activation
from tensorflow.keras import layers
from tqdm import tqdm
import tensorflow as tf
"""
Some part of this code has been adopted from
https://scikit-learn.org/stable/auto_examples/model_selection/plot_roc.html#sphx-glr-auto-examples-model-selection-plot-roc-py
"""

def risk_coverage_plot( steps, in_data, in_labels, in_pred_labels, in_max_scores, min_v, max_v, log ):    
    x_coverage =[]
    y_rejection = []
    x_steps = []
    y_risk = []
    min_thre = min_v
    max_thre = max_v
    
    for i in np.arange(min_thre, max_thre+steps , steps):
        threshold = i
        n_all_i = in_data.shape[0]
 
        #not be classified indices
        nb_c_i = np.where(in_max_scores < threshold)
        n_r_i = len(nb_c_i[0])
        sbc_labels = np.delete(in_labels, nb_c_i[0])
        sbc_pred_labels = np.delete(in_pred_labels, nb_c_i[0])
        n_a_i = sbc_labels.size
        coverage = n_a_i / n_all_i
        misclassified_indices = [sbc_pred_labels ==sbc_labels]
        misclassified_indices = np.flatnonzero(misclassified_indices)
        n_c_i = misclassified_indices.size
        n_m_i = n_a_i - n_c_i
        if (coverage==0):
                risk = 0
        else:
            risk = (n_m_i / n_all_i) / coverage
            
        if not coverage in x_coverage :
            x_coverage.append(coverage)
            y_risk.append(risk)
            y_rejection.append(n_r_i) 
            x_steps.append(threshold)
        if log:
            print(i)
            print("number of all inputs: ", n_all_i)
            print("number of rejected inputs: ", n_r_i)
            print("number of accepted inputs: ",n_a_i)
            print("the coverage is : ",coverage)
            print("number of correctly classified inputs: ",n_c_i)
            print("number of  misclassified inputs: ",n_m_i)
            print("The risk is: ",risk)
    return x_coverage, y_risk, x_steps, y_rejection

def cal_risk_coverage( steps, in_data, in_labels, in_pred_labels, in_max_scores, log ):    
    x_coverage =[]
    y_rejection = []
    x_steps = []
    y_risk = []
    min_thre = np.min(in_max_scores)
    max_thre = np.max(in_max_scores)
    
    for i in np.arange(min_thre, max_thre+steps , steps):
        threshold = i
        n_all_i = in_data.shape[0]
 
        #not be classified indices
        nb_c_i = np.where(in_max_scores < threshold)
        n_r_i = len(nb_c_i[0])

        sbc_labels = np.delete(in_labels, nb_c_i[0])
        sbc_pred_labels = np.delete(in_pred_labels, nb_c_i[0])
        n_a_i = sbc_labels.size
        coverage = n_a_i #/ n_all_i

        misclassified_indices = [sbc_pred_labels ==sbc_labels]
        misclassified_indices = np.flatnonzero(misclassified_indices)
        n_c_i = misclassified_indices.size
        n_m_i = n_a_i - n_c_i
        if (coverage==0):
                risk = 0
        else:
            risk = (n_m_i / n_all_i) #/ coverage
            
            print(coverage, "risk:" , risk, "threshold:",threshold, "n_m_i", n_m_i )
                
        if not coverage in x_coverage :
            x_coverage.append(coverage)
            y_risk.append(risk)
            y_rejection.append(n_r_i) 
            x_steps.append(threshold)
        if log:
            print(i)
            print("number of all inputs: ", n_all_i)
            print("number of rejected inputs: ", n_r_i)
            print("number of accepted inputs: ",n_a_i)
            print("the coverage is : ",coverage)
            print("number of correctly classified inputs: ",n_c_i)
            print("number of  misclassified inputs: ",n_m_i)
            print("The risk is: ",risk)
    return x_coverage, y_risk, x_steps, y_rejection

def combine_inliners_outliers(inliers, outliers, i_label = 0, o_label=1, verbose=False):
    """
        
    """
    temp_outliers = outliers
    temp_inliers  = inliers
    if verbose:
        print("The shape of the inlier dataset: ", temp_inliers.shape)
        print("The shape of the outlier dataset: ", temp_outliers.shape)
    if len(temp_inliers.shape) != len(temp_outliers.shape):
        print("The shape of inliers and outliers should be the same.")
        return None, None
    if temp_inliers.shape[0] != temp_outliers.shape[0]:
        print("The number of inliers and outliers should be the same.\nThe final size will be adjusted.")
        
        if inliers.shape[0]<outliers.shape[0]:
            temp_outliers = outliers[0:inliers.shape[0]]
        else:
            temp_inliers = inliers[0:outliers.shape[0]]
        print("The current size for each dataset is: ",temp_inliers.shape[0])
    if i_label == o_label:
        print("The labels of an inlier and outliers should not be the same.")
        return None, None

    
    if i_label==0:
        i_labels = np.zeros(temp_inliers.shape[0])
    else:
        i_labels = np.ones(temp_inliers.shape[0])
        
    if o_label==0:
        o_labels = np.zeros(temp_outliers.shape[0])
    else:
        o_labels = np.ones(temp_outliers.shape[0])       
              
    mixed_labels =  np.append(i_labels, o_labels)
    mixed_data = np.vstack((temp_inliers, temp_outliers))
    if verbose:
        print("the shape of the final dataset is",mixed_data.shape)
        print("the shape of the final dataset's labels is",mixed_data.shape)
        print("The label for an inlier and outlier are: ", i_label, o_label)
        
    return mixed_data, mixed_labels


def save_data_hdf5(data, ds_name, file_address, mode="w"):
    """
    r: Readonly, file must exist
    r+: Read/write, file must exist
    w:Create file, truncate if exists
    w- or x:Create file, fail if exists
    a:Read/write if exists, create otherwise (default)
    """
    h5_file = h5py.File(file_address+".hdf5", mode)
    dset = h5_file.create_dataset(ds_name, shape=data.shape, dtype='float64')
    dset[:] = data
    h5_file.close()

def get_dataset_hdf5(ds_name, file_address, without_Ext=True):
    if without_Ext:
        file_address=file_address+".hdf5"
        
    h5_file = h5py.File(file_address, "r")
    data = np.copy(h5_file[ds_name])
    h5_file.close()
    return data

def del_dataset_hdf5(ds_name, file_address, without_Ext=True): 
    """
    This does not reduce the size of file.
    """
    if without_Ext:
        file_address=file_address+".hdf5"
        
    h5_file = h5py.File(file_address, "a")
    del h5_file [ds_name]


def ext_all_act(model, images,list_of_layers=None,rem_top=1, verbose=False):
    """
    """
    if list_of_layers==None:
        for i,l in enumerate(model.layers):
            if i == len(model.layers)-rem_top: break
            if verbose: print("L_index",i,"L_name" ,l.name, l.output.shape)
            aux_model = Model(inputs=model.input, outputs=l.output)
            features  = aux_model.predict(images)
            features  = cal_l2_normalize_l1(features)
            if i==0:
                final_features = features
            else:
                final_features = np.concatenate((final_features, features), axis=(-1))
            if verbose: print("features:", final_features.shape)
    else:
        for i,indx in enumerate(list_of_layers):
            l = model.layers[indx]
            if verbose: print(indx, l.name, l.output.shape)
            aux_model = Model(inputs=model.input, outputs=l.output)
            features  = aux_model.predict(images)
            features  = cal_l2_normalize_l1(features)
            if i==0:
                final_features = features
            else:
                final_features = np.concatenate((final_features, features), axis=(-1))
            if verbose: print("features:", final_features.shape)          
        
    return final_features


def  extract_all_activations(model, images,list_of_layers=None, remove_top=1, batch_size=None, verbose=False):
    if batch_size==None:
        return  ext_all_act(model= model, images= images, list_of_layers= list_of_layers,\
                            rem_top=remove_top, verbose= verbose)
    if (batch_size>images.shape[0]):
        print("The bach size must be smaller than the size of input")
        return None
    
    num_steps = images.shape[0]//batch_size
    if num_steps==0:
        final_f = ext_all_act(model= model, images= images[l_b:h_b], list_of_layers= list_of_layers,\
                              rem_top= remove_top, verbose= verbose)
        return final_f
     
    for i in range(num_steps):
        l_b, h_b = i*batch_size , (i+1)*batch_size
        if verbose: 
              print(i,"/",num_steps)
              print("Lowe_bound:", l_b,"Upper_bound:", h_b)
        f = ext_all_act(model= model, images= images[l_b:h_b], list_of_layers= list_of_layers,\
                        rem_top= remove_top, verbose= verbose)
        if i==0:
            final_f = f
        else:
            final_f = np.vstack((final_f, f))
        
    if num_steps*batch_size <images.shape[0]:
        l_b = h_b
        h_b = images.shape[0]
        if verbose: print("Lowe_bound", l_b,"Upper_bound", h_b)
        f = ext_all_act(model= model, images= images[l_b:h_b], list_of_layers= list_of_layers,\
                        rem_top= remove_top, verbose= verbose)
        final_f = np.vstack((final_f, f))
    if verbose: print("FINAL SHAPE:",final_f.shape)    
    return final_f



def calculate_input_gradients(model, classes=None):
    """
    """
    if model==None:
        print("The model is none.")
        return None
    
    if classes==None:
        classes=range(model.output.shape[-1])
    #--------------------------------    
    gradients = {}
    sys.stdout.write("\n")
    for c in classes:
        grads = K.gradients(model.output[:,c], model.layers[0].input)[0]
        gradients[c]= K.function([model.layers[0].input], [grads])              
    
        sys.stdout.write('\r')
        sys.stdout.write("Progress of building the computation graph: "+str(int((c+1)*1.0/len(classes)*1000)*1.0/10)+"%")
        sys.stdout.flush()
    #--------------------------------
    return gradients

def perturb_inputs(in_grads_graph,in_data, in_labels, per_magnitude=0.007, operator="add"):
    input_grads=[]
    sys.stdout.write("\n")
    for i in range(in_data.shape[0]):
        exp =in_grads_graph[in_labels[i]]([in_data[i:i+1]])[0][0] 
        input_grads.append(exp)
        sys.stdout.write('\r')
        sys.stdout.write("Progress of calculating gradients: "+str(int((i+1)*1.0/in_data.shape[0]*1000)*1.0/10)+"%")
        sys.stdout.flush()
    input_grads = np.asarray(input_grads)
    if operator=="add":
        perturbed_data = in_data + (per_magnitude * (np.sign(input_grads)) )
    elif operator=="subtract":
        perturbed_data = in_data - (per_magnitude * (np.sign(input_grads)) )
    else:
        perturbed_data=None
    return perturbed_data

def perturb_inputs_odin(in_grads_graph,in_data, in_labels, per_magnitude=0.007):
    input_grads=[]
    sys.stdout.write("\n")
    for i in range(in_data.shape[0]):
        exp =in_grads_graph[in_labels[i]]([in_data[i:i+1]])[0][0] 
        input_grads.append(exp)
        sys.stdout.write('\r')
        sys.stdout.write("Progress of calculating gradients: "+str(int((i+1)*1.0/in_data.shape[0]*1000)*1.0/10)+"%")
        sys.stdout.flush()
    input_grads = np.asarray(input_grads)
    perturbed_data = in_data - (per_magnitude * (-1 * np.sign(input_grads)) )
    return perturbed_data

def load_images_split(in_ds_name, in_split="TEST"):
    img=None
    lb=None
    if in_split=="TEST":
        (_,_),(img, lb) = load_dataset(in_ds_name)
        
    elif in_split=="TRAIN":
        (img,lb),(_,_) = load_dataset(in_ds_name)
        
    else:
        print("The split parameter value should be either TRAIN or TEST.")

    return img, lb

def preprocess_images(in_id_name, images, in_id_model, verbose=False):
    if in_id_name=="CIFAR10" and in_id_model=="VGG16":
        mean = 120.707
        std = 64.15
        if verbose:
            print("Preprocessing was done for ",in_id_name, in_id_model)
        return (images-mean)/(std+1e-7)
    
    if in_id_name=="CIFAR100" and in_id_model=="VGG16":
        mean = 121.936
        std = 68.389
        if verbose:
            print("Preprocessing was done for ",in_id_name, in_id_model)
        return (images-mean)/(std+1e-7)
    
    if in_id_name=="CIFAR10" and in_id_model=="VGG16-Org":
        tmp_img = preprocess_input(images)
        if verbose:
            print("Preprocessing was done for ",in_id_name, in_id_model)
        return tmp_img
    
    if in_id_name=="CIFAR10" and (in_id_model=="ResNet-V1" or in_id_model=="ResNet-V1-44"):
        (x_train, _), (_, _) = keras.datasets.cifar10.load_data()
        x_train = x_train.astype('float32') / 255
        images = images.astype('float32') / 255
        x_train_mean = np.mean(x_train, axis=0)       
        if verbose:
            print("Preprocessing was done for ",in_id_name, in_id_model)
        return images- x_train_mean

    if in_id_name=="CIFAR10" and in_id_model=="DenseNet-40-12":
        images = images.astype('float32')
        # 'RGB'->'BGR'
        images = images[..., ::-1]
        # Zero-center by mean pixel
        images[..., 0] -= 103.939
        images[..., 1] -= 116.779
        images[..., 2] -= 123.68
        images *= 0.017 # scale values
     
        if verbose:
            print("Preprocessing was done for ",in_id_name, in_id_model)
        return images

    if in_id_name=="CIFAR100" and in_id_model=="ResNet-V1-44":
        (x_train, _), (_, _) = keras.datasets.cifar100.load_data()
        x_train = x_train.astype('float32') / 255
        images = images.astype('float32') / 255
        x_train_mean = np.mean(x_train, axis=0)       
        if verbose:
            print("Preprocessing was done for ",in_id_name, in_id_model)
        return images- x_train_mean
    
    
    
    if verbose: print("Preprocessing is not needed.")
    return images

def extract_all_features(in_model, in_data, in_layer_list, in_batch_size, in_file_address):
    for l in in_layer_list:
        features = extract_layer_features(in_model, in_data, l, in_batch_size)
        save_data_hdf5(features, "l_"+str(l),in_file_address, "a")
        print("The featurs of layer "+ str(l)+" was saved. The shape of saved features is "+  str(features.shape))    
        
def extract_layer_features(in_model, in_data, in_layer, in_batch_size, shorten_model=False):  
    for i_d in range(in_data.shape[0]//in_batch_size):
        batch_data = in_data[i_d*in_batch_size:(i_d+1)*in_batch_size]
        
        if shorten_model==False:
            batch_features = extract_features(in_model, batch_data, in_layer)
        else:
            batch_features=in_model.predict(batch_data)
            
        batch_features_processed = postprocess_features(batch_features)
        if i_d==0:
            features = batch_features_processed
        else:
            features = np.vstack((features,batch_features_processed))
            
    if (i_d+1)*in_batch_size <in_data.shape[0] :
        batch_data = in_data[(i_d+1)*in_batch_size:in_data.shape[0]]
        
        if shorten_model==False:
            batch_features = extract_features(in_model, batch_data, in_layer)
        else:
            batch_features=in_model.predict(batch_data)
            
        batch_features_processed = postprocess_features(batch_features)
        if i_d==0:
            features = batch_features_processed
        else:
            features = np.vstack((features,batch_features_processed))
        
    return features

def extract_features(in_model, in_img_perturbed, in_layer_inx):
    l=in_model.layers[in_layer_inx]
    aux_model = Model(inputs=in_model.input, outputs=l.output)
    return aux_model.predict(in_img_perturbed)

def postprocess_features(in_features):
    if len(in_features.shape)==4:
        output=np.sum(in_features, axis=(1,2))
    if len(in_features.shape)==2:
        output=in_features
    return output


def build_one_class_svm(train_images, test_images=None, ood_images=None, show_eval=False, nu_value=0.001, kernel='rbf', \
                                                      gamma_value ="scale"):
    #kernel=['linear', 'poly', 'rbf', 'sigmoid']
    
    ss = StandardScaler()
    ss.fit(train_images)
    train_images_ss = ss.transform(train_images)
    clf = svm.OneClassSVM(nu=nu_value, kernel=kernel, gamma=gamma_value)
    clf.fit(train_images_ss)
    if show_eval:
        test_img_ss  = ss.transform(test_images)
        ood_img_ss   = ss.transform(ood_images)
        y_pred_train = clf.predict(train_images_ss) 
        y_pred_test = clf.predict(test_img_ss)      
        y_pred_ood = clf.predict(ood_img_ss)

        n_error_train = y_pred_train[y_pred_train == -1].size
        n_error_test  = y_pred_test[y_pred_test == -1].size
        n_error_ood   = y_pred_ood[y_pred_ood == 1].size

        print('error in training_set ', n_error_train/train_img.shape[0])
        print('error in test_set ', n_error_test/test_img.shape[0])
        print('error in ODD test_set ', n_error_ood/ood_img.shape[0]) 
    
    return clf, ss 


def apply_temp_scale_to_model(in_model, logit_inx_from_top, verbose=False):
    def temp_scale_softmax(x):
        return tf.nn.softmax(x/1000, axis=-1)
    get_custom_objects().update({'temp_scale_softmax': Activation(temp_scale_softmax)})
    
    org_weights = in_model.layers[logit_inx_from_top].get_weights()
    
    x = in_model.layers[logit_inx_from_top].input
    o = layers.Dense(in_model.layers[-1].output.shape[-1], activation=temp_scale_softmax)(x)
    new_model=Model(inputs=in_model.input, outputs=[o])
    new_model.layers[-1].set_weights(org_weights)
    if verbose:
        new_mode.summary()

    return new_model

def apply_log_temp_scale_to_model(model, logit_inx_from_top, t_value =1000, verbose=False):
    def temp_scale_log_softmax(x):
        return tf.nn.log_softmax(x/t_value)
    get_custom_objects().update({'temp_scale_log_softmax': Activation(temp_scale_log_softmax)})
    
    org_weights = model.layers[logit_inx_from_top].get_weights()
    
    x = model.layers[logit_inx_from_top].input
    o = layers.Dense(model.layers[-1].output.shape[-1], activation=temp_scale_log_softmax)(x)
    new_model=Model(inputs=model.input, outputs=[o])
    new_model.layers[-1].set_weights(org_weights)
    if verbose:
        new_mode.summary()
    
    return new_model