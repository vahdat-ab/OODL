import matplotlib.pyplot as plt
import numpy as np
import h5py
# from numpy import linalg as LA
from   tensorflow.keras.models import Model
import tensorflow.keras.backend as K
import sys
from sklearn.preprocessing import StandardScaler
import tensorflow.keras as keras
from tensorflow.keras.layers import Activation
from tensorflow.keras import layers
from tensorflow.keras.layers import Lambda, Input
from tensorflow.keras.utils import get_custom_objects
from tensorflow.keras.layers import Activation
from tensorflow.keras import layers
import tensorflow as tf

import scipy.special as special
from general_setting import *
from metrics import *
from utility import calculate_input_gradients, perturb_inputs, preprocess_images, \
                    postprocess_features, save_data_hdf5,get_dataset_hdf5,\
                    build_one_class_svm, combine_inliners_outliers, apply_temp_scale_to_model,\
                    apply_log_temp_scale_to_model, perturb_inputs_odin, extract_layer_features

#------------------------------------------------------------------------------------------------------------------------
def detect_ood_softmax(in_org_model, in_model, in_mix_id_odd):
    pred = in_model.predict(in_mix_id_odd)
    scores = - np.max(pred, axis=1)
    return scores

#------------------------------------------------------------------------------------------------------------------------
def detect_ood_odin(in_org_model, in_model, in_mix_id_odd, in_extra_data, input_preprocessing=False):
    if input_preprocessing==False:
        per_in_mix_id_odd = in_mix_id_odd
        
    else:
        pred_labels= np.argmax( in_model[2].predict(in_mix_id_odd), axis=1 )
        print("used perturbation magnitude is ", in_extra_data[0])
        per_in_mix_id_odd = perturb_inputs_odin(in_model[1], in_mix_id_odd, pred_labels, per_magnitude=in_extra_data[0])
    scores = detect_ood_softmax(in_org_model, in_model[0], per_in_mix_id_odd)
    return scores

#------------------------------------------------------------------------------------------------------------------------
def detect_mah_dist(in_org_model, in_model, in_mix_id_odd, in_extra_data, in_num_class, input_preprocessing=False):     
    if input_preprocessing==False:
        per_in_mix_id_odd = in_mix_id_odd
        ev_model = in_model
    else:
        pred_labels= np.argmin(in_model[0].predict(in_mix_id_odd), axis=1 )
        print("used perturbation magnitude is ", in_extra_data[0])
        per_in_mix_id_odd = perturb_inputs(in_model[1], in_mix_id_odd, pred_labels,\
                                                  per_magnitude=in_extra_data[0], operator="subtract")
        ev_model = in_model[0]
        
    pred   = - ev_model.predict(per_in_mix_id_odd)
    scores = - np.max(pred, axis=1)
    return scores

#------------------------------------------------------------------------------------------------------------------------
def detect_entropy(in_org_model, in_model, in_mix_id_odd): 
    pred = in_model.predict(in_mix_id_odd)
    scores = special.entr(pred).sum(axis=-1)
    return scores


#------------------------------------------------------------------------------------------------------------------------
def detect_margin(in_org_model, in_model, in_mix_id_odd): 
    pred = in_model.predict(in_mix_id_odd)
    max_softmax = np.max(pred, axis=1)
    second_max_softmax  = np.sort(pred, axis=1)[:,-2]
    scores = -(max_softmax - second_max_softmax)
    return scores


#------------------------------------------------------------------------------------------------------------------------
def detect_mc_dropout(in_org_model, in_model, in_mix_id_odd): 
    nb_MC_samples = 100
    MC_output = K.function([in_model.layers[0].input, K.learning_phase()], [in_model.layers[-1].output])
    learning_phase = True  # use dropout at test time

    batch_size = 2000
    for i_d in range(in_mix_id_odd.shape[0]//batch_size):
        images_batch = in_mix_id_odd[i_d*batch_size:(i_d+1)*batch_size]
        MC_samples = [MC_output([images_batch, learning_phase])[0] for _ in range(nb_MC_samples)]
        MC_samples = np.array(MC_samples)        
        if i_d==0:
            features = MC_samples
        else:
            features = np.column_stack((features,MC_samples))

    if (i_d+1)*batch_size <in_mix_id_odd.shape[0]:
        images_batch = in_mix_id_odd[(i_d+1)*batch_size:in_mix_id_odd.shape[0]]
        MC_samples = [MC_output([images_batch, learning_phase])[0] for _ in range(nb_MC_samples)]
        MC_samples = np.array(MC_samples)  
        if i_d==0:
            features = MC_samples
        else:
            features = np.column_stack((features,MC_samples))
    
    variance = np.mean(features, axis=0)
    scores = - np.max(variance, axis=1)
    return scores

#------------------------------------------------------------------------------------------------------------------------
def detect_mutual_info(in_org_model, in_model, in_mix_id_odd): 
    nb_MC_samples = 100
    MC_output = K.function([in_model.layers[0].input, K.learning_phase()], [in_model.layers[-1].output])
    learning_phase = True  # use dropout at test time

    batch_size = 2000
    for i_d in range(in_mix_id_odd.shape[0]//batch_size):
        images_batch = in_mix_id_odd[i_d*batch_size:(i_d+1)*batch_size]
        MC_samples = [MC_output([images_batch, learning_phase])[0] for _ in range(nb_MC_samples)]
        MC_samples = np.array(MC_samples)        
        if i_d==0:
            features = MC_samples
        else:
            features = np.column_stack((features,MC_samples))

    if (i_d+1)*batch_size <in_mix_id_odd.shape[0]:
        images_batch = in_mix_id_odd[(i_d+1)*batch_size:in_mix_id_odd.shape[0]]
        MC_samples = [MC_output([images_batch, learning_phase])[0] for _ in range(nb_MC_samples)]
        MC_samples = np.array(MC_samples)  
        if i_d==0:
            features = MC_samples
        else:
            features = np.column_stack((features,MC_samples))
    
    expected_entropy = - np.mean(np.sum(features * np.log(features + 1e-10), axis=-1), axis=0)  # [batch size]
    expected_p = np.mean(features, axis=0)
    entropy_expected_p = - np.sum(expected_p * np.log(expected_p + 1e-10), axis=-1)  # [batch size]
    BALD_acq = entropy_expected_p - expected_entropy
    scores = BALD_acq
    return scores

#------------------------------------------------------------------------------------------------------------------------
def detect_ood_svm(in_org_model, in_model, in_mix_id_odd, in_extra_data, input_preprocessing=False):
    if input_preprocessing==False:
        per_in_mix_id_odd = in_mix_id_odd
    else:
        pred_labels= np.argmax( in_org_model.predict(in_mix_id_odd), axis=1 )
        print("used perturbation magnitude is ", in_extra_data[0])
        per_in_mix_id_odd = perturb_inputs_odin(in_model[1], in_mix_id_odd, pred_labels, per_magnitude=in_extra_data[0])

    features = in_model[0].predict(per_in_mix_id_odd)
    features_vector = postprocess_features(features)
    features_vector_norm = in_model[-1].transform(features_vector)
    scores =  - in_model[-2].score_samples(features_vector_norm)
    return scores

#------------------------------------------------------------------------------------------------------------------------
def odin_perturbation_magnitude(in_org_model, in_model, in_id_data, in_ood_data):
    np.random.seed(0)
    sln_eval_inx = np.random.randint(0, in_id_data.shape[0],(in_id_data.shape[0]*2)//10 )
    id_img = in_id_data[sln_eval_inx]
    id_pred_labels= np.argmax( in_model[-1].predict(id_img), axis=1 )

    np.random.seed(0)
    sln_eval_inx = np.random.randint(0, in_ood_data.shape[0],(in_ood_data.shape[0]*2)//10)
    ood_img = in_ood_data[sln_eval_inx]
    ood_pred_labels= np.argmax(in_model[-1].predict(ood_img), axis=1 )

    fpr_at_95_tpr = 101.0
    best_value=0.0
    for mag_val in PER_MAGNITUDE_LIST:
        print("\nper_mag:",mag_val)
        id_img_perturbed = perturb_inputs_odin(in_model[1], id_img, id_pred_labels, per_magnitude=mag_val)
        ood_img_perturbed = perturb_inputs_odin(in_model[1], ood_img, ood_pred_labels, per_magnitude=mag_val)

        m_data, m_labels = combine_inliners_outliers(id_img_perturbed,ood_img_perturbed)
        scores = detect_ood_odin(in_org_model, in_model, m_data, None, input_preprocessing=False)
        current_fpr_ = get_summary_statistics(scores,m_labels)["fpr_at_95_tpr"]
        print("\nfpr_at_95_tpr", current_fpr_)
        if current_fpr_ <fpr_at_95_tpr:
            fpr_at_95_tpr =current_fpr_
            best_value=mag_val
            print("<<set the magnitude value to "+ str(best_value) +" >>")
    print("The best perturbation magnitude is", best_value)
    return best_value


#------------------------------------------------------------------------------------------------------------------------
def mah_perturbation_magnitude(in_org_model, in_model, in_id_data, in_ood_data, in_num_class):
    np.random.seed(0)
    sln_eval_inx = np.random.randint(0, in_id_data.shape[0],(in_id_data.shape[0]*2)//10 )
    id_img = in_id_data[sln_eval_inx]
    id_pred_labels= np.argmin( in_model[0].predict(id_img), axis=1 )

    np.random.seed(0)
    sln_eval_inx = np.random.randint(0, in_ood_data.shape[0],(in_ood_data.shape[0]*2)//10)
    ood_img = in_ood_data[sln_eval_inx]
    ood_pred_labels= np.argmin(in_model[0].predict(ood_img), axis=1 )

    fpr_at_95_tpr = 101.0
    best_value=0.0
    for mag_val in PER_MAGNITUDE_LIST:
        print("\nper_mag:",mag_val)
        id_img_perturbed = perturb_inputs(in_model[1], id_img, id_pred_labels, per_magnitude=mag_val,\
                                          operator="subtract")
        ood_img_perturbed = perturb_inputs(in_model[1], ood_img, ood_pred_labels, per_magnitude=mag_val,\
                                           operator="subtract")

        m_data, m_labels = combine_inliners_outliers(id_img_perturbed,ood_img_perturbed)
        scores = detect_mah_dist(in_org_model, in_model[0], m_data, None, in_num_class, input_preprocessing=False)
        current_fpr_ = get_summary_statistics(scores,m_labels)["fpr_at_95_tpr"]
        print("\nfpr_at_95_tpr", current_fpr_)
        if current_fpr_ <fpr_at_95_tpr:
            fpr_at_95_tpr =current_fpr_
            best_value=mag_val
            print("<<set the magnitude value to "+ str(best_value) +" >>")
    print("The best perturbation magnitude is", best_value)
    return best_value
#------------------------------------------------------------------------------------------------------------------------
def ours_perturbation_magnitude(in_org_model, in_model, in_id_data, in_ood_data):
    np.random.seed(0)
    sln_eval_inx = np.random.randint(0, in_id_data.shape[0],(in_id_data.shape[0]*2)//10 )
    id_img = in_id_data[sln_eval_inx]
    id_pred_labels= np.argmax( in_org_model.predict(id_img), axis=1 )

    np.random.seed(0)
    sln_eval_inx = np.random.randint(0, in_ood_data.shape[0],(in_ood_data.shape[0]*2)//10)
    ood_img = in_ood_data[sln_eval_inx]
    ood_pred_labels= np.argmax(in_org_model.predict(ood_img), axis=1 )

    fpr_at_95_tpr = 101.0
    best_value=0.0
    for mag_val in PER_MAGNITUDE_LIST:
        print("\nper_mag:",mag_val)
        id_img_perturbed = perturb_inputs_odin(in_model[1], id_img, id_pred_labels, per_magnitude=mag_val)
        ood_img_perturbed = perturb_inputs_odin(in_model[1], ood_img, ood_pred_labels, per_magnitude=mag_val)

        m_data, m_labels = combine_inliners_outliers(id_img_perturbed,ood_img_perturbed)
        scores = detect_ood_svm(in_org_model, in_model, m_data, None, input_preprocessing=False)
        current_fpr_ = get_summary_statistics(scores,m_labels)["fpr_at_95_tpr"]
        print("\nfpr_at_95_tpr", current_fpr_)
        if current_fpr_ <fpr_at_95_tpr:
            fpr_at_95_tpr =current_fpr_
            best_value=mag_val
            print("<<set the magnitude value to "+ str(best_value) +" >>")
    print("The best perturbation magnitude is", best_value)
    return best_value

#------------------------------------------------------------------------------------------------------------------------
def get_pert_magnitude(in_ood_appr_name, in_org_model, in_model, in_id_data, in_ood_data, in_num_class):
    if in_ood_appr_name=="ours_w_p":
        per_val =  ours_perturbation_magnitude(in_org_model, in_model, in_id_data, in_ood_data)
    elif in_ood_appr_name=="odin_w_p":
        per_val =  odin_perturbation_magnitude(in_org_model, in_model, in_id_data, in_ood_data)
    elif in_ood_appr_name=="mah_dist_logits_w_p":
        per_val =  mah_perturbation_magnitude(in_org_model, in_model, in_id_data, in_ood_data, in_num_class)
        
    return per_val

#------------------------------------------------------------------------------------------------------------------------
def cal_mah_dist_parameters(in_model, in_num_class, in_images, in_labels):
    sample_class_mean=[]
    tmp_sigma=np.zeros((in_model.output.shape[-1], in_model.output.shape[-1]))
    for c in range(in_num_class):
        slc_img   = in_images[in_labels==c]
        logits    = in_model.predict(slc_img)
        sample_class_mean.append(np.mean(logits, axis=0))
        tmp_sigma = tmp_sigma+np.cov(logits, rowvar=0)

    sample_class_mean=np.asarray(sample_class_mean)
    sigma=tmp_sigma/in_images.shape[0]
    sigma_inv = np.linalg.pinv(sigma)
    return sample_class_mean, sigma_inv
#------------------------------------------------------------------------------------------------------------------------
def get_inference_mah_model(in_org_model, in_model, in_sample_mean, in_sigma_inverse, in_num_class):
    sample_mean = in_sample_mean
    sigma_inverse = in_sigma_inverse
    def output_of_lambda(input_shape):
        return (input_shape[0],in_num_class)
    
    def mah_clasification_layer(x):
        outputs=[]
        for i in range(in_num_class):
            diff_x_m = x - sample_mean[i]
            sigma_tens= tf.convert_to_tensor(sigma_inverse)
            sigma_tens= tf.cast(sigma_tens, tf.float32)
            tmp = tf.einsum('nl,lp->np', diff_x_m,sigma_tens)
            diff_x_m_f = K.expand_dims(diff_x_m, axis=2)
            output = tf.einsum('np,npz->nz', tmp, diff_x_m_f )
            if i==0:
                outputs = output
            else:
                outputs=tf.concat([outputs, output],1)
        return outputs
    
    x = in_model.output
    mah_layer = Lambda(mah_clasification_layer, output_shape=output_of_lambda)(x)
    mah_model=Model(inputs=in_model.input, outputs=mah_layer) 
    
    return mah_model