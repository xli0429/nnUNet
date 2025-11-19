# import multiprocessing
import shutil
# from multiprocessing import Pool

import SimpleITK as sitk
import numpy as np
from batchgenerators.utilities.file_and_folder_operations import *
from nnunetv2.dataset_conversion.generate_dataset_json import generate_dataset_json
from nnunetv2.paths import nnUNet_raw
from nnunetv2.paths import nnUNet_preprocessed
import json
import os
# from os.path import join 

def copy_Stanford_segmentation_and_convert_labels_to_nnUNet(in_file: str, out_file: str, in_file_prostate = None, include_prostate_label=False, include_regions=False, extra_label=None) -> None:
    # use this for segmentation only!!!
    # nnUNet wants the labels to be continuous. BraTS is 0, 1, 2, 4 -> we make that into 0, 1, 2, 3
    img = sitk.ReadImage(in_file)
    img_npy = sitk.GetArrayFromImage(img)
    if extra_label is not None:
        img_extra = sitk.ReadImage(extra_label)
        img_extra_npy = sitk.GetArrayFromImage(img_extra)
        
    if include_prostate_label==True or include_regions==True and in_file_prostate is not None:
        img_prostate = sitk.ReadImage(in_file_prostate)
        img_npy_prostate = sitk.GetArrayFromImage(img_prostate)
    
    seg_new = np.zeros_like(img_npy)
    if include_prostate_label or include_regions:
        # print("got here in func!")
        if extra_label == None:
            if not CsCancer:
                seg_new[img_npy+img_npy_prostate > 0.5] = 1
                seg_new[img_npy > 0.5] = 2
            else:
                seg_new[img_npy+img_npy_prostate > 0.5] = 1
                seg_new[img_npy > 0.5] = 2
                seg_new[img_npy > 1.5] = 3
        else:
            if not CsCancer:
                seg_new[img_npy+img_npy_prostate+img_extra_npy > 0.5] = 1
                seg_new[img_npy+img_extra_npy > 0.5] = 2
            else:
                seg_new[(img_npy > 0.5) | (img_npy_prostate > 0.5) | (img_extra_npy > 0.5)] = 1
                seg_new[(img_npy > 0.5) | (img_extra_npy > 0.5)] = 2
                seg_new[(img_npy > 1.5) | (img_extra_npy > 1.5)] = 3
                    
                    
    else:    
        # print("nope, here in func!")
        seg_new[img_npy > 0.5] = 1
        
    img_corr = sitk.GetImageFromArray(seg_new)
    img_corr.CopyInformation(img)
    sitk.WriteImage(img_corr, out_file)


def copy_PICAI_segmentation_and_convert_labels_to_nnUNet(in_file: str, out_file: str,in_file_prostate = None, include_prostate_label=False, include_regions=False, extra_label=None) -> None:
    # use this for segmentation only!!!
    # nnUNet wants the labels to be continuous. BraTS is 0, 1, 2, 4 -> we make that into 0, 1, 2, 3
    img = sitk.ReadImage(in_file)
    img_npy = sitk.GetArrayFromImage(img)
    if include_prostate_label==True or include_regions==True and in_file_prostate is not None:
        img_prostate = sitk.ReadImage(in_file_prostate)
        img_npy_prostate = sitk.GetArrayFromImage(img_prostate)

    seg_new = np.zeros_like(img_npy)
    seg_new = np.where(img_npy_prostate>0,1,0)
    seg_new = np.where(img_npy>0,2,seg_new)
    img_corr = sitk.GetImageFromArray(seg_new)
    img_corr.CopyInformation(img)
    sitk.WriteImage(img_corr, out_file)




if __name__ == '__main__':
    
    # whether or not to include prostate label as a class, in a non-region-based way
    include_prostate_label = True #True#  #True
    # whether or not to activate region-based training, flexible on the classes based on other factors set to true/false here
    include_regions = True
    # whether or not to inlcude the 3 MR channels for training 
    include_MR = True
    # whether or not to only include the 3 MR channels 
    MR_only = True
    # whether or not to use the input data projected to the TRUS space
    TRUS_space = False
    # whether or not to only use the labels registered to TRUS from MR
    MR_registered_label = False
    # whether or not to combine the labels from TRUS and the ones registered to TRUS from MR
    combine_label = False
    # use independent classes for clinically significant and indolent cancer
    # if this is set to Flase, then indolent cancer will be treated as normal tissue
    global CsCancer
    CsCancer = True
    # do not process the input data, only labels
    only_process_labels = True
        
    path_dict ={'stanford':
        {'t2': "/data/local_data/stanford/Bx_preprocessed/T2/",
        'adc': "/data/local_data/stanford/Bx_preprocessed/ADC/",
        'dwi':  "/data/local_data/stanford/Bx_preprocessed/DWI/",
        'prostate_mr': "/data/local_data/stanford/Bx_preprocessed/MRI_Prostate_Label/",
        'cancer': "/data/local_data/stanford/Bx_preprocessed/MRI_ROI_Bxconfirmed_Label/"},
        'picai':
        {'t2': "/data/local_data/picai/preprocessed/T2/",
        'adc': "/data/local_data/picai/preprocessed/ADC",
        'dwi': "/data/local_data/picai/preprocessed/DWI/",
        'prostate_mr': "/data/local_data/picai/preprocessed/MRI_Prostate_Label/",
        'cancer': "/data/local_data/picai/preprocessed/MRI_ROI_Label/"},
        'ucla':
        {'t2': "/data/local_data/ucla/Bx_preprocessed/T2/",
        'adc': "/data/local_data/ucla/Bx_preprocessed/ADC/",
        'dwi': "/data/local_data/ucla/Bx_preprocessed/DWI/",
        'prostate_mr': "/data/local_data/ucla/Bx_preprocessed/T2_Prostate_Label/",
        'cancer': "/data/local_data/ucla/Bx_preprocessed/T2_Targeted_Bxconfirmed_lesions/"}}
    
    
    stanford_fold_json= "/data/local_data/stanford/data_split/splits_final.json" 
    with open(stanford_fold_json) as json_file:
        stanford_fold_data = json.load(json_file)
    # print('stanford_fold_data.keys',stanford_fold_data[0].keys())
    
    stanford_train_ids = stanford_fold_data[0]['train']
    stanford_val_ids = stanford_fold_data[0]['val']
     

    with open('/mnt/pimed/data_processed/Stanford_Prostate_Processed/Stanford_Bx_preprocessed/data_split/Stanford_multimodal_bx_test_data.json') as json_file:
        stanford_test_json = json.load(json_file)
        
    stanford_test_list = stanford_test_json['bx_test']
    stanford_test_ids = [test['Anon_ID'] for test in stanford_test_list]

    stanford_ids = stanford_train_ids + stanford_val_ids      
    
    # picai_fold_json = "/data/local_data/picai/data_split/splits_all.json"
    # with open(picai_fold_json) as json_file:
    #     picai_fold_data = json.load(json_file)
    # picai_train_case_ids = picai_fold_data[0]['inner_train']
    # picai_val_case_ids = picai_fold_data[0]['inner_val']

    # for i in range(len(splits_final)):
    #     splits_final[i]['train'] += [x for x in picai_fold_data[i]['inner_train']]
    #     splits_final[i]['val'] += [x for x in picai_fold_data[i]['inner_val']]        


    # with open('/mnt/pimed/results2/Challenges/2022_PICAI/5Fold_CV/PICAI_test_names.json') as json_file:
    #     picai_test_ids = json.load(json_file)

    # picai_ids = picai_train_case_ids + picai_val_case_ids
    
    # with open('/data/local_data/ucla/data_split/split_all_names_new.json') as json_file:
    #     ucla_split_data = json.load(json_file)
    # ucla_train_case_ids = ucla_split_data['train']
    # ucla_val_case_ids = ucla_split_data['val']
    # ucla_test_case_ids = ucla_split_data['test']
    
    # ucla_ids = ucla_train_case_ids + ucla_val_case_ids
    
    # for i in range(len(splits_final)):
    #     splits_final[i]['train'] += [x for x in ucla_train_case_ids]
    #     splits_final[i]['val'] += [x for x in ucla_val_case_ids]        

    task_id = 201
    task_name = "BxMR_withRegions_3SEQ"


    foldername = "Dataset%03.0d_%s" % (task_id, task_name)

#     # setting up nnU-Net folders
#     print('\n nnUNet_raw:   ',nnUNet_raw,'\n')
    out_base = join(nnUNet_raw, foldername)
#     print('\n out_base:   ',out_base,'\n')
    imagestr = join(out_base, "imagesTr")
    labelstr = join(out_base, "labelsTr")
    imagests = join(out_base, "imagesTs")
    testtr = join(out_base, "TestTr")
    if not os.path.exists(imagestr):
        os.makedirs(imagestr)
    if not os.path.exists(labelstr):
        os.makedirs(labelstr)
    if not os.path.exists(imagests):
        os.makedirs(imagests)
    if not os.path.exists(testtr):
        os.makedirs(testtr)
    
    for stanford_id in stanford_ids:
        shutil.copyfile(join(path_dict['stanford']['t2'],  stanford_id+ "_t2.nii.gz"), join(imagestr, stanford_id + '_0000.nii.gz'))
        shutil.copyfile(join(path_dict['stanford']['adc'],  stanford_id+ "_adc.nii.gz"), join(imagestr, stanford_id + '_0001.nii.gz'))
        shutil.copyfile(join(path_dict['stanford']['dwi'],  stanford_id+ "_dwi.nii.gz"), join(imagestr, stanford_id + '_0002.nii.gz'))

        copy_Stanford_segmentation_and_convert_labels_to_nnUNet(join(path_dict['stanford']['cancer'], stanford_id + "_mri_roi_bxconfirmed_label.nii.gz"),
                                                                join(labelstr, stanford_id + '.nii.gz'), in_file_prostate = join(path_dict['stanford']['prostate_mr'], stanford_id + "_mri_prostate_label.nii.gz"), 
                                                                include_prostate_label=include_prostate_label, include_regions=include_regions)          

    # for picai_id in picai_ids:
    #     shutil.copyfile(join(path_dict['picai']['t2'],  picai_id+ "_t2.nii.gz"), join(imagestr, picai_id + '_0000.nii.gz'))
    #     shutil.copyfile(join(path_dict['picai']['adc'],  picai_id+ "_adc.nii.gz"), join(imagestr, picai_id + '_0001.nii.gz'))
    #     shutil.copyfile(join(path_dict['picai']['dwi'],  picai_id+ "_dwi.nii.gz"), join(imagestr, picai_id + '_0002.nii.gz'))

    #     copy_PICAI_segmentation_and_convert_labels_to_nnUNet(join(path_dict['picai']['cancer'], picai_id + "_lesions.nii.gz"),
    #                                                          join(labelstr, picai_id + '.nii.gz'),in_file_prostate = join(path_dict['picai']['prostate_mr'], picai_id + "_mask.nii.gz"), include_prostate_label=True, include_regions=True)

    # for ucla_id in ucla_ids:
    #     shutil.copyfile(join(path_dict['ucla']['t2'],  ucla_id+ "_t2.nii.gz"), join(imagestr, ucla_id + '_0000.nii.gz'))
    #     shutil.copyfile(join(path_dict['ucla']['adc'],  ucla_id+ "_adc.nii.gz"), join(imagestr, ucla_id + '_0001.nii.gz'))
    #     shutil.copyfile(join(path_dict['ucla']['dwi'],  ucla_id+ "_dwi.nii.gz"), join(imagestr, ucla_id + '_0002.nii.gz'))

    #     copy_Stanford_segmentation_and_convert_labels_to_nnUNet(join(path_dict['ucla']['cancer'], ucla_id + "_T2_bxconfirmed_label.nii.gz"), join(labelstr, ucla_id + '.nii.gz'),in_file_prostate = join(path_dict['ucla']['prostate_mr'], ucla_id + "_t2_gland.nii.gz"), include_prostate_label=True, include_regions=True)

    generate_dataset_json(out_base,
                            channel_names={
                                            0: 'T2',
                                            1: 'ADC',
                                            2: 'DWI'
                            },
                            labels={
                                'background': 0,
                                'prostate':[1,2,3],
                                'PCa': [2,3],
                                'csPCa': 3
                            },
                            regions_class_order=[1,2,3],
                            num_training_cases=len(stanford_ids),
                            file_ending='.nii.gz',)

    print('\n nnUNet_preprocessed:   ',nnUNet_preprocessed,'\n')
    preprocessed_path = join(nnUNet_preprocessed, foldername)
    if not os.path.exists(preprocessed_path):
        os.makedirs(preprocessed_path)
    splits_final_file = join(preprocessed_path,'splits_final.json')
    with open(splits_final_file, 'w', encoding='utf-8') as f:
        json.dump(stanford_fold_data, f, ensure_ascii=False, indent=4)


            
    for stanford_id in stanford_test_ids:
        shutil.copyfile(join(path_dict['stanford']['t2'],  stanford_id+ "_t2.nii.gz"), join(testtr, stanford_id + '_0000.nii.gz'))
        shutil.copyfile(join(path_dict['stanford']['adc'],  stanford_id+ "_adc.nii.gz"), join(testtr, stanford_id + '_0001.nii.gz'))
        shutil.copyfile(join(path_dict['stanford']['dwi'],  stanford_id+ "_dwi.nii.gz"), join(testtr, stanford_id + '_0002.nii.gz'))

        copy_Stanford_segmentation_and_convert_labels_to_nnUNet(join(path_dict['stanford']['cancer'], stanford_id + "_mri_roi_bxconfirmed_label.nii.gz"),
                                                            join(labelstr, stanford_id + '.nii.gz'), in_file_prostate = join(path_dict['stanford']['prostate_mr'], stanford_id + "_mri_prostate_label.nii.gz"), 
                                                            include_prostate_label=include_prostate_label, include_regions=include_regions)          
    # for picai_id in picai_test_ids:
    #     shutil.copyfile(join(path_dict['picai']['t2'],  picai_id+ "_t2_cropped.nii.gz"), join(testtr, picai_id + '_0000.nii.gz'))
    #     shutil.copyfile(join(path_dict['picai']['adc'],  picai_id+ "_adc_cropped.nii.gz"), join(testtr, picai_id + '_0001.nii.gz'))
    #     shutil.copyfile(join(path_dict['picai']['dwi'],  picai_id+ "_dwi_cropped.nii.gz"), join(testtr, picai_id + '_0002.nii.gz'))

    #     copy_PICAI_segmentation_and_convert_labels_to_nnUNet(join(path_dict['picai']['cancer'], picai_id + "_lesions_cropped.nii.gz"),
    #                                                          join(labelstr, picai_id + '.nii.gz'),in_file_prostate = join(path_dict['picai']['prostate_mr'], picai_id + "_mask_cropped.nii.gz"), include_prostate_label=True, include_regions=True)


    # for ucla_id in ucla_test_case_ids:
    #     shutil.copyfile(join(path_dict['ucla']['t2'],  ucla_id+ "_t2.nii.gz"), join(testtr, ucla_id + '_0000.nii.gz'))
    #     shutil.copyfile(join(path_dict['ucla']['adc'],  ucla_id+ "_adc.nii.gz"), join(testtr, ucla_id + '_0001.nii.gz'))
    #     shutil.copyfile(join(path_dict['ucla']['dwi'],  ucla_id+ "_dwi.nii.gz"), join(testtr, ucla_id + '_0002.nii.gz'))

    #     copy_Stanford_segmentation_and_convert_labels_to_nnUNet(join(path_dict['ucla']['cancer'], ucla_id + "_T2_bxconfirmed_label.nii.gz"), join(labelstr, ucla_id + '.nii.gz'),in_file_prostate = join(path_dict['ucla']['prostate_mr'], ucla_id + "_t2_gland.nii.gz"), include_prostate_label=True, include_regions=True)
