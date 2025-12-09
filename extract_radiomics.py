import os
import logging
import pandas as pd
from radiomics import featureextractor, logger
import SimpleITK as sitk

logger.setLevel(logging.DEBUG)
handler = logging.FileHandler(filename='log/IMPLUSED_log.txt', mode='w')
formatter = logging.Formatter("%(levelname)s:%(name)s: %(message)s")
handler.setFormatter(formatter)
logger.addHandler(handler)

def extract_features(base_path, output_file, is_adc=False):
    """
    extract radiomics features from nii files
    params:
        base_path: ADC/micro folder path
        output_file
        is_adc: True/False
    """

    settings = {
        'binWidth': 0.1,
        'interpolator': sitk.sitkBSpline,
        # 'normalize': True
    }

    extractor = featureextractor.RadiomicsFeatureExtractor(**settings)
    extractor.disableAllFeatures()
    
    extractor.enableFeatureClassByName('firstorder')
    extractor.enableFeatureClassByName('shape')
    extractor.enableFeatureClassByName('glszm')
    extractor.enableFeatureClassByName('glrlm')
    extractor.enableFeatureClassByName('ngtdm')
    extractor.enableFeatureClassByName('gldm')
    extractor.enableFeaturesByName(glcm=['Autocorrelation', 'ClusterProminence', 'ClusterShade', 'ClusterTendency', 'Contrast', 'Correlation','DifferenceAverage','DifferenceEntropy','DifferenceVariance', 'JointEnergy', 'JointEntropy', 'Imc1', 'Imc2', 'Idm','MCC','Idmn','Id','Idn','InverseVariance','MaximumProbability','JointAverage','SumEntropy','SumSquares'])


    extractor.enableImageTypes(Original={}, Wavelet={}, Square={}, 
                             SquareRoot={}, Logarithm={}, Exponential={}, Gradient={})
    
    all_features = []
    
    for patient_id in sorted(os.listdir(base_path)):
        if not patient_id.startswith('Patient_'):
            continue
            
        patient_path = os.path.join(base_path, patient_id)
        
        mask_path = os.path.join(patient_path, f'mask.nii')
        if not os.path.exists(mask_path):
            print(f"warning: {mask_path} not exist, skip this patient")
            continue
        
        if is_adc:
            # ADC: 25hz, 50hz, PGSE
            image_files = [
                f'25hz.nii',
                f'50hz.nii',
                f'PGSE.nii'
            ]
        else:
            # micro: 1, 2, 3, 4
            image_files = [
                f'd.nii',
                f'vin.nii',
                f'Din.nii',
                f'Dex.nii'
            ]
        
        patient_features = {}  
                
        for img_file in image_files:
            image_path = os.path.join(patient_path, img_file)
            if not os.path.exists(image_path):
                print(f"warning: {image_path} not exist, skip this image")
                continue
            try:
                feature_vector = extractor.execute(image_path, mask_path)
                
                feature_vector['Patient'] = patient_id
                
                suffix = img_file.split('_')[-1].split('.')[0]
                
                items = list(feature_vector.items())
                for i, (key, value) in enumerate(items[22:], 23):  
                    if key not in ['Patient']:  
                        if key.startswith('original_shape_'):
                            # for shape features，save once
                            if suffix == '25hz'or suffix == 'd':
                                patient_features[key] = value  
                        else:
                            new_key = f"{key}_{suffix}"
                            patient_features[new_key] = value
                    else:
                        patient_features[key] = value  
            except Exception as e:
                print(f"failed {patient_id}/{img_file}: {str(e)}")    
        all_features.append(patient_features)   
        print(f"success: {patient_id}/{img_file}")
                
    if all_features:
        df = pd.DataFrame(all_features)
        
        # sort by Patient id (Patient_1, Patient_2, ...)
        df['Patient_num'] = df['Patient'].str.extract('(\d+)').astype(int)
        df = df.sort_values('Patient_num').drop('Patient_num', axis=1)
        
        df['label'] = [0] * 25 + [1] * 96

        # ages = pd.read_excel('../qinghai_result/所有患者信息.xlsx', usecols=['年龄'])
        # df['age'] = ages.values
        # cols = ['Patient','age'] + [col for col in df.columns if col not in ['Patient','age']]

        cols = ['Patient'] + [col for col in df.columns if col not in ['Patient']]
        df = df[cols]

        df.to_excel(output_file, index=False)
        print(f"save to {output_file}")
    else:
        print("warning: no features extracted")

def main(cls):
    save_dir='../radiomics_feature/'+cls
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    # process ADC nii files
    adc_path = '../qinghai_result/mat2nii_normalize(ADC)/'+cls
    extract_features(adc_path, save_dir+'/ADC_normalize.xlsx', is_adc=True)


    # process micro nii files
    micro_path = '../qinghai_result/mat2nii_normalize(micro)/'+cls
    extract_features(micro_path, save_dir+'/micro_normalize.xlsx', is_adc=False)


if __name__ == '__main__':
    main('IMPULSED')