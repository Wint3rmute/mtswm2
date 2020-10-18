from parse_stroke_data_file import get_all_data_files
from parse_dataset_description import get_readme_features_and_diagnoses
import numpy as np

from sklearn.feature_selection import chi2, mutual_info_classif

if __name__ == "__main__":


    datas = get_all_data_files()

    (
        non_heart_origin_pain_data,
        angina_prectoris_data,
        angina_prectoris_2_data,
        mi_data,
        mi_np_data,
    ) = datas

    print("Example data shapes, in format (number_of_specimen, number_of_features)")
    print(non_heart_origin_pain_data.shape)
    print(angina_prectoris_data.shape)

    print()
    print("Concatenating all the input data matrixes...")
    scikit_compliant_x_matrix = np.concatenate([*datas])
    print("Done")
    print()

    print("Make sure that the data after concatenation is")
    print("still in format (number_of_specimen, number_of_features)")
    print(scikit_compliant_x_matrix.shape)
    print()

    print(scikit_compliant_x_matrix[1])
    
    print('Time to generate the Y values..')
    scikit_compliant_y_matrix = np.concatenate(
        [np.full(data.shape[0], index) for (index, data) in enumerate(datas)]
    )
    print(f'Done, the shape is', scikit_compliant_y_matrix.shape)
    print()
   
    print('Calculating the chi2 score')
    chi2_score = chi2(
        scikit_compliant_x_matrix,
        scikit_compliant_y_matrix
    )

    print('Done')
    # print(chi2_score)
    print()

    print('Now using the mutual info classifier')
    mutual_info_classif_score = mutual_info_classif(
        scikit_compliant_x_matrix,
        scikit_compliant_y_matrix
    )

    print(mutual_info_classif_score)
    
    print()
    readme_features, readme_diagnoses = get_readme_features_and_diagnoses()

    features_with_scores = list(zip(readme_features, mutual_info_classif_score))
    features_with_scores.sort(key=lambda tup: tup[1] ,reverse=True)

    # print(features_with_scores)
    for feature, score in features_with_scores:
        print(f'{feature}: {score}')

