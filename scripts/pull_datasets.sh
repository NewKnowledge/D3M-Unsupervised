cd ../../datasets
Datasets=('SEMI_1040_sylva_prior_MIN_METADATA' 'SEMI_1044_eye_movements_MIN_METADATA' 'SEMI_1217_click_prediction_small_MIN_METADATA')
for i in "${Datasets[@]}"; do
    git lfs pull -I "seed_datasets_current/$i/"
done