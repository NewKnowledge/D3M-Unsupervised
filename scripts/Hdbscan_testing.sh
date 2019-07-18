#!/bin/bash -e 

# make directory to save functional pipelines
mkdir pipelines

# create text file to record scores and timing information
touch scores.txt
echo "DATASET, SCORE, EXECUTION TIME" >> scores.txt

Datasets=('SEMI_1040_sylva_prior' 'SEMI_1217_click_prediction_small')
for i in "${Datasets[@]}"; do

  # generate and save pipeline + metafile
  python3 "/src/timeseriesd3mwrappers/TimeSeriesD3MWrappers/pipelines/Hdbscan_pipeline_best.py" $i

  # make sure there is only 1 json and 1 meta file in folder
  touch dummy.meta
  touch dummy.json
  rm *.meta
  rm *.json

  # test and score pipeline (time execution)
  start=`date +%s` 
  python3 -m d3m runtime -d /datasets/ fit-score -m *.meta -p *.json -c scores.csv
  end=`date +%s`
  runtime=$((end-start))

  # copy pipeline if execution time is less than one hour (limit on runtime for evaluation)
  if [ $runtime -lt 3600 ]; then 
     echo "$i took less than 1 hour, copying pipeline"
     cp *.meta pipelines/
     cp *.json pipelines/
  fi

  # save information to scores text file
  IFS=, read col1 score col3 col4 < <(tail -n1 scores.csv)
  echo "$i, $score, $runtime" >> scores.txt
  
  # delete scores file
  rm scores.csv

done
