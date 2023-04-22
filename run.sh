#! /bin/bash
echo "play"
python main.py --catalog_json txtwcatalog_json.txt --train_ratio 0.9 --path_to_dataset_raw /source/DataRepository/V1.1.0/raw_dataset_demo --path_to_dataset_revise /source/DataRepository/V1.1.0/revise_dataset/demo --json_record record_demo.json  --merge-audio-to-max True
