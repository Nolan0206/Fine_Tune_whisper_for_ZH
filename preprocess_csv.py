import argparse

from colorama import init

# 从本地导入
from src import *
import warnings

init(autoreset=True)

warnings.filterwarnings('ignore')

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='pre_processing_csv')
    parser.add_argument('--catalog', default=None, help='txt*catalog or dir<catalog')
    parser.add_argument('--ratio', default=0.8, help='Percentage of training set')
    parser.add_argument('--num_proc', default=4, help='Number of processes')
    parser.add_argument('--path_datasetdict', default="./revise_dataset/example",
                        help='Output path of the revised dataset')
    parser.add_argument('--csv_record', default=None, type=str, help='The csv file that has been written')
    parser.add_argument('--path_whisper_hf', default=None, type=str, help='Local Configuration')
    parser.add_argument('--interleave', default=False, help='')
    parser.add_argument('--merge_audio_to_max', default=False,
                        help='if passed, then it will merge audios to `MAX_AUDIO_DURATION`')

    args = parser.parse_args()
    show_argparse(args)

    csv_file_list = read_json_config(args.catalog)
    print(csv_file_list)

    history_csv_dict = open_json_record(args.csv_record)
    new_csv_list = find_new_csv(history_csv_dict, csv_file_list)

    if args.merge_audio_to_max:
        # TODO
        pass
    else:
        common_voice = create_dataset(new_csv_list, args.ratio)
        feature_extractor, tokenizer, processor = get_whisper_hf(args.path_whisper_hf)
        preprocess_dataset = get_preprocess_dataset(feature_extractor, tokenizer)
        common_voice = common_voice.map(preprocess_dataset, remove_columns=common_voice.column_names["train"],
                                        num_proc=args.num_proc)
        create_new_folder(args.path_datasetdict)
        common_voice.save_to_disk(args.path_datasetdict)
        print("owo===========owo")
        print(f"The preprocessed arrow data has been saved to the folder: {args.path_datasetdict}")

        update_csv_dict = update_data_csv(history_csv_dict, csv_file_list)
        write_json_record(args.csv_record, update_csv_dict)
