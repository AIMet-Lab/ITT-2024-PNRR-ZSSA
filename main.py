
import argparse
import configparser

import pandas

import utilities


def make_parser() -> argparse.ArgumentParser:

    parser = argparse.ArgumentParser("Zero-shot Sentiment Analysis and Keyword Extraction Script.")

    output_folder_help = "Folder which will contain the outputs of the script."
    parser.add_argument("--output_folder", type=str, help=output_folder_help, default="outputs/exp1/")

    config_path_help = "Path to the Config file containing the parameters of the script."
    parser.add_argument("--config_path", type=str, help=config_path_help, default="config/default.ini")

    return parser


if __name__ == "__main__":

    arg_parser = make_parser()
    args = arg_parser.parse_args()

    output_folder = args.output_folder
    config_path = args.config_path

    config = configparser.ConfigParser()
    _ = config.read(config_path)

    use_neutral = config["DEFAULT"].getboolean("use_neutral")
    do_data_analysis = config["DEFAULT"].getboolean("do_data_analysis")
    images_format = config["DEFAULT"]["images_format"]
    verbose = config["DEFAULT"].getboolean("verbose")

    datasets_config_path = config["DEFAULT"]["datasets_config_path"]
    keywords_config_path = config["DEFAULT"]["keywords_config_path"]
    models_config_path = config["DEFAULT"]["models_config_path"]

    datasets_config = pandas.read_csv(datasets_config_path)
    keywords_config = pandas.read_csv(keywords_config_path)
    models_config = pandas.read_csv(models_config_path)

    sentiment_folder, keyword_folder, graphs_folder, logs_folder = utilities.get_output_subfolders(output_folder)

    if do_data_analysis:
        utilities.data_analysis(datasets_config, graphs_folder, images_format)

    utilities.sentiment_analysis(datasets_config, models_config, use_neutral, sentiment_folder, logs_folder, verbose)
    utilities.keywords_extraction(datasets_config, models_config, keywords_config, keyword_folder, verbose)
    utilities.analyse_key_ext_results(keywords_config, keyword_folder, graphs_folder, images_format, verbose)
