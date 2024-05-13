
import os
import logging
import time
import sys

import pandas
import wordcloud
import matplotlib
import transformers
import scipy

import matplotlib.pyplot as plt
import numpy as np


def castles_data_cleaning(data_path: str) -> pandas.DataFrame:

    raw_df = pandas.read_csv(data_path)
    clean_df = pandas.DataFrame()
    clean_df['text'] = raw_df['Review']
    clean_df['score'] = raw_df['Rating']
    return clean_df


def restaurants_data_cleaning(data_path: str) -> pandas.DataFrame:
    raw_df = pandas.read_csv(data_path)
    clean_df = pandas.DataFrame()
    clean_df['text'] = raw_df['text']
    clean_df['score'] = raw_df['rating']
    return clean_df


def hotels_data_cleaning(data_path: str) -> pandas.DataFrame:
    raw_df = pandas.read_csv(data_path)
    clean_df = pandas.DataFrame()
    clean_df['text'] = raw_df['reviews.text']
    clean_df['score'] = raw_df['reviews.rating']
    return clean_df


def tripadvisor_hotels_data_cleaning(data_path: str) -> pandas.DataFrame:
    raw_df = pandas.read_csv(data_path)
    clean_df = pandas.DataFrame()
    clean_df['text'] = raw_df['Review']
    clean_df['score'] = raw_df['Rating']
    return clean_df


def get_dataframe(data_id: str, data_path: str) -> pandas.DataFrame:

    if data_id == 'castles':
        df = castles_data_cleaning(data_path)
    elif data_id == 'hotels':
        df = hotels_data_cleaning(data_path)
    elif data_id == 'restaurants':
        df = restaurants_data_cleaning(data_path)
    elif data_id == 'trip_hotels':
        df = tripadvisor_hotels_data_cleaning(data_path)
    else:
        raise NotImplementedError

    return df


def get_sentiment_given_score(row) -> str:

    score = row['score']
    if score < 2.5:
        return "negative"
    elif score > 3.5:
        return "positive"
    else:
        return "neutral"


def get_loggers(filepath: str):

    logger_path = filepath

    stream_logger = logging.getLogger("Log Stream")
    file_logger = logging.getLogger("Log File")

    stream_handler = logging.StreamHandler(sys.stdout)
    file_handler = logging.FileHandler(logger_path, 'a+')

    stream_handler.setLevel(logging.INFO)
    file_handler.setLevel(logging.INFO)

    stream_logger.addHandler(stream_handler)
    file_logger.addHandler(file_handler)

    stream_logger.setLevel(logging.INFO)
    file_logger.setLevel(logging.INFO)

    return stream_logger, file_logger


def get_stream_logger():

    stream_logger = logging.getLogger("Log Stream")
    stream_handler = logging.StreamHandler(sys.stdout)
    stream_handler.setLevel(logging.INFO)
    stream_logger.addHandler(stream_handler)
    stream_logger.setLevel(logging.INFO)

    return stream_logger


def get_output_subfolders(output_folder: str) -> tuple:

    if not os.path.exists(output_folder):
        os.mkdir(output_folder)

    sentiment_folder = output_folder + "sentiment_analysis/"
    if not os.path.exists(sentiment_folder):
        os.mkdir(sentiment_folder)

    keyword_folder = output_folder + "keywords_extraction/"
    if not os.path.exists(keyword_folder):
        os.mkdir(keyword_folder)

    graphs_folder = output_folder + "graphs/"
    if not os.path.exists(graphs_folder):
        os.mkdir(graphs_folder)

    logs_folder = output_folder + "logs/"
    if not os.path.exists(logs_folder):
        os.mkdir(logs_folder)

    return sentiment_folder, keyword_folder, graphs_folder, logs_folder


def make_wordcloud(dataset_id: str, dataset_path: str, save_path: str, max_n_words=50, colormap='cividis'):

    df = get_dataframe(dataset_id, dataset_path)

    text_data = ' '.join(df['text'].to_list())

    wc = wordcloud.WordCloud(
        background_color='white',
        max_words=max_n_words,
        max_font_size=40,
        scale=3,
        random_state=42,
        min_word_length=3,
        colormap=colormap
    ).generate(str(text_data))

    fig = plt.figure(1, figsize=(10, 6), tight_layout=True)
    plt.axis('off')
    plt.imshow(wc)
    plt.savefig(save_path)
    plt.close(fig)


def make_dataset_hist(dataset_ids: list, dataset_paths: list, save_path: str):

    matplotlib.rc('xtick', labelsize=10)
    matplotlib.rc('ytick', labelsize=10)
    fig, axs = plt.subplots(1, 3, figsize=(10, 3), tight_layout=True)

    counter = 0

    for i in range(len(dataset_ids)):
        df = get_dataframe(dataset_ids[i], dataset_paths[i])
        counts, edges, bars = axs[counter].hist(df['score'], color="crimson", bins=[1, 2, 3, 4, 5, 6], align='left',
                                                rwidth=0.5)
        axs[counter].set_xticks([1, 2, 3, 4, 5])
        axs[counter].set_xlim([0, 6])
        axs[counter].set_ylim([0, df.__len__()])
        axs[counter].set_title(dataset_ids[i], fontsize=18)
        axs[counter].bar_label(bars)

        if counter == 0:
            axs[counter].set_ylabel("# reviews", fontsize=15)

        if counter == 1:
            axs[counter].set_xlabel("stars", fontsize=15)

        counter += 1

    plt.savefig(save_path)
    plt.close(fig)


def make_dataset_hist_single(dataset_id: str, dataset_path: str, save_path: str):

    matplotlib.rc('xtick', labelsize=10)
    matplotlib.rc('ytick', labelsize=10)

    plt.figure(tight_layout=True)

    df = get_dataframe(dataset_id, dataset_path)
    counts, edges, bars = plt.hist(df['score'], color="crimson", bins=[1, 2, 3, 4, 5, 6], align='left',
                                   rwidth=0.5)

    plt.xticks([1, 2, 3, 4, 5])
    plt.xlim([0, 6])
    plt.ylim([0, df.__len__()])
    plt.title(dataset_id, fontsize=18)
    plt.bar_label(bars)
    plt.ylabel("# reviews", fontsize=15)
    plt.xlabel("stars", fontsize=15)

    plt.savefig(save_path)
    plt.close()

def make_results_hist(median_df: pandas.DataFrame, iqr_df: pandas.DataFrame, keywords: list, save_path: str):

    matplotlib.rc('xtick', labelsize=20)
    matplotlib.rc('ytick', labelsize=20)
    fig, axs = plt.subplots(5, 2, figsize=(20, 26), tight_layout=True)

    fontsize = 40
    for i in range(len(keywords)):

        axs[i, 0].hist(median_df.values[:, i], color="crimson")
        if i == 2:
            axs[i, 0].set_ylabel("# of Samples", fontsize=fontsize)

        if i == 4:
            axs[i, 0].set_xlabel("Median", fontsize=fontsize)
            axs[i, 1].set_xlabel("IQ Range", fontsize=fontsize)

        axs[i, 1].hist(iqr_df.values[:, i], color="steelblue")
        axs[i, 1].set_ylabel(f"{keywords[i]}", fontsize=fontsize)
        axs[i, 1].yaxis.set_label_position("right")

    plt.savefig(save_path)
    plt.close(fig)


def data_analysis(config_datasets: pandas.DataFrame, graphs_folder: str, images_format: str):

    da_graphs_path = graphs_folder + "data_analysis/"
    if not os.path.exists(da_graphs_path):
        os.mkdir(da_graphs_path)

    dataset_ids = config_datasets['ID'].tolist()
    dataset_paths = config_datasets['PATH'].tolist()

    hist_save_path = da_graphs_path + "dataset_hist" + images_format
    if len(dataset_ids) > 1:
        make_dataset_hist(dataset_ids, dataset_paths, hist_save_path)
    else:
        make_dataset_hist_single(dataset_ids[0], dataset_paths[0], hist_save_path)

    for i in range(len(dataset_ids)):

        dataset_id = dataset_ids[i]
        dataset_path = dataset_paths[i]

        save_path = da_graphs_path + f"{dataset_id}_wordcloud" + images_format
        make_wordcloud(dataset_id, dataset_path, save_path)

    return


def sentiment_analysis(datasets_config: pandas.DataFrame, models_config: pandas.DataFrame, use_neutral: bool,
                       sentiment_folder: str, logs_folder: str, verbose: bool):

    dataset_ids = datasets_config["ID"].tolist()
    dataset_paths = datasets_config["PATH"].tolist()

    model_ids = models_config["ID"].tolist()
    model_links = models_config["LINK"].tolist()

    logger_path = logs_folder + "sentiment_analysis_log.csv"
    logger_exist = os.path.exists(logger_path)
    stream_logger, file_logger = get_loggers(logger_path)
    if not logger_exist:
        file_logger.info("DATASET,"
                         "MODEL,"
                         "TOTAL,"
                         "TOTAL_POSITIVE,"
                         "TOTAL_NEUTRAL,"
                         "TOTAL_NEGATIVE,"
                         "CORRECT,"
                         "CORRECT_POSITIVE,"
                         "CORRECT_NEUTRAL,"
                         "CORRECT_NEGATIVE")

    stream_logger.info("Executing Sentiment Analysis...")

    for df_index in range(len(dataset_ids)):

        dataset_id = dataset_ids[df_index]
        df = get_dataframe(dataset_ids[df_index], dataset_paths[df_index])

        df['sentiment'] = df.apply(get_sentiment_given_score, axis="columns")

        if not use_neutral:
            df = df[df['sentiment'] != 'neutral']

        total_negative = df[df['sentiment'] == "negative"].count().max()
        total_positive = df[df['sentiment'] == "positive"].count().max()
        total_neutral = df[df['sentiment'] == "neutral"].count().max()

        for model_index in range(len(model_ids)):

            model_id = model_ids[model_index]
            model_link = model_links[model_index]

            results_file_path = f"{sentiment_folder}{dataset_id}_{model_id}_neutral={use_neutral}.csv"

            stream_logger.info(f"Computing Results for Dataset: {dataset_id}, Model: {model_link}")

            correct_negative = 0
            correct_positive = 0
            correct_neutral = 0

            classifier = transformers.pipeline("zero-shot-classification", model=model_link, use_fast=False)
            if use_neutral:
                candidate_labels = ["negative", "positive", "neutral"]
            else:
                candidate_labels = ["negative", "positive"]

            results_dict = {
                'text': [],
                'target': [],
                'output': [],
                'time': []
            }

            sample_counter = 0
            for _, sample in df.iterrows():

                if verbose and sample_counter % 100 == 0:
                    stream_logger.info(f"Computing Sample {sample_counter}...")
                sample_counter += 1

                sequence_to_classify = sample['text']

                start = time.perf_counter()
                output = classifier(sequence_to_classify, candidate_labels, multi_label=False)
                end = time.perf_counter()

                label = output['labels'][np.array(output['scores']).argmax()]
                if label == sample['sentiment']:
                    if label == 'negative':
                        correct_negative += 1
                    elif label == 'positive':
                        correct_positive += 1
                    else:
                        correct_neutral += 1

                results_dict['text'].append(sequence_to_classify)
                results_dict['target'].append(sample['sentiment'])
                results_dict['output'].append(label)
                results_dict['time'].append(end - start)

            results_df = pandas.DataFrame(results_dict)
            results_df.to_csv(results_file_path, index=False)

            file_logger.info(f"{dataset_id},"
                             f"{model_id},"
                             f"{total_positive + total_neutral + total_negative},"
                             f"{total_positive},"
                             f"{total_neutral},"
                             f"{total_negative},"
                             f"{correct_positive + correct_neutral + correct_negative},"
                             f"{correct_positive},"
                             f"{correct_neutral},"
                             f"{correct_negative}")

    return


def keywords_extraction(datasets_config: pandas.DataFrame, models_config: pandas.DataFrame,
                        keywords_config: pandas.DataFrame, keyword_folder: str, verbose: bool):

    dataset_ids = datasets_config["ID"].tolist()
    dataset_paths = datasets_config["PATH"].tolist()

    model_ids = models_config["ID"].tolist()
    model_links = models_config["LINK"].tolist()

    stream_logger = get_stream_logger()
    stream_logger.info("Executing Keyword Extraction...")

    for df_index in range(len(dataset_ids)):

        dataset_id = dataset_ids[df_index]
        data_subfolder = f"{keyword_folder}{dataset_id}/"
        if not os.path.exists(data_subfolder):
            os.mkdir(data_subfolder)

        df = get_dataframe(dataset_ids[df_index], dataset_paths[df_index])

        # We select the subset of the keywords_config dataframe relevant for the dataset at hand.
        key_data_config = keywords_config[keywords_config['DATASET_ID'] == dataset_id]
        # And we extract the keywords_ids.
        keywords_ids = key_data_config['KEYWORDS_ID'].tolist()

        for keywords_id in keywords_ids:

            keyid_subfolder = f"{data_subfolder}{keywords_id}/"
            if not os.path.exists(keyid_subfolder):
                os.mkdir(keyid_subfolder)

            # We select the subset of the kewords_config dataframe relevant for the keyword_id at hand.
            keys_df = key_data_config[key_data_config['KEYWORDS_ID'] == keywords_id]
            # We drop the columns which do not contain keywords (we do this since in principle we could have more than
            # five keywords).
            keys_df = keys_df.drop(['DATASET_ID', 'KEYWORDS_ID'], axis=1)
            # We extract the list of keywords assuming that the selection by DATASEt_ID and KEYWORDS_ID has given us a
            # dataframe with a single row.
            keywords = keys_df.values.squeeze().tolist()

            for model_index in range(len(model_ids)):

                model_id = model_ids[model_index]
                model_link = model_links[model_index]

                stream_logger.info(f"Computing Results for Dataset: {dataset_id}, "
                                   f"Keywords ID: {keywords_id}, Model: {model_link}")

                classifier = transformers.pipeline("zero-shot-classification", model=model_link, use_fast=False)

                results_dict = {
                    'text': [],
                    'time': []
                }
                for keyword in keywords:
                    results_dict[keyword] = []

                sample_counter = 0
                for _, sample in df.iterrows():

                    if verbose and sample_counter % 100 == 0:
                        stream_logger.info(f"Computing Sample {sample_counter}...")
                    sample_counter += 1

                    sequence_to_classify = sample['text']

                    start = time.perf_counter()
                    output = classifier(sequence_to_classify, keywords, multi_label=True)
                    end = time.perf_counter()

                    results_dict['text'].append(sequence_to_classify)
                    results_dict['time'].append(end - start)

                    for i in range(len(output['labels'])):
                        results_dict[output['labels'][i]].append(output['scores'][i])

                results_df = pandas.DataFrame(results_dict)

                results_file_path = f"{keyid_subfolder}{dataset_id}_{keywords_id}_{model_id}.csv"
                results_df.to_csv(results_file_path, index=False)

    return


def analyse_key_ext_results(keywords_config: pandas.DataFrame, keyword_folder: str, graphs_folder: str,
                            images_format: str, verbose: bool):

    stream_logger = get_stream_logger()
    stream_logger.info("Executing Keyword Extraction Results Analysis...")

    data_key_couples = keywords_config[['DATASET_ID', 'KEYWORDS_ID']].values.tolist()

    for data_key_couple in data_key_couples:

        res_path = f"{keyword_folder}{data_key_couple[0]}/{data_key_couple[1]}/"
        subset_res_files = os.listdir(res_path)
        if len(subset_res_files) > 0:

            result_dfs = {res_file: pandas.read_csv(f"{res_path}{res_file}") for res_file in subset_res_files}
            num_samples = result_dfs[subset_res_files[0]].__len__()
            keywords = result_dfs[subset_res_files[0]].drop(['text', 'time'], axis=1).columns.to_list()

            res_by_sample = []
            median_by_sample = []
            iqr_by_sample = []
            for i in range(num_samples):

                if verbose and i % 100 == 0:
                    stream_logger.info(f"Computing Sample {i}...")

                temp_matrix = []
                review = None
                for res_file in subset_res_files:
                    temp_matrix.append(result_dfs[res_file].iloc[i].to_numpy()[2:])

                    if review is None:
                        review = result_dfs[res_file].iloc[i].text

                temp_matrix = np.array(temp_matrix, dtype=float)
                medians = np.median(temp_matrix, axis=0)
                iqr = scipy.stats.iqr(temp_matrix, axis=0)

                res_by_sample.append((review, temp_matrix))
                median_by_sample.append(medians)
                iqr_by_sample.append(iqr)

            median_by_sample = np.array(median_by_sample)
            iqr_by_sample = np.array(iqr_by_sample)

            median_df = pandas.DataFrame(data=median_by_sample, columns=keywords)
            iqr_df = pandas.DataFrame(data=iqr_by_sample, columns=keywords)

            median_df.to_csv(f"{res_path}{data_key_couple[0]}_{data_key_couple[1]}_median_df.csv", index=False)
            iqr_df.to_csv(f"{res_path}{data_key_couple[0]}_{data_key_couple[1]}_iqr_df.csv", index=False)

            res_hist_save_path = f"{graphs_folder}{data_key_couple[0]}_{data_key_couple[1]}_hist{images_format}"
            make_results_hist(median_df, iqr_df, keywords, res_hist_save_path)

    return

def temp():
    return