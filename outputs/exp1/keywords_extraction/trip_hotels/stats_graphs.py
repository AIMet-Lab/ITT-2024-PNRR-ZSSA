import matplotlib.pyplot as plt
import matplotlib
import pandas
import numpy as np

dataset_id = "trip_hotels"

key0_median = pandas.read_csv(f"expert/{dataset_id}_expert_median_df.csv")
key1_median = pandas.read_csv(f"cloud/{dataset_id}_cloud_median_df.csv")

key0_iqr = pandas.read_csv(f"expert/{dataset_id}_expert_iqr_df.csv")
key1_iqr = pandas.read_csv(f"cloud/{dataset_id}_cloud_iqr_df.csv")

keywords_0 = key0_median.columns.to_list()
keywords_1 = key1_median.columns.to_list()

key0_median = key0_median.to_numpy()
key1_median = key1_median.to_numpy()

key0_iqr = key0_iqr.to_numpy()
key1_iqr = key1_iqr.to_numpy()


n_bins = 10
legend_size = 25
matplotlib.rc('xtick', labelsize=20)
matplotlib.rc('ytick', labelsize=20)
fig, axs = plt.subplots(5, 2, figsize=(20, 26), tight_layout=True)

median_colors = ["darkgreen", "darkseagreen"]
iqr_colors = ["purple", "thistle"]

fontsize = 40
for i in range(5):

    stack_median = np.stack((key0_median[:, i], key1_median[:, i]), axis=1)
    stack_iqr = np.stack((key0_iqr[:, i], key1_iqr[:, i]), axis=1)

    labels = [keywords_0[i], keywords_1[i]]

    axs[i, 0].hist(stack_median, color=median_colors, label=labels)
    if i == 2:
        axs[i, 0].set_ylabel("# of Samples", fontsize=fontsize)

    if i == 4:
        axs[i, 0].set_xlabel("Median", fontsize=fontsize)
        axs[i, 1].set_xlabel("IQ Range", fontsize=fontsize)

    axs[i, 1].hist(stack_iqr, color=iqr_colors, label=labels)
    axs[i, 0].legend(prop={'size': legend_size})
    axs[i, 1].legend(prop={'size': legend_size})


plt.savefig(f"{dataset_id}_hist.pdf")
plt.close()