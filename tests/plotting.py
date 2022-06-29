import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick
import os
import pdb

# order: Gaussian, Grassman, Laplace, HCF, CNN

# make table of general results


def print_result_table(bin_results, norm_results, approach, preprocess, base_dir):
    """
    """
    assert (len(bin_results[0]) == len(approach))
    assert (len(norm_results[0]) == len(approach))

    def process_results(av_acc, av_train_ts, av_test_ts, var_acc, var_train_ts, var_test_ts):
        """ format results for table """
        av_acc = [val * 100 for val in av_acc]
        var_acc = [val * 100 for val in var_acc]
        results = np.around(np.asarray(
            [av_acc, var_acc, av_train_ts, var_train_ts, av_test_ts, var_test_ts]),
                            decimals=2).T
        return results

    if False:
        approach = [
            'Gaussian: ', 'Grassman: ', 'Laplace: ', 'CNN Model 1: ', 'CNN Model 2: ', 'PCA KNN: ',
            'LocSVM16: ', 'LocSVM64: '
        ]

    def print_table(results):
        """ Print the table """
        for ii in range(len(approach)):
            if False:
                print(approach[ii] + ' & ' + str(results[ii][0][0]) + r'$\pm$' +
                      str(results[ii][0][1]) + ' & ' + str(results[ii][0][2]) + r'$\pm$' +
                      str(results[ii][0][3]) + ' & ' + str(results[ii][0][4]) + r'$\pm$' +
                      str(results[ii][0][5]) + r' \\ ')
            print(approach[ii] + ' & ' + str(results[ii][0]) + r'$\pm$' + str(results[ii][1]) +
                  ' & ' + str(results[ii][2]) + r'$\pm$' + str(results[ii][3]) + ' & ' +
                  str(results[ii][4]) + r'$\pm$' + str(results[ii][5]) + r' \\ ')
        return

    def save_table(results, file_name):
        """ Save the table """
        import sys
        original_stdout = sys.stdout
        file_path = os.path.join(base_dir, file_name)
        with open(file_name, 'w') as file:
            sys.stdout = file
            print_table(results)
            sys.stdout = original_stdout
        return

    # format the results properly
    bin_results = process_results(*bin_results)
    norm_results = process_results(*norm_results)

    # Print result tables to console
    print_table(bin_results)
    print_table(norm_results)

    # NEED TO CHECK THIS FUNCTION
    # save table to text file
    save_table(bin_results, f'test_results/experiment1/binary_{preprocess}_results_table.txt')
    save_table(norm_results, f'test_results/experiment1/normalized_{preprocess}_results_table.txt')
    return None


def print_table_6(scores, stds, method_names):
    scores = scores * 100
    stds = stds * 100
    print(r'\begin{tabular}{lllll}')
    print(r'    \toprule')
    print(r'    Method & 20\% train & 40\% train & 60\% train & 80\% train \\')
    print(r'    \midrule')
    for i in range(scores.shape[1]):
        print(f'    {method_names[i]} & ', end='')
        for j in range(scores.shape[0]):
            print(f'{scores[j, i]:.2f} $\\pm$ ', end='')
            print(f'{stds[j, i]:.2f} ', end='')
            if j < scores.shape[0] - 1:
                print(r'& ', end='')
        print(r'\\')
    print(r'    \bottomrule \\')
    print(r'\end{tabular}')


def print_table_8(norm, binary, method_names):
    norm_scores = norm[:, 0, :] * 100
    bin_scores = binary[:, 0, :] * 100
    norm_stds = norm[:, 3, ] * 100
    bin_stds = binary[:, 3, ] * 100
    norm_methods = [f'{m} (normalize)' for m in method_names]
    bin_methods = [f'{m} (binary)' for m in method_names]
    print(r'\begin{tabular}{lllllll}')

    data = [(norm_scores, norm_stds, norm_methods), (bin_scores, bin_stds, bin_methods)]

    for scores, stds, method_names in data:
        print(r'    \toprule')
        print(r'    Method & Test A & Test B & Test C & Test D & Test E & Test F \\')
        print(r'    \midrule')
        for i in range(scores.shape[1]):
            print(f'    {method_names[i]} & ', end='')
            for j in range(scores.shape[0]):
                print(f'{scores[j, i]:.2f} $\\pm$ ', end='')
                print(f'{stds[j, i]:.2f} ', end='')
                if j < scores.shape[0] - 1:
                    print(r'& ', end='')
            print(r'\\')
        print(r'    \bottomrule \\')

    print(r'\end{tabular}')


def plot_class_size_results(results, model_names, preprocess, base_dir):
    """
    """
    def get_scores(result):
        scores = np.zeros((len(result[0]), ))
        stds = np.zeros((len(result[0]), ))
        for ii in range(len(result[0])):
            scores[ii] = result[0][ii]
            stds[ii] = result[3][ii]
        return scores, stds

    def make_plot(scores, stds, model_names, preprocess, type_name):
        fig, ax = plt.subplots()
        # for ii in range(scores.shape[0]):
        #     ax.errorbar(np.arange(scores.shape[0]), scores[ii], yerr=stds[ii])
        for ii in range(scores.shape[1]):
            ax.errorbar(np.arange(scores.shape[0]), scores[:, ii], yerr=stds[:, ii])
            # ax.plot(np.arange(scores.shape[0]), scores[:,ii])

        # split_sizes = [0.2, 0.4, 0.6, 0.8]
        # Gaussian, Grassman, Laplace, HCF, CNN
        if False:
            method_names = [
                'Gaussian SVM {}'.format(type_name), 'Grassmann SVM {}'.format(type_name),
                'Laplace SVM {}'.format(type_name), 'CNN Model 1 {}'.format(type_name),
                'CNN Model 2 {}'.format(type_name), 'PCA KNN {}'.format(type_name),
                'Hermite SVM n=16', 'Hermite SVM n=64'
            ]

        method_names = [f'{s} ({type_name})' for s in model_names]

        ax.legend(method_names, ncol=2, loc='lower right')
        ax.set_ylim(0.7, 1)
        ax.set_yticks(np.linspace(0.7, 1, 7))
        ax.set_xticks(np.arange(0, scores.shape[0]))
        ax.set_xticklabels(["20%", "40%", "60%", "80%"])
        ax.set_xlabel("Training ratio")
        ax.set_ylabel("Test accuracy (%)")
        ax.yaxis.set_major_formatter(mtick.PercentFormatter(1))

        # plt.grid(True, which="both", axis="y")
        plt.tight_layout()
        file_name = os.path.join(base_dir, f'scores_vs_split_size_{preprocess}_{type_name}.png')
        # print()
        # print_table_6(scores, stds, method_names)
        plt.savefig(file_name)
        return

    assert (len(results[0][0][0]) == len(model_names))

    all_norm_scores = np.zeros((len(results), len(results[0][0][0])))
    all_bin_scores = np.zeros((len(results), len(results[0][0][0])))
    all_norm_stds = np.zeros((len(results), len(results[0][0][0])))
    all_bin_stds = np.zeros((len(results), len(results[0][0][0])))
    for ii in range(len(results)):
        norm_scores, norm_stds = get_scores(results[ii][0])
        bin_scores, bin_stds = get_scores(results[ii][1])
        all_norm_scores[ii, :] = norm_scores
        all_bin_scores[ii, :] = bin_scores
        all_norm_stds[ii, :] = norm_stds
        all_bin_stds[ii, :] = bin_stds
    make_plot(all_norm_scores, all_norm_stds, model_names, preprocess, 'normalized')
    make_plot(all_bin_scores, all_bin_stds, model_names, preprocess, 'binary')
    return


def make_single_person_plots(norm_results, bin_results, preprocess, method_names, base_dir):
    """
    """
    def set_decimal_place(results):
        acc_SVDG = np.around(np.asarray(results[0]), decimals=2)
        acc_GG = np.around(np.asarray(results[1]), decimals=2)
        acc_SVDL = np.around(np.asarray(results[2]), decimals=2)
        acc_CNN1 = np.around(np.asarray(results[3]), decimals=2)
        acc_CNN2 = np.around(np.asarray(results[4]), decimals=2)
        acc_CNN3 = np.around(np.asarray(results[4]), decimals=2)
        return acc_SVDG, acc_GG, acc_SVDL, acc_CNN1, acc_CNN2, acc_CNN3

    def plot_results(results, type_name, preprocess, model_names):
        # acc_SVDG, acc_GG, acc_SVDL, acc_CNN1, acc_CNN2, acc_CNN3 = set_decimal_place(norm_results)

        # method_names = [
        #     'Gaussian SVM {}'.format(type_name), 'Grassmann SVM {}'.format(type_name),
        #     'Laplace SVM {}'.format(type_name), 'CNN Model 1  {}'.format(type_name),
        #     'CNN Model 2  {}'.format(type_name), 'CNN Model 3  {}'.format(type_name)
        # ]

        fig, ax = plt.subplots()
        for ii in range(len(model_names)):
            ax.errorbar(np.arange(6), results[:, 0, ii], yerr=results[:, 3, ii])

        plt.xticks([0, 1, 2, 3, 4, 5], ['A', 'B', 'C', 'D', 'E', 'F'])
        plt.ylim(0, 1)
        # ax.legend(method_names, ncol=2)
        method_names = [f'{s} ({type_name})' for s in model_names]
        ax.legend(method_names, loc='lower right')

        # ax.set_xticklabels(['A', 'B', 'C', 'D', 'E', 'F'])
        ax.set_xlabel("Person tested")
        ax.set_ylabel("Test accuracy (%)")
        ax.yaxis.set_major_formatter(mtick.PercentFormatter(1))
        file_name = os.path.join(base_dir, "{}_{}_person_vs_all".format(type_name, preprocess))
        plt.savefig(file_name)
        return

    plot_results(norm_results, 'normalize', preprocess, method_names)
    plot_results(bin_results, 'binary', preprocess, method_names)
    print_table_8(norm_results, bin_results, method_names)
    return
