print("starting to run myattack.py")

import csv
import re
from collections import defaultdict
import numpy as np
# import matplotlib.pyplot as plt

def correlation():
    with open('cleaned_dixon_train_data.csv') as csv_file_2:
        csv_reader_2 = csv.reader(csv_file_2, delimiter=',')
        line_count_2 = 0

        naughty_count_list = []
        attack_list = []
        url_list = []
        norm_list = []

        for row in csv_reader_2:
            if line_count_2 == 0:
                print()
                print(row)
            else:
                # Create the naught_count array so we can calculate pearson's rank later
                naughty_count_list.append(float(row[5]))

                # Create the norm array so we can calculate pearson's rank later
                norm_list.append(float(row[7]))

                # Create the attack value array for pearson's rank calculations
                attack_list.append(int(float(row[3]) > 0.0))

                # Checking for URL and adding to url list
                if row[4] == "True":
                    url_list.append(1.0)
                else:
                    url_list.append(0.0)

            line_count_2 += 1

        print("lines:", line_count_2-1, "\n")

        # Correlation calculations
        print("attack - naughty_count")
        print(np.corrcoef(np.array(attack_list), np.array(naughty_count_list)))
        print()

        print("attack - contains_url")
        print(np.corrcoef(np.array(attack_list), np.array(url_list)))
        print()

        print("attack - norm")
        print(np.corrcoef(np.array(attack_list), np.array(norm_list)))
        print()


def get_bin_11(v):
    if v == 0.0:
        return 0
    elif v <= 0.1:
        return 1
    elif v <= 0.2:
        return 2
    elif v <= 0.3:
        return 3
    elif v <= 0.4:
        return 4
    elif v <= 0.5:
        return 5
    elif v <= 0.6:
        return 6
    elif v <= 0.7:
        return 7
    elif v <= 0.8:
        return 8
    elif v <= 0.9:
        return 9
    else:
        return 10


def get_bin_6(v):
    if v == 0.0:
        return 0
    elif v <= 0.2:
        return 1
    elif v <= 0.4:
        return 2
    elif v <= 0.6:
        return 3
    elif v <= 0.8:
        return 4
    else:
        return 5


def attack_dataset():
    # Get the naughty words
    naughty_words = []
    with open('naughty_words.txt') as text_file:
        for line in text_file:
            naughty_words.append(line.rstrip())

    with open('dixon_train_data.csv') as csv_file:
        csv_reader = csv.reader(csv_file, delimiter=',')
        line_count = 0
        http_row_count = 0
        blank_row_count = 0
        anomaly_count = 0
        total_attack_sum = 0
        attack_sum = 0
        greater_0 = 0
        attack_naughty_sum = 0
        attack_not_naughty_sum = 0
        naughty_greater_0 = 0
        naughty_equal_0 = 0
        greater_09 = 0
        attack_1 = 0
        no_naughty_attack_high = 0

        max_norm_naughty = 0
        max_norm_naughty_str = ""
        max_norm_naughty_attack = 0
        norm_sum = 0
        norm_naughty_sum = 0

        # titles: ['rev_id', 'comment', 'year', 'logged_in', 'ns', 'sample', 'split', 'attack']
        with open('cleaned_dixon_train_data.csv', mode='w') as csv_write_file:
            csv_writer = csv.writer(csv_write_file, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)

            naughty_dict = defaultdict(int)
            attack_dict = defaultdict(int)
            threshold_dict = defaultdict(float)
            length_dict = defaultdict(int)
            threshold_dict[0] = 0

            for row in csv_reader:

                is_corrupt = False
                contains_url = False
                naughty_count = 0
                bin_11 = 0

                if line_count == 0:
                    print("\n", row)
                    line_count += 1
                    csv_writer.writerow([row[0], row[1], row[6], row[7], "contains_URL",
                                         "naughty_count", "attack > 0.0", "attack > 0.4", "naughty_words_frequency",
                                         "bin_11", "bin_6"])
                else:
                    line_count += 1
                    total_attack_sum += float(row[7])

                    attack_val = float(row[7])
                    bin_11 = get_bin_11(attack_val)
                    bin_6 = get_bin_6(attack_val)


                    # to calculate average attack across comments with attack > 0
                    if float(row[7]) > 0:
                        greater_0 += 1
                        attack_sum += float(row[7])

                    # Checking for attacking threshold
                    t = 0
                    while t < 1.0:
                        if float(row[7]) < t:
                            threshold_dict[t] += 1
                        t += 0.01


                    # Checking for really bad comments
                    if float(row[7]) > 0.9:
                        # print(row[1])
                        greater_09 += 1

                    # Checking for the worst comments
                    if float(row[7]) == 1.0:
                        attack_1 += 1


                    # Check for web links
                    if re.search(r'http://', row[1]):
                        http_row_count += 1
                        contains_url = True

                    # rows with no text
                    if re.match(r'^\s+$', row[1]):
                        blank_row_count += 1
                        is_corrupt = True


                    # naughty word count, create dictionary for counting occurences with naughty_count naughty words
                    for n_word in naughty_words:
                        occurences = len(row[1].lower().split(n_word)) - 1
                        naughty_count += occurences
                    naughty_dict[naughty_count] += 1

                    # calculate average attack in comments with naughty words
                    if naughty_count > 0:
                        attack_naughty_sum += float(row[7])
                        naughty_greater_0 += 1
                    else:
                        attack_not_naughty_sum += float(row[7])
                        naughty_equal_0 += 1

                    # Calculating the normalised naughty count
                    comment_length = len((row[1].split(' ')))
                    length_dict[comment_length] += 1

                    norm = naughty_count / len((row[1].split(' ')))
                    norm_sum += norm
                    if norm > max_norm_naughty:
                        max_norm_naughty = norm
                        max_norm_naughty_str = row[1]
                        max_norm_naughty_attack = float(row[7])

                    if norm > 0:
                        norm_naughty_sum += norm


                    # dictionary for number of naughty words in high-attacking comments
                    if float(row[7]) > 0.4:
                        attack_dict[naughty_count] += 1

                    # print anomolies with lots of swears but not attacking.
                    if float(row[7]) < 0.4 and naughty_count > 20:
                        anomaly_count += 1

                    # count anomolies with no swear words but high attacking
                    if float(row[7]) > 0.4 and naughty_count == 0:
                        no_naughty_attack_high += 1


                    # Write to clean file, removing the apostrophes
                    if not is_corrupt:
                        no_punct_comment = re.sub('[":=]', '', row[1])
                        csv_writer.writerow([row[0], no_punct_comment.lower(), row[6], row[7],
                                             contains_url, naughty_count,
                                             int(float(row[7]) > 0.0), int(float(row[7]) > 0.4), norm, bin_11, bin_6])


                    # print current progress of lines processed
                    if line_count % 10000 == 0:
                        print(line_count)

            print("\nProcessed", line_count-1, "comments.")
            # print(no_naughty_attack_high, "comments with high attack value but no naughty words")
            # print(http_row_count, "comments with a URL")
            # print(blank_row_count, "rows with no text")
            # print(anomaly_count, "anomolies (no attack but lots of naughty words)")
            # print(greater_09, "comments with attack value > 0.9")
            # print(attack_1, "comments with attack value == 1.0")
            # print("average attack value is", total_attack_sum/69526, "across all comments")
            # print("average attack value is", attack_sum/greater_0, "across comments with attack > 0")
            # print("average attack value is", attack_naughty_sum/naughty_greater_0, "across comments with naughty words")
            # print("average attack value is", attack_not_naughty_sum/naughty_equal_0,
            #       "across comments with no naughty words\n")
            # print("Average norm value:", norm_sum/69526)
            # print("Average norm value in naughty word comments:", norm_naughty_sum/17205)
            # print("Maximum norm_naughty:", max_norm_naughty)
            # print("String:", max_norm_naughty_str)
            # print("Attack:", max_norm_naughty_attack, "\n")

            # # NAUGHTY DICT PLOT
            # plt.plot(naughty_dict.keys(), naughty_dict.values())
            # plt.axis([0, 20, 0, 60000])
            # plt.title("distribution of naughty words over all comments")
            # plt.savefig("Screenshots/naughty word dist. 1-20.png")
            # plt.show()
            #
            # # ATTACK DICT PLOT
            # plt.plot(attack_dict.keys(), attack_dict.values())
            # plt.axis([0, 20, 0, 5000])
            # plt.title("distribution of naughty words over the attacking comments")
            # plt.savefig("Screenshots/naughty word dist. 1-20 (attack > 0.4).png")
            # plt.show()
            #
            # # THRESHOLD PLOT
            # plt.plot(sorted(threshold_dict.keys()), sorted(threshold_dict.values()))
            # plt.title("Number of comments with attack under certain thresholds. N=69527")
            # plt.savefig("Screenshots/threshold graph.png")
            # plt.show()
            #
            # # LENGTH PLOT
            # plt.plot(sorted(length_dict.keys()), sorted(length_dict.values()))
            # plt.title("Number of comments with certain lengths. N=69527")
            # plt.savefig("Screenshots/length graph.png")
            # plt.show()
            #
            # # PRINTING DICTS
            # for key in sorted(naughty_dict):
            #     print("There are", naughty_dict[key], "comments with", key, "naughty words in.")
            #
            # print("---------------------")
            #
            # for key in sorted(attack_dict):
            #     print("There are", attack_dict[key], "attacking comments with", key, "naughty words in.")


def count_bins():
    print("counting bins")
    with open('cleaned_dixon_train_data.csv') as csv_file_3:
        csv_reader_3 = csv.reader(csv_file_3, delimiter=',')
        line_count_3 = 0

        # key is the upper bound for the bin
        bins = defaultdict(int)

        for row in csv_reader_3:
            if line_count_3 == 0:
                print()
                print(row)
            else:
                attack_val = float(row[3])
                if attack_val == 0.0:
                    bins[0.0] += 1
                elif attack_val <= 0.1:
                    bins[0.1] += 1
                elif attack_val <= 0.2:
                    bins[0.2] += 1
                elif attack_val <= 0.3:
                    bins[0.3] += 1
                elif attack_val <= 0.4:
                    bins[0.4] += 1
                elif attack_val <= 0.5:
                    bins[0.5] += 1
                elif attack_val <= 0.6:
                    bins[0.6] += 1
                elif attack_val <= 0.7:
                    bins[0.7] += 1
                elif attack_val <= 0.8:
                    bins[0.8] += 1
                elif attack_val <= 0.9:
                    bins[0.9] += 1
                else:
                    bins[1.0] += 1

            line_count_3 += 1

    print("processed", line_count_3-1, "clean lines")
    print(bins)

    # # BIN PLOT
    # x_data = []
    # y_data = []
    # for key in sorted(bins):
    #     x_data += [key]
    #     y_data += [bins[key]]
    #     print(key, bins[key])
    #
    # plt.bar(x_data, y_data, width=-0.05, align='center')
    # plt.axis([-0.1, 1.1, 0, 37000])
    # plt.title("Number of comments in attack value ranges")
    # plt.show()

# attack_dataset()
correlation()
# count_bins()