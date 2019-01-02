print("starting to run myattack.py")

import csv
import re
from collections import defaultdict
import numpy as np
import matplotlib.pyplot as plt

def correlation():
    with open('cleaned_formspring.csv') as csv_file_2:
        csv_reader_2 = csv.reader(csv_file_2, delimiter=',')
        line_count_2 = 0

        naughty_count_list = []
        label_list = []
        url_list = []
        norm_list = []

        for row in csv_reader_2:
            if line_count_2 == 0:
                print()
                print(row)
            else:
                # Create the naught_count array so we can calculate pearson's rank later
                naughty_count_list.append(float(row[3]))

                # Create the norm array so we can calculate pearson's rank later
                norm_list.append(float(row[4]))

                # Create the attack value array for pearson's rank calculations
                label_list.append(float(row[0]))

                # Checking for URL and adding to url list
                if row[2] == "True":
                    url_list.append(1.0)
                else:
                    url_list.append(0.0)
            line_count_2 += 1

        print(label_list.count(0.0))
        print(label_list.count(1.0))

        print("lines:", line_count_2-1, "\n")

        # Correlation calculations
        print("attack - naughty_count")
        print(np.corrcoef(np.array(label_list), np.array(naughty_count_list)))
        print()

        print("attack - contains_url")
        print(np.corrcoef(np.array(label_list), np.array(url_list)))
        print()

        print("attack - norm")
        print(np.corrcoef(np.array(label_list), np.array(norm_list)))
        print()


def formspring():
    # Get the naughty words
    naughty_words = []
    with open('naughty_words.txt') as text_file:
        for line in text_file:
            naughty_words.append(line.rstrip())

    with open('formspring.csv') as csv_file:
        csv_reader = csv.reader(csv_file, delimiter='\t')
        line_count = 0
        http_row_count = 0
        blank_row_count = 0

        max_norm_naughty = 0
        max_norm_naughty_str = ""
        max_norm_naughty_label = 0
        norm_sum = 0
        norm_naughty_sum = 0

        # titles: ['rev_id', 'comment', 'year', 'logged_in', 'ns', 'sample', 'split', 'attack']
        with open('cleaned_formspring.csv', mode='w') as csv_write_file:
            csv_writer = csv.writer(csv_write_file, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)

            naughty_dict = defaultdict(int)
            attack_dict = defaultdict(int)
            length_dict = defaultdict(int)
            label_count = defaultdict(int)
            label_count_naughty = defaultdict(int)
            label_count_not_naughty = defaultdict(int)

            for row in csv_reader:

                is_blank = False
                contains_url = False
                naughty_count = 0

                if line_count == 0:
                    print("\n", row)
                    line_count += 1
                    csv_writer.writerow(["label_bullying", "text_message", "contains_url", "naughty_count", "norm"])
                else:
                    line_count += 1

                    # Get the text from the input data file
                    post = row[1]
                    question = row[2]
                    answer = row[3]

                    # is this a cyberbullying example?
                    yes_array = ["Yes", "yes"]
                    ans1 = int(row[5] in yes_array)
                    severity1 = row[6]
                    ans2 = int(row[8] in yes_array)
                    severity2 = row[9]
                    ans3 = int(row[11] in yes_array)
                    severity3 = row[12]

                    # If 2 people say 'Yes' to bullying, then we class this instance as cyberbullying
                    if ans1 + ans2 + ans3 >= 2:
                        label_bullying = 1
                    else:
                        label_bullying = 0

                    # count the 1s and 0s in all messages
                    label_count[label_bullying] += 1

                    # Check for web links
                    if re.search(r'http://', row[1]):
                        http_row_count += 1
                        contains_url = True

                    # rows with no text
                    if re.match(r'^\s+$', row[1]):
                        blank_row_count += 1
                        is_blank = True

                    # naughty word count, create dictionary for counting occurences with naughty_count naughty words
                    for n_word in naughty_words:
                        occurences = len(post.lower().split(n_word)) - 1
                        naughty_count += occurences
                    naughty_dict[naughty_count] += 1

                    # distribution of naughty words over the bullying text messages
                    if label_bullying == 1:
                        attack_dict[naughty_count] += 1

                    # Calculating the normalised naughty count
                    comment_length = len((post.split(' ')))
                    length_dict[comment_length] += 1
                    norm = naughty_count / len((post.split(' ')))
                    if norm > max_norm_naughty:
                        max_norm_naughty = norm
                        max_norm_naughty_str = post
                        max_norm_naughty_label = label_bullying

                    # track the running total norm values (for all comments)
                    norm_sum += norm
                    # track the running total norm values (for just naughty comments)
                    if naughty_count > 0:
                        norm_naughty_sum += norm

                    if naughty_count > 0:
                        # count the 1s and 0s in naughty word messages
                        label_count_naughty[label_bullying] += 1
                    else:
                        # count the 1s and 0s in non-naughty word messages
                        label_count_not_naughty[label_bullying] += 1

                    # Write to clean file, removing the apostrophes
                    if not (is_blank):
                        clean_comment = re.sub('[\":=#&;()@\'?<>!,./\\\*\\n1234567890]', '', post)
                        clean_comment = re.sub('<br>', '', clean_comment)
                        csv_writer.writerow(
                            [label_bullying, clean_comment.lower(), contains_url, naughty_count, norm])

                    # print current progress of lines processed
                    if line_count % 1000 == 0:
                        print(line_count)

            print("\nProcessed", line_count - 1, "posts.")
            print("There are", label_count[1], "positive examples, and", label_count[0], "negative examples")
            print("There are", label_count_naughty[1], "positive examples, and", label_count_naughty[0],
                  "negative examples in naughty posts")
            print("There are", label_count_not_naughty[1], "positive examples, and", label_count_not_naughty[0],
                  "negative examples in non-naughty posts")
            print(http_row_count, "posts with a URL")
            print(blank_row_count, "rows with no text")
            print("Average norm value:", norm_sum / 12773)
            print("Average norm value in naughty word tweets:", norm_naughty_sum / 1965)
            print("Maximum norm:", max_norm_naughty)
            print("String:", max_norm_naughty_str)
            print("Label:", max_norm_naughty_label, "\n")

            # NAUGHTY DICT PLOT
            plt.plot(naughty_dict.keys(), naughty_dict.values())
            # plt.axis([0, 7, 0, 5000])
            plt.title("distribution of naughty words over all posts")
            plt.savefig("Screenshots/formspring_naughty word dist. 1-7.png")
            # plt.show()

            plt.close()

            # ATTACK DICT PLOT
            plt.plot(attack_dict.keys(), attack_dict.values())
            # plt.axis([0, 7, 0, 1200])
            plt.title("distribution of naughty words over the bullying posts")
            plt.savefig("Screenshots/formspring_naughty word dist. 1-7 (bullying instances).png")
            # plt.show()

            plt.close()

            # LENGTH PLOT
            plt.plot(sorted(length_dict.keys()), sorted(length_dict.values()))
            plt.title("Number of posts with certain lengths. N=12773")
            plt.savefig("Screenshots/formspring_length graph.png")
            # plt.show()

            # PRINTING DICTS
            for key in sorted(naughty_dict):
                print("There are", naughty_dict[key], "posts with", key, "naughty words in.")

            print("---------------------")

            for key in sorted(attack_dict):
                print("There are", attack_dict[key], "bullying posts with", key, "naughty words in.")


def tweets_1000():
    # Get the naughty words
    naughty_words = []
    with open('naughty_words.txt') as text_file:
        for line in text_file:
            naughty_words.append(line.rstrip())

    with open('twitter_dataset.csv') as csv_file:
        csv_reader = csv.reader(csv_file, delimiter=',')
        line_count = 0
        http_row_count = 0
        blank_row_count = 0

        max_norm_naughty = 0
        max_norm_naughty_str = ""
        max_norm_naughty_label = 0
        norm_sum = 0
        norm_naughty_sum = 0

        # titles: ['rev_id', 'comment', 'year', 'logged_in', 'ns', 'sample', 'split', 'attack']
        with open('cleaned_twitter_dataset.csv', mode='w') as csv_write_file:
            csv_writer = csv.writer(csv_write_file, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)

            naughty_dict = defaultdict(int)
            attack_dict = defaultdict(int)
            length_dict = defaultdict(int)
            label_count = defaultdict(int)
            label_count_naughty = defaultdict(int)
            label_count_not_naughty = defaultdict(int)

            for row in csv_reader:

                is_blank = False
                contains_url = False
                naughty_count = 0

                if line_count == 0:
                    print("\n", row)
                    line_count += 1
                    csv_writer.writerow(["label_bullying", "text_message", "contains_url", "naughty_count", "norm"])
                else:
                    line_count += 1
                    label_bullying = int(row[1] == "Bullying")
                    tweet = row[0]

                    # count the 1s and 0s in all messages
                    label_count[label_bullying] += 1

                    # Check for web links
                    if re.search(r'http://', row[1]):
                        http_row_count += 1
                        contains_url = True

                    # rows with no text
                    if re.match(r'^\s+$', row[1]):
                        blank_row_count += 1
                        is_blank = True

                    # naughty word count, create dictionary for counting occurences with naughty_count naughty words
                    for n_word in naughty_words:
                        occurences = len(tweet.lower().split(n_word)) - 1
                        naughty_count += occurences
                    naughty_dict[naughty_count] += 1

                    # distribution of naughty words over the bullying text messages
                    if label_bullying == 1:
                        attack_dict[naughty_count] += 1

                    # Calculating the normalised naughty count
                    comment_length = len((tweet.split(' ')))
                    length_dict[comment_length] += 1
                    norm = naughty_count / len((tweet.split(' ')))
                    if norm > max_norm_naughty:
                        max_norm_naughty = norm
                        max_norm_naughty_str = tweet
                        max_norm_naughty_label = label_bullying

                    # track the running total norm values (for all comments)
                    norm_sum += norm
                    # track the running total norm values (for just naughty comments)
                    if naughty_count > 0:
                        norm_naughty_sum += norm

                    if naughty_count > 0:
                        # count the 1s and 0s in naughty word messages
                        label_count_naughty[label_bullying] += 1
                    else:
                        # count the 1s and 0s in non-naughty word messages
                        label_count_not_naughty[label_bullying] += 1

                    # Write to clean file, removing the apostrophes
                    if not (is_blank):
                        no_punct_comment = re.sub('[\":=#&;\'?!,./\\\*\\n]', '', tweet)
                        csv_writer.writerow(
                            [label_bullying, no_punct_comment.lower(), contains_url, naughty_count, norm])

                    # print current progress of lines processed
                    if line_count % 1000 == 0:
                        print(line_count)

            print("\nProcessed", line_count - 1, "tweets.")
            print("There are", label_count[1], "positive examples, and", label_count[0], "negative examples")
            print("There are", label_count_naughty[1], "positive examples, and", label_count_naughty[0],
                  "negative examples in naughty tweets")
            print("There are", label_count_not_naughty[1], "positive examples, and", label_count_not_naughty[0],
                  "negative examples in non-naughty tweets")
            print(http_row_count, "tweets with a URL")
            print(blank_row_count, "rows with no text")
            print("Average norm value:", norm_sum / 8817)
            print("Average norm value in naughty word tweets:", norm_naughty_sum / 4477)
            print("Maximum norm:", max_norm_naughty)
            print("String:", max_norm_naughty_str)
            print("Label:", max_norm_naughty_label, "\n")

            # NAUGHTY DICT PLOT
            plt.plot(naughty_dict.keys(), naughty_dict.values())
            # plt.axis([0, 7, 0, 5000])
            plt.title("distribution of naughty words over all tweets")
            plt.savefig("Screenshots/twitter_naughty word dist. 1-7.png")
            # plt.show()

            plt.close()

            # ATTACK DICT PLOT
            plt.plot(attack_dict.keys(), attack_dict.values())
            # plt.axis([0, 7, 0, 1200])
            plt.title("distribution of naughty words over the bullying tweets")
            plt.savefig("Screenshots/twitter_naughty word dist. 1-7 (bullying instances).png")
            # plt.show()

            plt.close()

            # LENGTH PLOT
            plt.plot(sorted(length_dict.keys()), sorted(length_dict.values()))
            plt.title("Number of tweets with certain lengths. N=1065")
            plt.savefig("Screenshots/twitter_length graph.png")
            # plt.show()

            # PRINTING DICTS
            for key in sorted(naughty_dict):
                print("There are", naughty_dict[key], "tweets with", key, "naughty words in.")

            print("---------------------")

            for key in sorted(attack_dict):
                print("There are", attack_dict[key], "bullying tweets with", key, "naughty words in.")


def tweets_8000():
    # Get the naughty words
    naughty_words = []
    with open('naughty_words.txt') as text_file:
        for line in text_file:
            naughty_words.append(line.rstrip())

    with open('text_messages.csv') as csv_file:
        csv_reader = csv.reader(csv_file, delimiter=',')
        line_count = 0
        http_row_count = 0
        blank_row_count = 0

        max_norm_naughty = 0
        max_norm_naughty_str = ""
        max_norm_naughty_label = 0
        norm_sum = 0
        norm_naughty_sum = 0

        # titles: ['rev_id', 'comment', 'year', 'logged_in', 'ns', 'sample', 'split', 'attack']
        with open('cleaned_text_messages.csv', mode='w') as csv_write_file:
            csv_writer = csv.writer(csv_write_file, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)

            naughty_dict = defaultdict(int)
            attack_dict = defaultdict(int)
            length_dict = defaultdict(int)
            label_count = defaultdict(int)
            label_count_naughty = defaultdict(int)
            label_count_not_naughty = defaultdict(int)

            for row in csv_reader:

                is_blank = False
                contains_url = False
                is_NAME = False
                naughty_count = 0

                if line_count == 0:
                    print("\n", row)
                    line_count += 1
                    csv_writer.writerow(["label_bullying", "text_message", "contains_url", "naughty_count", "norm"])
                else:
                    line_count += 1
                    label_bullying = int(row[0])
                    text_message = row[1]

                    # count the 1s and 0s in all messages
                    label_count[label_bullying] += 1

                    # Check for web links
                    if re.search(r'http://', row[1]):
                        http_row_count += 1
                        contains_url = True

                    # rows with no text
                    if re.match(r'^\s+$', row[1]):
                        blank_row_count += 1
                        is_blank = True

                    name_count = 0
                    if text_message == "#NAME?":
                        name_count += 1
                        is_NAME = True

                    # naughty word count, create dictionary for counting occurences with naughty_count naughty words
                    for n_word in naughty_words:
                        occurences = len(text_message.lower().split(n_word)) - 1
                        naughty_count += occurences
                    naughty_dict[naughty_count] += 1

                    # distribution of naughty words over the bullying text messages
                    if label_bullying == 1:
                        attack_dict[naughty_count] += 1

                    # Calculating the normalised naughty count
                    comment_length = len((text_message.split(' ')))
                    length_dict[comment_length] += 1
                    norm = naughty_count / len((text_message.split(' ')))
                    if norm > max_norm_naughty:
                        max_norm_naughty = norm
                        max_norm_naughty_str = text_message
                        max_norm_naughty_label = label_bullying

                    # track the running total norm values (for all comments)
                    norm_sum += norm
                    # track the running total norm values (for just naughty comments)
                    if naughty_count > 0:
                        norm_naughty_sum += norm

                    if naughty_count > 0:
                        # count the 1s and 0s in naughty word messages
                        label_count_naughty[label_bullying] += 1
                    else:
                        # count the 1s and 0s in non-naughty word messages
                        label_count_not_naughty[label_bullying] += 1

                    # Write to clean file, removing the apostrophes
                    if not (is_blank) and not (is_NAME):
                        no_punct_comment = re.sub('[\":=#&;\'?!@,./\\\\\n*]', '', text_message)
                        csv_writer.writerow([label_bullying, no_punct_comment.lower(), contains_url, naughty_count, norm])

                    # print current progress of lines processed
                    if line_count % 1000 == 0:
                        print(line_count)

            print("\nProcessed", line_count-1, "tweets.")
            print("There are", label_count[1], "positive examples, and", label_count[0], "negative examples")
            print("There are", label_count_naughty[1], "positive examples, and", label_count_naughty[0], "negative examples in naughty tweets")
            print("There are", label_count_not_naughty[1], "positive examples, and", label_count_not_naughty[0], "negative examples in non-naughty tweets")
            print(http_row_count, "tweets with a URL")
            print(blank_row_count, "rows with no text")
            print("Average norm value:", norm_sum/8817)
            print("Average norm value in naughty word tweets:", norm_naughty_sum/4477)
            print("Maximum norm:", max_norm_naughty)
            print("String:", max_norm_naughty_str)
            print("Label:", max_norm_naughty_label, "\n")


            # NAUGHTY DICT PLOT
            plt.plot(naughty_dict.keys(), naughty_dict.values())
            plt.axis([0, 7, 0, 5000])
            plt.title("distribution of naughty words over all tweets")
            plt.savefig("Screenshots/tweets8000_naughty word dist. 1-7.png")
            # plt.show()

            plt.close()

            # ATTACK DICT PLOT
            plt.plot(attack_dict.keys(), attack_dict.values())
            plt.axis([0, 7, 0, 1200])
            plt.title("distribution of naughty words over the bullying tweets")
            plt.savefig("Screenshots/tweets8000_naughty word dist. 1-7 (bullying instances).png")
            # plt.show()

            plt.close()

            # LENGTH PLOT
            plt.plot(sorted(length_dict.keys()), sorted(length_dict.values()))
            plt.title("Number of messages with certain lengths. N=8817")
            plt.savefig("Screenshots/tweets8000_length graph.png")
            # plt.show()

            # PRINTING DICTS
            for key in sorted(naughty_dict):
                print("There are", naughty_dict[key], "tweets messages with", key, "naughty words in.")

            print("---------------------")

            for key in sorted(attack_dict):
                print("There are", attack_dict[key], "bullying messages with", key, "naughty words in.")


# formspring()
# tweets_8000()
# tweets_1000()
correlation()