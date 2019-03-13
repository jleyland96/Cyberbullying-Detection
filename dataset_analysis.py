print("starting to run dataset_analysis.py")

import csv
import re
from collections import defaultdict
import numpy as np
import matplotlib.pyplot as plt
import pickle

# GLOVE. Create dictionary where keys are words and the values are the vectors for the words
# print("getting GLOVE embeddings size 300...")
# file = open('glove.6B/glove.6B.300d.txt', "r").readlines()
# gloveDict = {}
# for line in file:
#     info = line.split(' ')
#     key = info[0]
#     vec = []
#     for elem in info[1:]:
#         vec.append(elem.rstrip())
#     gloveDict[key] = vec
# print(len(gloveDict), "words in the GLOVE dictionary\n")


def correlation():
    with open('cleaned_dixon.csv') as csv_file_2:
        csv_reader_2 = csv.reader(csv_file_2, delimiter=',')
        line_count_2 = 0

        naughty_count_list = []
        label_list = []
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
            line_count_2 += 1

        print(label_list.count(0.0))
        print(label_list.count(1.0))

        print("lines:", line_count_2-1, "\n")

        # Correlation calculations
        print("attack - naughty_count")
        print(np.corrcoef(np.array(label_list), np.array(naughty_count_list)))
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
        length_running_total = 0

        max_norm_naughty = 0
        max_norm_naughty_str = ""
        max_norm_naughty_label = 0
        norm_sum = 0
        norm_naughty_sum = 0

        glove_running_total_before = 0
        glove_zero_count_before = 0
        glove_running_total_after = 0
        glove_zero_count_after = 0
        max_glove_len = 0

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
                    length_running_total += comment_length
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

                    # ------ GLOVE -------
                    # for each word in this sentence
                    count = 0
                    for word in post.split(' '):
                        # if the word is in our gloveDict, then add element-wise to our output X
                        if word in gloveDict:
                            count += 1

                    glove_running_total_before += count
                    if count == 0:
                        glove_zero_count_before += 1
                    # ----- END GLOVE ----

                    clean_comment = re.sub('<br>', ' ', post)
                    clean_comment = re.sub('[\":=#&;()@\'?<>!,./\\\*\\n1234567890]', '', clean_comment)

                    # ------ GLOVE -------
                    # for each word in this sentence
                    count = 0
                    for word in clean_comment.lower().split(' '):
                        # if the word is in our gloveDict, then add element-wise to our output X
                        if word in gloveDict:
                            count += 1

                    glove_running_total_after += count
                    if count == 0:
                        glove_zero_count_after += 1

                    if count > max_glove_len:
                        max_glove_len = count
                    # ----- END GLOVE ----

                    # Write to clean file, removing the apostrophes
                    if not(is_blank) and count>0:
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
            print("Label:", max_norm_naughty_label)
            print("Average tweet length:", length_running_total / 12773)
            print("Average glove count before:", glove_running_total_before / 12773)
            print("Glove zero count before:", glove_zero_count_before)
            print("Average glove count after:", glove_running_total_after / 12773)
            print("Glove zero count after:", glove_zero_count_after)
            print("Max glove count:", max_glove_len)

            print(max(list(length_dict.keys())))

            # NAUGHTY DICT PLOT
            # plt.close()
            # plt.plot(naughty_dict.keys(), naughty_dict.values())
            # # plt.axis([0, 7, 0, 5000])
            # plt.title("distribution of naughty words over all posts")
            # plt.savefig("Screenshots/formspring_naughty word dist. 1-7.png")
            # # plt.show()
            #
            # plt.close()
            #
            # # ATTACK DICT PLOT
            # plt.plot(attack_dict.keys(), attack_dict.values())
            # # plt.axis([0, 7, 0, 1200])
            # plt.title("distribution of naughty words over the bullying posts")
            # plt.savefig("Screenshots/formspring_naughty word dist. 1-7 (bullying instances).png")
            # # plt.show()
            #
            # plt.close()
            #
            # # LENGTH PLOT
            # print(length_dict)
            # length_keys = list(sorted(length_dict.keys()))
            # length_values = []
            # for i in length_keys:
            #     length_values.append(length_dict[i])
            # print(length_keys)
            # print(length_values)
            #
            # plt.plot(length_keys, length_values)
            # plt.xlabel("length")
            # plt.ylabel("frequency")
            # plt.title("Number of tweets with certain lengths. N=1065")
            # plt.savefig("Screenshots/formspring_length graph.png")

            # # PRINTING DICTS
            # for key in sorted(naughty_dict):
            #     print("There are", naughty_dict[key], "posts with", key, "naughty words in.")
            #
            # print("---------------------")
            #
            # for key in sorted(attack_dict):
            #     print("There are", attack_dict[key], "bullying posts with", key, "naughty words in.")


def tweets_1000():
    # Get the naughty words
    naughty_words = []
    with open('naughty_words.txt') as text_file:
        for line in text_file:
            naughty_words.append(line.rstrip())

    with open('twitter_1K.csv') as csv_file:
        csv_reader = csv.reader(csv_file, delimiter=',')
        line_count = 0
        http_row_count = 0
        blank_row_count = 0

        max_norm_naughty = 0
        max_norm_naughty_str = ""
        max_norm_naughty_label = 0
        norm_sum = 0
        norm_naughty_sum = 0
        length_running_total = 0

        glove_running_total_before = 0
        glove_zero_count_before = 0
        glove_running_total_after = 0
        glove_zero_count_after = 0
        max_glove_len = 0

        # titles: ['rev_id', 'comment', 'year', 'logged_in', 'ns', 'sample', 'split', 'attack']
        with open('cleaned_twitter_1K.csv', mode='w') as csv_write_file:
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
                    length_running_total += comment_length
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

                    # ------ GLOVE -------
                    # for each word in this sentence
                    count = 0
                    for word in tweet.split(' '):
                        # if the word is in our gloveDict, then add element-wise to our output X
                        if word in gloveDict:
                            count += 1

                    glove_running_total_before += count
                    if count == 0:
                        glove_zero_count_before += 1
                    # ----- END GLOVE ----

                    no_punct_comment = re.sub('[\":=#&;\'?!,./\\\*\\n]', '', tweet)

                    # ------ GLOVE -------
                    # for each word in this sentence
                    count = 0
                    for word in no_punct_comment.lower().split(' '):
                        # if the word is in our gloveDict, then add element-wise to our output X
                        if word in gloveDict:
                            count += 1

                    glove_running_total_after += count
                    if count == 0:
                        glove_zero_count_after += 1

                    if count > max_glove_len:
                        max_glove_len = count
                    # ----- END GLOVE ----

                    # Write to clean file, removing the apostrophes
                    if not (is_blank) and count>0:
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
            print("Average norm value:", norm_sum / 1065)
            print("Average norm value in naughty word tweets:", norm_naughty_sum / 471)
            print("Maximum norm:", max_norm_naughty)
            print("String:", max_norm_naughty_str)
            print("Label:", max_norm_naughty_label)
            print("Average tweet length:", length_running_total / 1065)
            print("Average glove count before:", glove_running_total_before / 1065)
            print("Glove zero count before:", glove_zero_count_before)
            print("Average glove count after:", glove_running_total_after / 1065)
            print("Glove zero count after:", glove_zero_count_after)
            print("Max glove count:", max_glove_len)

            print(max(list(length_dict.keys())))

            # # NAUGHTY DICT PLOT
            # plt.close()
            # plt.plot(naughty_dict.keys(), naughty_dict.values())
            # # plt.axis([0, 12, 0, 700])
            # plt.title("distribution of naughty words over all Tweets")
            # plt.savefig("Screenshots/twitter_naughty word dist. 1-7.png")
            # # plt.show()
            #
            # plt.close()
            #
            # # ATTACK DICT PLOT
            # plt.plot(attack_dict.keys(), attack_dict.values())
            # # plt.axis([0, 12, 0, 200])
            # plt.title("distribution of naughty words over the bullying tweets")
            # plt.savefig("Screenshots/twitter_naughty word dist. 1-7 (bullying instances).png")
            # # plt.show()
            #
            # plt.close()
            #
            # # LENGTH PLOT
            # print(length_dict)
            # length_keys = list(sorted(length_dict.keys()))
            # length_values = []
            # for i in length_keys:
            #     length_values.append(length_dict[i])
            # print(length_keys)
            # print(length_values)
            #
            # plt.plot(length_keys, length_values)
            # plt.xlabel("length")
            # plt.ylabel("frequency")
            # plt.title("Number of tweets with certain lengths. N=1065")
            # plt.savefig("Screenshots/twitter_length graph.png")
            #
            # plt.close()

            # # PRINTING DICTS
            # for key in sorted(naughty_dict):
            #     print("There are", naughty_dict[key], "tweets with", key, "naughty words in.")
            #
            # print("---------------------")
            #
            # for key in sorted(attack_dict):
            #     print("There are", attack_dict[key], "bullying tweets with", key, "naughty words in.")


def dixon():
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
        length_running_total = 0

        max_norm_naughty = 0
        max_norm_naughty_str = ""
        max_norm_naughty_label = 0
        norm_sum = 0
        norm_naughty_sum = 0

        glove_running_total_before = 0
        glove_misses_before = 0
        glove_zero_count_before = 0
        glove_running_total_after = 0
        glove_misses_after = 0
        glove_zero_count_after = 0
        max_glove_len = 0

        # titles: ['rev_id', 'comment', 'year', 'logged_in', 'ns', 'sample', 'split', 'attack']
        with open('cleaned_dixon.csv', mode='w') as csv_write_file:
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
                    label_bullying = int(float(row[7]) > 0.0)
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
                    length_running_total += comment_length
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

                    # ------ GLOVE -------
                    # for each word in this sentence
                    count = 0
                    for word in text_message.split(' '):
                        # if the word is in our gloveDict, then add element-wise to our output X
                        if word in gloveDict:
                            count += 1
                        else:
                            glove_misses_before += 1

                    glove_running_total_before += count
                    if count == 0:
                        glove_zero_count_before += 1
                    # ----- END GLOVE ----

                    # remove URLs
                    no_punct_comment = re.sub(r'http\S+', '', text_message)
                    # remove punctuation
                    no_punct_comment = re.sub('[\":=#&;\'?!@,./\\\\\n*]', '', no_punct_comment)
                    # remove multiple spaces, replace with one space
                    no_punct_comment = re.sub(' +', ' ', no_punct_comment)

                    # ------ GLOVE -------
                    # for each word in this sentence
                    count = 0
                    for word in no_punct_comment.lower().split(' '):
                        # if the word is in our gloveDict, then add element-wise to our output X
                        if word in gloveDict:
                            count += 1
                        else:
                            glove_misses_after += 1

                    glove_running_total_after += count
                    if count == 0:
                        glove_zero_count_after += 1

                    if count > max_glove_len:
                        max_glove_len = count
                    # ----- END GLOVE ----

                    # Write to clean file, removing the apostrophes
                    if not (is_blank) and not (is_NAME) and count > 0:
                        csv_writer.writerow(
                            [label_bullying, no_punct_comment.lower(), contains_url, naughty_count, norm])

                    # print current progress of lines processed
                    if line_count % 1000 == 0:
                        print(line_count)

            # remove mentions
            # remove RT tags
            # remove not in glove embedding. If not, see if stretched. Remove repeated letters. Remove if still no.

            print("\nProcessed", line_count-1, "tweets.")
            print("There are", label_count[1], "positive examples, and", label_count[0], "negative examples")
            print("There are", label_count_naughty[1], "positive examples, and", label_count_naughty[0], "negative examples in naughty tweets")
            print("There are", label_count_not_naughty[1], "positive examples, and", label_count_not_naughty[0], "negative examples in non-naughty tweets")
            print(http_row_count, "tweets with a URL")
            print(blank_row_count, "rows with no text")
            print("Average norm value:", norm_sum/8817)
            print("Average norm value in naughty word tweets:", norm_naughty_sum/4340)
            print("Maximum norm:", max_norm_naughty)
            print("String:", max_norm_naughty_str)
            print("Label:", max_norm_naughty_label)
            print("Average tweet length:", length_running_total/8817)
            print("Average glove count before:", glove_running_total_before / 8817)
            print("Glove word hits before:", glove_running_total_before)
            print("Glove word misses before:", glove_misses_before)
            print("Glove zero count before:", glove_zero_count_before)
            print("Average glove count after:", glove_running_total_after / 8817)
            print("Glove word hits after:", glove_running_total_after)
            print("Glove word misses after:", glove_misses_after)
            print("Glove zero count after:", glove_zero_count_after)
            print("Max glove count:", max_glove_len)

            print(max(list(length_dict.keys())))

            # plt.plot(naughty_dict.keys(), naughty_dict.values())
            # plt.axis([0, 7, 0, 5000])
            # plt.xlabel("number of naughty words")
            # plt.ylabel("frequency")
            # plt.title("distribution of naughty words over tweets/naughty texts")
            # plt.savefig("Screenshots/tweets8000_naughty word dist. 1-7.png")
            # # plt.show()
            #
            # plt.close()
            #
            # # ATTACK DICT PLOT
            # plt.plot(attack_dict.keys(), attack_dict.values())
            # plt.axis([0, 7, 0, 1200])
            # plt.xlabel("number of naughty words")
            # plt.ylabel("frequency")
            # plt.title("distribution of naughty words over the bullying texts")
            # plt.savefig("Screenshots/tweets8000_naughty word dist. 1-7 (bullying instances).png")
            # # plt.show()
            #
            # plt.close()
            #
            print(length_dict)
            length_keys = list(sorted(length_dict.keys()))
            length_values = []
            for i in length_keys:
                length_values.append(length_dict[i])
            print(length_keys)
            print(length_values)

            plt.plot(length_keys, length_values)
            plt.xlabel("length")
            plt.ylabel("frequency")
            plt.title("Number of texts with certain lengths. N=" + str(line_count-1))
            plt.savefig("Screenshots/dixon_length graph.png")

            # # PRINTING DICTS
            # for key in sorted(naughty_dict):
            #     print("There are", naughty_dict[key], "tweets messages with", key, "naughty words in.")
            #
            # print("---------------------")
            #
            # for key in sorted(attack_dict):
            #     print("There are", attack_dict[key], "bullying messages with", key, "naughty words in.")


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
        length_running_total = 0

        max_norm_naughty = 0
        max_norm_naughty_str = ""
        max_norm_naughty_label = 0
        norm_sum = 0
        norm_naughty_sum = 0

        glove_running_total_before = 0
        glove_misses_before = 0
        glove_zero_count_before = 0
        glove_running_total_after = 0
        glove_misses_after = 0
        glove_zero_count_after = 0
        max_glove_len = 0

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
                    length_running_total += comment_length
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

                    # ------ GLOVE -------
                    # for each word in this sentence
                    count = 0
                    for word in text_message.split(' '):
                        # if the word is in our gloveDict, then add element-wise to our output X
                        if word in gloveDict:
                            count += 1
                        else:
                            glove_misses_before += 1

                    glove_running_total_before += count
                    if count == 0:
                        glove_zero_count_before += 1
                    # ----- END GLOVE ----

                    # remove punctuation
                    no_punct_comment = re.sub('[\":=#&;\'?!@,./\\\\\n*]', '', text_message)
                    # remove multiple spaces, replace with one space
                    no_punct_comment = re.sub(' +', ' ', no_punct_comment)

                    # ------ GLOVE -------
                    # for each word in this sentence
                    count = 0
                    for word in no_punct_comment.lower().split(' '):
                        # if the word is in our gloveDict, then add element-wise to our output X
                        if word in gloveDict:
                            count += 1
                        else:
                            glove_misses_after += 1

                    glove_running_total_after += count
                    if count == 0:
                        glove_zero_count_after += 1

                    if count > max_glove_len:
                        max_glove_len = count
                    # ----- END GLOVE ----

                    # Write to clean file, removing the apostrophes
                    if not (is_blank) and not (is_NAME) and count > 0:
                        csv_writer.writerow(
                            [label_bullying, no_punct_comment.lower(), contains_url, naughty_count, norm])

                    # print current progress of lines processed
                    if line_count % 1000 == 0:
                        print(line_count)

            # remove mentions
            # remove RT tags
            # remove not in glove embedding. If not, see if stretched. Remove repeated letters. Remove if still no.

            print("\nProcessed", line_count-1, "tweets.")
            print("There are", label_count[1], "positive examples, and", label_count[0], "negative examples")
            print("There are", label_count_naughty[1], "positive examples, and", label_count_naughty[0], "negative examples in naughty tweets")
            print("There are", label_count_not_naughty[1], "positive examples, and", label_count_not_naughty[0], "negative examples in non-naughty tweets")
            print(http_row_count, "tweets with a URL")
            print(blank_row_count, "rows with no text")
            print("Average norm value:", norm_sum/8817)
            print("Average norm value in naughty word tweets:", norm_naughty_sum/4340)
            print("Maximum norm:", max_norm_naughty)
            print("String:", max_norm_naughty_str)
            print("Label:", max_norm_naughty_label)
            print("Average tweet length:", length_running_total/8817)
            print("Average glove count before:", glove_running_total_before / 8817)
            print("Glove word hits before:", glove_running_total_before)
            print("Glove word misses before:", glove_misses_before)
            print("Glove zero count before:", glove_zero_count_before)
            print("Average glove count after:", glove_running_total_after / 8817)
            print("Glove word hits after:", glove_running_total_after)
            print("Glove word misses after:", glove_misses_after)
            print("Glove zero count after:", glove_zero_count_after)
            print("Max glove count:", max_glove_len)

            print(max(list(length_dict.keys())))

            # plt.plot(naughty_dict.keys(), naughty_dict.values())
            # plt.axis([0, 7, 0, 5000])
            # plt.xlabel("number of naughty words")
            # plt.ylabel("frequency")
            # plt.title("distribution of naughty words over tweets/naughty texts")
            # plt.savefig("Screenshots/tweets8000_naughty word dist. 1-7.png")
            # # plt.show()
            #
            # plt.close()
            #
            # # ATTACK DICT PLOT
            # plt.plot(attack_dict.keys(), attack_dict.values())
            # plt.axis([0, 7, 0, 1200])
            # plt.xlabel("number of naughty words")
            # plt.ylabel("frequency")
            # plt.title("distribution of naughty words over the bullying texts")
            # plt.savefig("Screenshots/tweets8000_naughty word dist. 1-7 (bullying instances).png")
            # # plt.show()
            #
            # plt.close()
            #
            # print(length_dict)
            # length_keys = list(sorted(length_dict.keys()))
            # length_values = []
            # for i in length_keys:
            #     length_values.append(length_dict[i])
            # print(length_keys)
            # print(length_values)
            #
            # plt.plot(length_keys, length_values)
            # plt.xlabel("length")
            # plt.ylabel("frequency")
            # plt.title("Number of texts with certain lengths. N=8817")
            # plt.savefig("Screenshots/tweets8000_length graph.png")

            # # PRINTING DICTS
            # for key in sorted(naughty_dict):
            #     print("There are", naughty_dict[key], "tweets messages with", key, "naughty words in.")
            #
            # print("---------------------")
            #
            # for key in sorted(attack_dict):
            #     print("There are", attack_dict[key], "bullying messages with", key, "naughty words in.")


def tweets_16k():
    # NOTE:
    # this method also removes any comments that don't have any glove hits at all

    # Get the naughty words
    naughty_words = []
    with open('naughty_words.txt') as text_file:
        for line in text_file:
            naughty_words.append(line.rstrip())

    with open('twitter_16K.csv') as csv_file:
        csv_reader = csv.reader(csv_file, delimiter='\t')
        line_count = 0
        http_row_count = 0
        blank_row_count = 0
        length_running_total = 0

        max_norm_naughty = 0
        max_norm_naughty_str = ""
        max_norm_naughty_label = 0
        norm_sum = 0
        norm_naughty_sum = 0

        glove_running_total_before = 0
        glove_misses_before = 0
        glove_zero_count_before = 0
        glove_running_total_after = 0
        glove_misses_after = 0
        glove_zero_count_after = 0
        max_glove_len = 0

        # titles: ['rev_id', 'comment', 'year', 'logged_in', 'ns', 'sample', 'split', 'attack']
        with open('processed_tweets_16k_3class_aaaaa.csv', mode='w') as csv_write_file:
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
                    csv_writer.writerow(["label_bullying", "tweet", "contains_url", "naughty_count", "norm"])

                line_count += 1
                tweet = row[1]
                label = row[2]

                if label == "none":
                    label_bullying = 0
                elif label == "racism":
                    label_bullying = 1
                else:
                    label_bullying = 2

                # count the 1s and 0s in all messages
                label_count[label_bullying] += 1

                # Check for web links
                if re.search(r'http://', row[1]):
                    http_row_count += 1
                    contains_url = True

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
                if comment_length > 100:
                    print(tweet)
                length_dict[comment_length] += 1
                length_running_total += comment_length
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

                # # ------ GLOVE -------
                # # for each word in this sentence
                # count = 0
                # for word in tweet.split(' '):
                #     # if the word is in our gloveDict, then add element-wise to our output X
                #     if word in gloveDict:
                #         count += 1
                #     else:
                #         glove_misses_before += 1
                #
                # glove_running_total_before += count
                # if count == 0:
                #     glove_zero_count_before += 1
                # # ----- END GLOVE ----

                # remove punctuation
                no_punc_tweet = re.sub('[\":=#&;\'?!@,./\\\\\n*]', '', tweet)
                # remove multiple spaces, replace with one space
                no_punc_tweet = re.sub(' +', ' ', no_punc_tweet)

                # # ------ GLOVE -------
                # # for each word in this sentence
                # count = 0
                # for word in no_punc_tweet.lower().split(' '):
                #     # if the word is in our gloveDict, then add element-wise to our output X
                #     if word in gloveDict:
                #         count += 1
                #     else:
                #         glove_misses_after += 1
                #
                # glove_running_total_after += count
                # if count == 0:
                #     glove_zero_count_after += 1
                #
                # if count > max_glove_len:
                #     max_glove_len = count
                # # ----- END GLOVE ----

                # Write to clean file, removing the apostrophes
                # if not (is_blank) and not (is_NAME) and count > 0:
                if not (is_blank) and not (is_NAME) and count>0:
                    csv_writer.writerow(
                        [label_bullying, tweet, contains_url, naughty_count, norm])

                # print current progress of lines processed
                if line_count % 1000 == 0:
                    print(line_count)

            # remove mentions
            # remove RT tags
            # remove not in glove embedding. If not, see if stretched. Remove repeated letters. Remove if still no.

            print("\nProcessed", line_count - 1, "tweets.")
            print("There are", label_count[1], "positive examples, and", label_count[0], "negative examples")
            print("There are", label_count_naughty[1], "positive examples, and", label_count_naughty[0],
                  "negative examples in naughty tweets")
            print("There are", label_count_not_naughty[1], "positive examples, and", label_count_not_naughty[0],
                  "negative examples in non-naughty tweets")
            print(http_row_count, "tweets with a URL")
            print(blank_row_count, "rows with no text")
            print("Average norm value:", norm_sum / 8817)
            print("Average norm value in naughty word tweets:", norm_naughty_sum / 4340)
            print("Maximum norm:", max_norm_naughty)
            print("String:", max_norm_naughty_str)
            print("Label:", max_norm_naughty_label)
            print("Average tweet length:", length_running_total / 8817)
            print("Average glove count before:", glove_running_total_before / 8817)
            print("Glove word hits before:", glove_running_total_before)
            print("Glove word misses before:", glove_misses_before)
            print("Glove zero count before:", glove_zero_count_before)
            print("Average glove count after:", glove_running_total_after / 8817)
            print("Glove word hits after:", glove_running_total_after)
            print("Glove word misses after:", glove_misses_after)
            print("Glove zero count after:", glove_zero_count_after)
            print("Max glove count:", max_glove_len)

            print(max(list(length_dict.keys())))

            print(label_count)

            # plt.plot(naughty_dict.keys(), naughty_dict.values())
            # plt.axis([0, 7, 0, 5000])
            # plt.xlabel("number of naughty words")
            # plt.ylabel("frequency")
            # plt.title("distribution of naughty words over tweets/naughty texts")
            # plt.savefig("Screenshots/tweets8000_naughty word dist. 1-7.png")
            # # plt.show()
            #
            # plt.close()
            #
            # # ATTACK DICT PLOT
            # plt.plot(attack_dict.keys(), attack_dict.values())
            # plt.axis([0, 7, 0, 1200])
            # plt.xlabel("number of naughty words")
            # plt.ylabel("frequency")
            # plt.title("distribution of naughty words over the bullying texts")
            # plt.savefig("Screenshots/tweets8000_naughty word dist. 1-7 (bullying instances).png")
            # # plt.show()
            #
            # plt.close()
            #
            print(length_dict)
            length_keys = list(sorted(length_dict.keys()))
            length_values = []
            for i in length_keys:
                length_values.append(length_dict[i])
            print(length_keys)
            print(length_values)

            # plt.plot(length_keys, length_values)
            # plt.xlabel("length")
            # plt.ylabel("frequency")
            # plt.title("Number of tweets with certain lengths. N=16049")
            # plt.savefig("Screenshots/tweets_16K_length graph.png")

            # # PRINTING DICTS
            # for key in sorted(naughty_dict):
            #     print("There are", naughty_dict[key], "tweets messages with", key, "naughty words in.")
            #
            # print("---------------------")
            #
            # for key in sorted(attack_dict):
            #     print("There are", attack_dict[key], "bullying messages with", key, "naughty words in.")


def clean_tweets_16k():
    with open('processed_tweets_16K_3class.csv') as csv_file:
        csv_reader = csv.reader(csv_file, delimiter=',')
        line_count = 0

        # titles: ['rev_id', 'comment', 'year', 'logged_in', 'ns', 'sample', 'split', 'attack']
        with open('cleaned_tweets_16k_3class.csv', mode='w') as csv_write_file:
            csv_writer = csv.writer(csv_write_file, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)

            for row in csv_reader:

                if line_count == 0:
                    pass

                line_count += 1
                label_bullying = row[0]
                tweet = row[1]
                contains_url = row[2]
                naughty_count = row[3]
                norm = row[4]

                # remove RT tags
                tweet = re.sub('RT @[\w_]+: ', '', tweet)

                # remove mentions
                tweet = re.sub('@[\w_]+', '', tweet)

                # remove URLs
                tweet = re.sub(r'http\S+', '', tweet)

                # remove multiple spaces, replace with one space
                tweet = re.sub(' +', ' ', tweet)

                # remove punctuation
                tweet = re.sub('[\":=#&;\'?!@,./\\\\\n*]', '', tweet)

                # Write to clean file, removing the apostrophes
                # if not (is_blank) and not (is_NAME) and count > 0:
                csv_writer.writerow([label_bullying, tweet.lower(), contains_url, naughty_count, norm])

                # print current progress of lines processed
                if line_count % 1000 == 0:
                    print(line_count)


def count_labels():
    with open('cleaned_tweets_16K_3class.csv') as csv_file:
        csv_reader = csv.reader(csv_file, delimiter=',')
        label_count = defaultdict(int)

        for row in csv_reader:
            label_count[row[0]] += 1

    print(label_count)


# formspring()
# tweets_8000()
# tweets_1000()
# tweets_16k()
# clean_tweets_16k()
correlation()
# count_labels()
# dixon()