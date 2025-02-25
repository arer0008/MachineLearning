# Note: you're not allowed to use any other library in this file.

def get_ngrams_wid(arr_sentences, window_size=1):
    """
    Goals: get frequency of any two words within window_size.
    # Constrains:
    - Window size changes upon request.
    # Solutions:
        <It is recommended to write your approach before implementation>
    ## BigO:
    ## Memory:
    # Test:
      String: I like deep learning
      Window_size = 2
      Output:
          I -> (+)like deep;
          like -> (-) I; (+) deep learning
          deep -> (-) I like; (+) learning
          learning -> (-) like deep
    :param arr_sentences: list of sentences
    :param window_size: size of the look-up window
    :return: freq_dict - frequency dictionary
    """
    freq_dict = {}
    # TODO: Your implementation here
    i = 0
    for sentence in arr_sentences:
        words = sentence.upper().split()
        for index, word in enumerate(words):
            for x in range(1, window_size+1):
                if index+x < len(words):
                    freq_dict.setdefault((word, words[index+x]), 0)
                    freq_dict[(word, words[index+x])] += 1
                if index-x > -1:
                    freq_dict.setdefault((word, words[index-x]), 0)
                    freq_dict[(word, words[index-x])] += 1

    # End of implementation
    return freq_dict


def find_cooccurrencies(arr_sentences, window_size=1):
    """
    Goals: scan all sentences, winthin a pre-defined window size to return co-occurrency matrix X
    # Solutions:
        <It is recommended to write your approach before implementation>
    ## BigO:
    ## Memory:
    # Complexity:
    # Test case:

    Input: inputs = ["I like deep learning .", "I like nlp .", "I enjoy flying ."]
    Output:
        vocab_dict.keys = ['and', 'nlp', 'like', 'i', 'enjoy', 'flying', 'deep', '.', 'learning', 'with']
        X = [[0, 1, 0, 0, 0, 0, 1, 1, 1, 0],
            [1, 0, 1, 0, 0, 0, 0, 2, 1, 0],
            [0, 1, 0, 0, 0, 0, 1, 1, 1, 0],
            [0, 0, 0, 0, 1, 1, 1, 0, 0, 0],
            [0, 0, 0, 1, 0, 1, 0, 0, 0, 1],
            [0, 0, 0, 1, 1, 0, 1, 0, 0, 1],
            [1, 0, 1, 1, 0, 1, 0, 1, 2, 1],
            [1, 2, 1, 0, 0, 0, 1, 0, 1, 0],
            [1, 1, 1, 0, 0, 0, 2, 1, 0, 1],
            [0, 0, 0, 0, 1, 1, 1, 0, 1, 0]]

    :param arr_sentences: list of sentences
    :window_size: a predifined 'radius'
    :return:
        X: a co-occurrency matrix X
        vocab_dict.keys(): vocabulary list
    """
    vocab_dict = {}
    X = []
    #arr_sentences = ["I like deep learning .", "I like nlp .", "I enjoy flying .", "I enjoy deep learning ."]

    # TODO: Your implementation here
    i = 0
    for sentence in arr_sentences:
        for word in sentence.upper().split():
            if word not in vocab_dict.values():
                vocab_dict.update({i:word})
                i+=1

    freq_dict = get_ngrams_wid(arr_sentences, window_size)
    for x in vocab_dict.values():
        list = []
        for y in vocab_dict.values():
            if (x,y) in freq_dict.keys():
                list.append(freq_dict.get((x,y)))
            else:
                list.append(0)
        X.append(list)


    #print(X)
    # End of implementation
    return X, vocab_dict.values()
