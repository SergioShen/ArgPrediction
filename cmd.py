# -*- coding: utf-8 -*-
"""
Author: Sijie Shen

Date: Aug 10, 2017
"""

import subprocess
import input_data
import numpy as np

DATA_LENGTH = input_data.VOCAB_SIZE


def get_prob(a, b, cvindex):
    """
    Get N-Gram prediction from Java program
    :param a: token1
    :param b: token2
    :param cvindex: fold index of cross validation
    :return: a softmax vector(float32 list)
    """
    result = subprocess.check_output("java -jar 3GramModel.jar -m {} -w {} {}".format(cvindex, a, b), shell=True)
    result = bytes.decode(result)
    prob_str = result.split("\n")

    prob = []
    for i in range(DATA_LENGTH):
        prob.append(float(prob_str[i]))

    return np.array(prob, dtype=np.float32)


if __name__ == "__main__":
    print(get_prob("ac", "unknown"))
