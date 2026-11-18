import math

TARANTULA = "Tarantula"
OCHIAI = "Ochiai"
OP2 = "Op2"
BARINEL = "Barinel"
DSTAR = "Dstar"
RUSSELL_RAO = "Russell_rao"
SIMPLE_MATCHING = "Simple_matching"
ROGERS_TANIMOTO = "Rogers_tanimoto"
AMPLE = "Ample2"
JACCARD = "Jaccard"
COHEN = "Cohen"
SCOTT = "Scott"
ROGOT1 = "Rogot1"
GEOMETRIC_MEAN = "Geometric_mean"
M2 = "M2"
WONG1 = "Wong1"
SOKAL = "Sokal"
SORENSEN_DICE = "Sorensen_dice"
DICE = "Dice"
HUMANN = "Humman"
M1 = "M1"
WONG2 = "Wong2"
WONG3 = "Wong3"
ZOLTAR = "Zoltar"
OVERLAP = "Overlap"
EUCLID = "Euclid"
ROGOT2 = "Rogot2"
HAMMING = "Hamming"
FLEISS = "Fleiss"
ANDERBERG = "Anderberg"
GOODMAN = "Goodman"
HARMONIC_MEAN = "Harmonic_mean"
KULCZYNSKI1 = "Kulczynski1"
KULCZYNSKI2 = "Kulczynski2"

TARANTULA_SCORE = "Tarantula_score"
TARANTULA_AVERAGE = "Tarantula_average"
OCHIAI_SCORE = "Ochiai_score"
OCHIAI_AVERAGE = "Ochiai_average"
OP2_SCORE = "Op2_score"
OP2_AVERAGE = "Op2_average"
DSTAR_SCORE = "Dstar_score"
DSTAR_AVERAGE = "Dstar_average"
BARINEL_SCORE = "Barinel_score"
BARINEL_AVERAGE = "Barinel_average"
RUSSELL_RAO_SCORE = "Russell_rao_score"
RUSSELL_RAO_AVERAGE = "Russell_rao_average"
SIMPLE_MATCHING_SCORE = "Simple_matching_score"
SIMPLE_MATCHING_AVERAGE = "Simple_matching_average"
ROGERS_TANIMOTO_SCORE = "Rogers_tanimoto_score"
ROGERS_TANIMOTO_AVERAGE = "Rogers_tanimoto_average"
AMPLE_SCORE = "Ample2_score"
AMPLE_AVERAGE = "Ample2_average"
JACCARD_SCORE = "Jaccard_score"
JACCARD_AVERAGE = "Jaccard_average"
COHEN_SCORE = "Cohen_score"
COHEN_AVERAGE = "Cohen_average"
SCOTT_SCORE = "Scott_score"
SCOTT_AVERAGE = "Scott_average"
ROGOT1_SCORE = "Rogot1_score"
ROGOT1_AVERAGE = "Rogot1_average"
GEOMETRIC_MEAN_SCORE = "Geometric_mean_score"
GEOMETRIC_MEAN_AVERAGE = "Geometric_mean_average"
M2_SCORE = "M2_score"
M2_AVERAGE = "M2_average"
WONG1_SCORE = "Wong1_score"
WONG1_AVERAGE = "Wong1_average"
SOKAL_SCORE = "Sokal_score"
SOKAL_AVERAGE = "Sokal_average"

SORENSEN_DICE_SCORE = "Sorensen_dice_score"
DICE_SCORE = "Dice_score"
HUMANN_SCORE = "Humman_score"
M1_SCORE = "M1_score"
WONG2_SCORE = "Wong2_score"
WONG3_SCORE = "Wong3_score"
ZOLTAR_SCORE = "Zoltar_score"
OVERLAP_SCORE = "Overlap_score"
EUCLID_SCORE = "Euclid_score"
ROGOT2_SCORE = "Rogot2_score"
HAMMING_SCORE = "Hamming_score"
FLEISS_SCORE = "Fleiss_score"
ANDERBERG_SCORE = "Anderberg_score"
GOODMAN_SCORE = "Goodman_score"
HARMONIC_MEAN_SCORE = "Harmonic_mean_score"
KULCZYNSKI1_SCORE = "Kulczynski1_score"
KULCZYNSKI2_SCORE = "Kulczynski2_score"
SORENSEN_DICE_AVERAGE = "Sorensen_dice_average"
DICE_AVERAGE = "Dice_average"
HUMANN_AVERAGE = "Humman_average"
M1_AVERAGE = "M1_average"
WONG2_AVERAGE = "Wong2_average"
WONG3_AVERAGE = "Wong3_average"
ZOLTAR_AVERAGE = "Zoltar_average"
OVERLAP_AVERAGE = "Overlap_average"
EUCLID_AVERAGE = "Euclid_average"
ROGOT2_AVERAGE = "Rogot2_average"
HAMMING_AVERAGE = "Hamming_average"
FLEISS_AVERAGE = "Fleiss_average"
ANDERBERG_AVERAGE = "Anderberg_average"
GOODMAN_AVERAGE = "Goodman_average"
HARMONIC_MEAN_AVERAGE = "Harmonic_mean_average"
KULCZYNSKI1_AVERAGE = "Kulczynski1_average"
KULCZYNSKI2_AVERAGE = "Kulczynski2_average"


def tarantula_calculation(fails, passes, total_failed_tests, total_passed_tests):
    if total_failed_tests == 0 or total_passed_tests == 0:
        return 0
    if fails == 0:
        return 0

    return (fails / total_failed_tests) / \
           ((fails / total_failed_tests) +
            (passes / total_passed_tests))


def ochiai_calculation(fails, passes, total_failed_tests, total_passed_tests):
    if total_failed_tests == 0:
        return 0
    if fails == 0:
        return 0
    return fails / math.sqrt(total_failed_tests * (
            fails + passes))


def op2_calculation(fails, passes, total_failed_tests, total_passed_tests):
    return fails - passes / (total_passed_tests + 1)


def barinel_calculation(fails, passes, total_failed_tests, total_passed_tests):
    if fails == 0 and passes == 0:
        return 0
    return 1 - passes / (passes + fails)


def dstar_calculation(fails, passes, total_failed_tests, total_passed_tests):
    temp = passes + (total_failed_tests - fails)
    if fails == 0:
        return 0
    elif temp == 0:
        return 1000
    return (fails * fails) / temp


def russell_rao_calculation(fails, passes, total_failed_tests, total_passes_tests):
    if total_failed_tests + total_passes_tests == 0:
        return 0
    else:
        return fails / (total_failed_tests + total_passes_tests)


def simple_matching_calculation(fails, passes, total_failed_tests, total_passes_tests):
    if total_passes_tests + total_failed_tests == 0:
        return 0
    else:
        return (fails + total_passes_tests - passes) / (total_failed_tests + total_passes_tests)


def rogers_tanimoto_calculation(fails, passes, total_failed_tests, total_passes_tests):
    temp = fails + (total_passes_tests - passes) + 2 * (total_failed_tests - fails + passes)
    if temp == 0:
        return 0
    else:
        return (fails + (total_passes_tests - passes)) / temp


def ample2_calculation(fails, passes, total_failed_tests, total_passes_tests):
    if total_failed_tests == 0:
        return 0
    elif total_passes_tests == 0:
        return 1
    else:
        return fails / total_failed_tests - passes / total_passes_tests


def jaccard_calculation(fails, passes, total_failed_tests, total_passes_tests):
    if (total_failed_tests + passes) == 0:
        return 0
    else:
        return fails / (total_failed_tests + passes)


def cohen_calculation(fails, passes, total_failed_tests, total_passed_tests):
    temp = (fails + passes) * (total_passed_tests) + total_failed_tests * (
                total_failed_tests - fails + total_passed_tests - passes)
    if temp == 0:
        return 0
    else:
        return (2 * fails * (total_passed_tests - passes) - 2 * (total_failed_tests - fails) * passes) / temp


def scott_calculation(fails, passes, total_failed_tests, total_passes_tests):
    temp = (2 * fails + total_failed_tests - fails + passes) * (
                2 * (total_passes_tests - passes) + total_failed_tests - fails + passes)
    if temp == 0:
        return 0
    else:
        return (4 * fails * (total_passes_tests - passes) - 4 * (total_failed_tests - fails) * passes - (
                    total_failed_tests - fails - passes) * (total_failed_tests - fails - passes)) / temp


def rogot1_calculation(fails, passes, total_failed_tests, total_passes_tests):
    temp1 = 2 * fails + total_failed_tests - fails + passes
    temp2 = 2 * (total_passes_tests - passes) + total_failed_tests - fails + passes

    if temp1 == 0 or temp2 == 0:
        return 0
    else:
        return (fails / temp1 + (total_passes_tests - passes) / temp2) / 2


def geometric_mean_calculation(fails, passes, total_failed_tests, total_passes_tests):
    temp = (fails + passes) * (
                total_failed_tests - fails + total_passes_tests - passes) * total_failed_tests * total_passes_tests
    if temp <= 0:
        return 0
    else:
        return (fails * (total_passes_tests - passes) - (total_failed_tests - fails) * passes) / math.sqrt(temp)


def m2_calculation(fails, passes, total_failed_tests, total_passes_tests):
    temp = fails + (total_passes_tests - passes) + 2 * (total_failed_tests - fails + passes)
    if temp == 0:
        return 0
    else:
        return fails / temp


def wong1_calculation(fails, passes, total_failed_tests, total_passes_tests):
    return fails


def sokal_calculation(fails, passes, total_failed_tests, total_passes_tests):
    temp = 2 * (fails + total_passes_tests - passes) + (total_failed_tests - fails) + passes
    if temp == 0:
        return 0
    else:
        return 2 * (fails + total_passes_tests - passes) / temp


def sorensen_dice_calculation(fails, passes, total_failed_tests, total_passes_tests):
    temp = 2 * fails + (total_failed_tests - fails) + passes
    if temp == 0:
        return 0
    else:
        return (2 * fails) / temp


def dice_calculation(fails, passes, total_failed_tests, total_passes_tests):
    temp = fails + (total_failed_tests - fails) + passes
    if temp == 0:
        return 0
    else:
        return (2 * fails) / temp


def humman_calculation(fails, passes, total_failed_tests, total_passes_tests):
    temp = total_failed_tests + total_passes_tests
    if temp == 0:
        return 0
    else:
        return (fails + (total_passes_tests - passes) - (total_failed_tests - fails) - passes) / temp


def m1_calculation(fails, passes, total_failed_tests, total_passes_tests):
    temp = (total_failed_tests - fails) + passes
    if temp == 0:
        return 0
    else:
        return (fails + (total_passes_tests - passes)) / temp


def wong2_calculation(fails, passes, total_failed_tests, total_passes_tests):
    return fails - passes


def wong3_calculation(fails, passes, total_failed_tests, total_passes_tests):
    if total_failed_tests + total_failed_tests == 0:
        return -1000
    if passes <= 2:
        return fails - passes
    elif 2 < passes and passes <= 10:
        return fails - (2 + 0.1 * (passes - 2))
    else:
        return fails - (2.8 + 0.001 * (passes - 10))


def zoltar_calculation(fails, passes, total_failed_tests, total_passes_tests):
    if fails == 0:
        return 0
    temp = total_failed_tests + passes + (10000 * passes * (total_failed_tests - fails)) / fails
    if temp == 0:
        return 0
    else:
        return fails / temp


def overlap_calculation(fails, passes, total_failed_tests, total_passes_tests):
    temp = min(fails, total_failed_tests - fails, passes)
    if temp == 0:
        return 0
    else:
        return fails / temp


def euclid_calculation(fails, passes, total_failed_tests, total_passes_tests):
    temp = fails + (total_passes_tests - passes)
    if temp <= 0:
        return 0
    else:
        return math.sqrt(temp)


def rogot2_calculation(fails, passes, total_failed_tests, total_passes_tests):
    temp1 = fails + passes
    temp2 = total_failed_tests
    temp3 = total_passes_tests
    temp4 = (total_passes_tests - passes) + (total_failed_tests - fails)

    if temp1 == 0 or temp2 == 0:
        return 0
    if temp3 == 0 or temp4 == 0:
        return (1 / 4) * ((fails / temp1) + (fails / temp2))
    else:
        return (1 / 4) * ((fails / temp1) + (fails / temp2) + ((total_passes_tests - passes) / temp3) + (
                    (total_passes_tests - passes) / temp4))


def hamming_calculation(fails, passes, total_failed_tests, total_passes_tests):
    return fails + (total_passes_tests - passes)


def fleiss_calculation(fails, passes, total_failed_tests, total_passes_tests):
    temp = (2 * fails + (total_failed_tests - fails) + passes) + (
                2 * (total_passes_tests - passes) + (total_failed_tests - fails) + passes)
    if temp == 0:
        return 0
    else:
        temp2 = 4 * fails * (total_passes_tests - passes) - 4 * (total_failed_tests - fails) * passes - (
                    total_failed_tests - fails - passes) * (total_failed_tests - fails - passes)
        return temp2 / temp


def anderberg_calculation(fails, passes, total_failed_tests, total_passes_tests):
    temp = fails + 2 * (total_failed_tests - fails + passes)
    if temp == 0:
        return 0
    else:
        return fails / temp


def goodman_calculation(fails, passes, total_failed_tests, total_passes_tests):
    temp = 2 * fails + (total_failed_tests - fails) + passes
    if temp == 0:
        return 0
    else:
        return (2 * fails - (total_failed_tests - fails) - passes) / temp


def harmonic_mean_calculation(fails, passes, total_failed_tests, total_passes_tests):
    temp = (fails + passes) * (
                total_passes_tests - passes + total_failed_tests - fails) * total_failed_tests * total_passes_tests
    if temp == 0:
        return 0
    else:
        temp2 = fails * (total_passes_tests - passes) - (total_failed_tests - fails) * passes
        temp3 = (fails + passes) * (
                    total_passes_tests - passes + total_failed_tests - fails) + total_failed_tests * total_passes_tests
        return (temp2 * temp3) / temp


def kulczynski1_calculation(fails, passes, total_failed_tests, total_passes_tests):
    temp = (total_failed_tests - fails) + passes
    if temp == 0:
        return 0
    else:
        return fails / temp


def kulczynski2_calculation(fails, passes, total_failed_tests, total_passes_tests):
    temp1 = total_failed_tests
    temp2 = fails + passes
    if temp1 == 0 or temp2 == 0:
        return 0
    else:
        return (1 / 2) * (fails / temp1 + fails / temp2)
