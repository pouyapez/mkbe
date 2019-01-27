import numpy as np


def Mrr(Score_N, Score):
    """ calculate MRR for each sample in test dataset """
    S_n = Score_N.tolist()
    S = Score_N
    for i in Score_N:
        S_n = Score_N.tolist()
        if np.absolute(i - Score) < 0.0001:
            Score_N = np.delete(Score_N, S_n.index(i))
    MR = np.append(Score_N, Score)
    MR = MR.tolist()
    MR.sort(reverse=True)
    MR = MR.index(Score)
    return 1. / (MR + 1), MR + 1


def hits(Score_N, Score, is_print=True):
    MRR = 0
    hit1 = 0
    hit2 = 0
    hit3 = 0
    for i in range(len(Score)):
        mrr, hit = Mrr(Score_N[i], Score[i])
        MRR = MRR + mrr
        if hit == 1:
            hit1 += 1
        if hit < 3:
            hit2 += 1
        if hit < 4:
            hit3 += 1

    MRR = MRR * (1. / len(Score))

    if is_print:
        print('MRR:', MRR)
        print('hit1:', hit1 * (1. / len(Score)))
        print('hit2:', hit2 * (1. / len(Score)))
        print('hit3:', hit3 * (1. / len(Score)))
        return hit1 * (1. / len(Score))
    return hit1 * (1. / len(Score)), [MRR, hit1 * (1. / len(Score)), hit2 * (1. / len(Score)), hit3 * (1. / len(Score))]


if __name__ == "__main__":
    Score = np.load('scores/scalar_embedding_positive.npy')
    Score_N = np.load('scores/scalar_embedding_negative.npy')

    print(len(Score), len(Score_N))
    print(Score[0:5], Score_N[0:5])

    MRR = 0
    hit1 = 0
    hit2 = 0
    hit3 = 0
    for i in range(len(Score)):
        mrr, hit = Mrr(Score_N[i], Score[i])
        MRR = MRR + mrr
        if hit == 1:
            hit1 += 1
        if hit < 3:
            hit2 += 1
        if hit < 4:
            hit3 += 1

    MRR = MRR * (1. / len(Score))
    print('MRR:', MRR)
    print('hit1:', hit1 * (1. / len(Score)))
    print('hit2:', hit2 * (1. / len(Score)))
    print('hit3:', hit3 * (1. / len(Score)))
