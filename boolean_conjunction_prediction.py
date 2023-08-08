import random

from collections import namedtuple

sample_unit = namedtuple('sample_unit', ('x', 'y'))

def evaluate(h, x):
    """
    Evaluate function h on sample unit x i.e.
    h(x) = y

    parameters
    ----------
    h: iterable
       (h_1, ..., h_n) represents a conjunciton
       of at most n boolean literals.
       h[i] = -1 => h_i absent, 0 => h_i', 1 => h_i
              None => h not defined
                   => y not defined (return None)
    x: iterable
       (x_1, ..., x_n) represents the sample unit
       for which is input to the hypothesis.
       x[i] \in {1, 0}

    return
    ------
    y: int
       h(x) = h_1(x_1) ... and ... h_n(x_n) = y
       y \in {1, 0} or None if h not defined well
    """

    assert len(h) == len(x)
    y = True
    for i in range(len(x)):
        if h[i] == -1:
            continue
        elif h[i] == 0:
            y = y and not x[i]
        elif h[i] == 1:
            y = y and x[i]
        else:
            return None

    return int(y)

def generate_sample(m, target_concept):
    """
    Generate i.i.d. sample of size m

    parameters
    ----------
    m: int
       sample size
    target_concept: iterable
                    the function to be prediceted
    return
    ------
    sample: list of sample_units
            sample of size m
    """

    n = len(target_concept)
    sample = [
     sample_unit(
      x := tuple(random.choice((1, 0)) for __ in range(n)),
      evaluate(target_concept, x)
     )
     for _ in range(m)]

    return sample

def ERM_PAC_algorithm(train_data):
    """
    Return a hypothesis consistent with train data

    parameters
    ----------
    train_data: list of sample_units
                sample (training data)

    return
    ------
    ch: iterable
        (h_1, ..., h_n) represents a conjunciton
        of at most n boolean literals.
        ch[i] = -1->ch_i absent, 0->ch_i', 1->ch_i
    """

    assert len(train_data) > 0
    n = len(train_data[0].x)
    ch = [None for _ in range(n)]
    for sample_unit in train_data:
        if sample_unit.y == 0:
            continue
        elif sample_unit.y == 1:
            for i in range(len(sample_unit.x)):
                if ch[i] is None:
                    ch[i] = sample_unit.x[i]
                else:
                    if ch[i] != sample_unit.x[i]:
                        ch[i] = -1

    return ch

def emperical_error(sample, hypothesis):
    """
    Average error of prediction using hypothesis on the given sample

    parameters
    ----------
    sample: list of sample_units
    hypothesis: iterable

    returns
    -------
    average_error: float
    """

    indicator_sum = sum(
        evaluate(hypothesis, sample_unit.x) != sample_unit.y
        for sample_unit in sample)
    average_error = indicator_sum / len(sample)

    return average_error

def generalization_error_estimate(target_concept, hypothesis, max_iter=10000):
    """
    An estimate of the true error using a hypothesis to achieve the target
    concept

    """

    iterations = 0
    indicator_sum = 0
    while iterations < max_iter:
        x = tuple(random.choice((1, 0)) for __ in range(n))
        t_x = evaluate(target_concept, x)
        h_x = evaluate(hypothesis, x)
        if t_x != h_x:
            indicator_sum += 1
        iterations += 1
    estimated_error = indicator_sum / max_iter
    return estimated_error


if __name__ == "__main__":
    # max number of literals in boolean conjunction
    n = 10
    target_concept = tuple(random.choice((-1, 0, 1))
                       for _ in range(n))
    # size of training sample
    m = 200
    train_data = generate_sample(m, target_concept)
    hypothesis = ERM_PAC_algorithm(train_data)
    error = emperical_error(train_data, hypothesis)

    print('Target Concept:', *target_concept)
    print('Hypothesis:    ', *hypothesis)
    print('Check for consistency, error on train data:', error)

    print('Estimate of generalization error:',
           generalization_error_estimate(target_concept, hypothesis))




