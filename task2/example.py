import numpy as np

def transform(X):
    # This transforms [x1, .., xn]  ->  [x1, .., xn, 1]
    # np.insert needs the position of where the 1 should be inserted (which is 400)
    # and the axis along which the datapoints are stacked
    a = len(X.shape) - 1  # If X is 1-dim. we want to add at axis 0. If 2-dim, we want 1
    d = X.shape[-1]       # If X is R^(m*n), this gives n. If X is R^(n) this gives also n
    return np.insert(X, d, 1, axis = a)


def mapper(key, value):

    # Constant values
    learning_rate = 0.15
    first_decay_rate = 0.9
    second_decay_rate = 0.99
    epsilon = 10e-8

    # Variables
    weights = np.zeros(401)
    first_moment = np.zeros(401)
    second_moment = np.zeros(401)

    # Data
    classes, features = parse_value(value)
    features = transform(features)
    N = classes.shape[0]

    # Go once through all data points
    for n in range(N):
        # Learn
        y = classes[n]
        x = features[n, :].transpose() # squeeze makes the (1, 401)-slice into a (401)-vector
        gradient = learn(y, x, weights)
        # Update
        n_iter = n+1 # Adam needs the iteration number
        first_moment, second_moment, weights = ADAM(n_iter, gradient, first_moment, second_moment,\
                weights, learning_rate, first_decay_rate, second_decay_rate, epsilon)
        # Yielding weights at every learning step reduces variance
        # But this doesn't have to happen at *every* step
        yield 1, weights
    #


def reducer(key, values):
    
    yield np.asarray(values).mean(axis=0)



#
#  CUSTOM METHODS
#


def parse_value(value):
    
    """
    I have no idea why Alex's idea doesn't work. Numpy has trouble reshaping for some reason
    If anyone gets this to work, we all would be happy

    matrix = np.matrix([v.split(' ') for v in value]).astype(float)
    classes = matrix[:, 0]
    features = matrix[:, 1:]

    """
    
    Num = len(value)
    Dim = 400

    classes  = np.empty([Num], dtype = int)
    features = np.empty([Num, Dim])

    for n in range(Num):
        pic_as_strings = value[n].split()

        # The first value is the class
        classes[n] = float(pic_as_strings[0])
        # (Directly calling int() on strings doesn't work, float() is more powerful)

        # The rest are features
        for d in range(0, Dim):
            features[n, d] = float(pic_as_strings[d+1])
    
    
    return classes, features


def learn(y, x, w):
    loss = y * np.dot(w, x)
    gradient = - y * x
    # Hinge loss
    if loss >= 1:
        loss = 0
        gradient = 0
    return gradient

# https://arxiv.org/pdf/1412.6980.pdf
# I used the small optimization where m_hat and v_hat are replaced with a new alpha
def ADAM(t, grad, m, v, w, alpha, beta1, beta2, eps):
    new_m = beta1 * m + (1 - beta1) * grad
    new_v = beta2 * v + (1 - beta2) * (grad**2)
    new_a = alpha * np.sqrt(1 - beta2**t) / (1 - beta1**t)
    new_w = w - new_a * new_m / (np.sqrt(new_v + eps))
    return new_m, new_v, new_w