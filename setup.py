import os

if __name__ == "__main__":

    if not os.path.exists('data/'):
        os.mkdir('data/')

    if not os.path.exists('models/'):
        os.mkdir('models/')