from matplotlib import pyplot as plt

if __name__ == '__main__':
    x = [2000, 4000, 6000, 8000, 10000, 12000, 14000, 16000, 18000, 20000]
    y1 = [6, 7, 7, 7, 7, 7, 8, 8, 8, 8]
    y2 = [4, 5, 5, 5, 5, 5, 5, 5, 6, 6]
    y3 = [3, 4, 4, 4, 4, 4, 4, 4, 4, 4]
    y4 = [3, 3, 3, 3, 3, 3, 3, 3, 3, 4]

    fig, ax = plt.subplots()

    ax.plot(x, y1, label="ORDER=2")
    ax.plot(x, y2, label="ORDER=4")
    ax.plot(x, y3, label="ORDER=8")
    ax.plot(x, y4, label="ORDER=16")
    plt.legend()

    plt.ylim(0, 10)

    plt.show()
