from pyspark import SparkContext


def print_hi(name):
    print(f'Hi, {name}')
    sc = SparkContext(appName="getEvenNums")
    x = sc.parallelize([1, 2, 3, 4])
    y = x.filter(lambda x: (x % 2 == 0))
    print(y.collect())
    sc.stop()


if __name__ == '__main__':
    print_hi('PyCharm')
