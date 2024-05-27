"""
I copied the following code from pylessons.com
"""

def fetch_data(url):
    """
    Fetches a dataset and saves it in Datasets/mnist
    :param url: the url of the dataset
    :return: data formatted in a numpy array
    """
    if os.path.exists(path) is False:
        os.makedirs(path)

    fp = os.path.join(path, hashlib.md5(url.encode('utf-8')).hexdigest())
    if os.path.isfile(fp):
        with open(fp, "rb") as f:
            data = f.read()
    else:
        with open(fp, "wb") as f:
            data = requests.get(url).content
            f.write(data)
    return np.frombuffer(gzip.decompress(data), dtype=np.uint8).copy()