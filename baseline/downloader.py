import os


def download(path):
    fname = os.path.basename(path)
    if os.path.exists(fname):
        print('File already exists, not downloading.')
        return fname, True

    print('Downloading ' + path)

    def progress(count, block_size, total_size):
        if count % 20 == 0:
            print('Downloaded %02.02f/%02.02f MB\n' % (
                count * block_size / 1024.0 / 1024.0,
                total_size / 1024.0 / 1024.0))

    from six.moves import urllib
    filepath, _ = urllib.request.urlretrieve(
        path, filename=fname, reporthook=progress)
    return filepath, False


def download_and_extract(path, dst):
    import zipfile
    filepath, is_exist = download(path)
    if not os.path.exists(dst):
        os.makedirs(dst)
    if not is_exist:
        with zipfile.ZipFile(filepath, 'r') as zipfile:
            zipfile.extractall(dst)
