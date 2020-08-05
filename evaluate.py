import argparse
import os
import requests
from sklearn.metrics import classification_report

rs = requests.session()


# returns lists of filenames and abs filepaths of a directory
def get_filenames_and_full_paths_for_images(base_dir):
    path_list = []
    image_names = []
    for path, subdirs, files in os.walk(base_dir):
        for name in files:
            if name.endswith(".jpg") or name.endswith(".jpeg") or name.endswith(".png") or name.endswith(".txt"):
                # if name.endswith(".jpeg"):
                get_path = os.path.join(path, name)
                path_list.append(get_path)
                image_names.append(name)
    return image_names, path_list


# returns response of detect_mask api
def call_face_mask_detection_api(filepath):
    # data = cv2.imread(filepath)
    url = 'http://{}:{}/detect_mask'.format(args.ip, args.port)
    with open(filepath, "rb") as f:
        data = f.read()
    headers = {
        'Content-Type': 'image/jpeg'
    }

    resp = rs.post(url=url, headers=headers,
                   data=data)

    resp = resp.json()
    print(resp)

    return resp


# prepare predictions based on detect_mask responses; if masked then 1 otherwise 0 is appended in the list of predictions
def get_gts_preds(fnames, filepaths):
    preds = []
    gts = []
    for i in range(0, len(filepaths)):
        fp = filepaths[i]
        fn = fnames[i]
        resp = call_face_mask_detection_api(fp)
        if fn.rsplit('_', 1)[0] == 'mask':
            gts.append(1)
        elif fn.rsplit('_', 1)[0] == 'no-mask':
            gts.append(0)

        if resp['mask'] > resp['no-mask']:
            preds.append(1)
        else:
            preds.append(0)
    return gts, preds


# check if the path provided as argument is a directory
def dir_path(string):
    if os.path.isdir(string):
        return string
    else:
        raise NotADirectoryError(string)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='classification report for a directory containing all masked images')
    parser.add_argument('--dirpath', default='/home/tigerit/PycharmProjects/mask_detection/dataset',
                        type=dir_path,
                        help='directory containing all masked images')
    parser.add_argument('--ip', default='localhost', type=str, help='mask detection ip')
    parser.add_argument('--port', default='5000', type=str, help='mask detection port')
    args = parser.parse_args()

    fnames, fpaths = get_filenames_and_full_paths_for_images(args.dirpath)

    y_trues, y_preds = get_gts_preds(fnames, fpaths)

    # classification_report(y_trues, y_preds)
    target_names = ['no-mask', 'mask']

    result = classification_report(y_trues, y_preds, target_names=target_names)

    print(result)
