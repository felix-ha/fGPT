import dropbox
import contextlib
import datetime
import os
import time
import tarfile

from constants import *
from data_prep import read_from_json, get_token_int_dicts
from tokenizer import create_encoder, create_decoder
from model import simpleGPT, generate
import torch
from main import get_model
import streamlit as st


@contextlib.contextmanager
def stopwatch(message):
    """Context manager to print how long a block of code took."""
    t0 = time.time()
    try:
        yield
    finally:
        t1 = time.time()
        print('Total elapsed time for %s: %.3f' % (message, t1 - t0))


def upload(dbx, fullname, folder, subfolder, name, overwrite=False):
    """Upload a file.

    Return the request response, or None in case of error.
    """
    path = '/%s/%s/%s' % (folder, subfolder.replace(os.path.sep, '/'), name)
    while '//' in path:
        path = path.replace('//', '/')
    mode = (dropbox.files.WriteMode.overwrite
            if overwrite
            else dropbox.files.WriteMode.add)
    mtime = os.path.getmtime(fullname)
    with open(fullname, 'rb') as f:
        data = f.read()
    with stopwatch('upload %d bytes' % len(data)):
        try:
            res = dbx.files_upload(
                data, path, mode,
                client_modified=datetime.datetime(*time.gmtime(mtime)[:6]),
                mute=True)
        except dropbox.exceptions.ApiError as err:
            print('*** API error', err)
            return None
    print('uploaded as', res.name.encode('utf8'))
    return res


def download(dbx, folder, subfolder, name):
    """Download a file.

    Return the bytes of the file, or None if it doesn't exist.
    """
    path = '/%s/%s/%s' % (folder, subfolder.replace(os.path.sep, '/'), name)
    while '//' in path:
        path = path.replace('//', '/')
    with stopwatch('download'):
        try:
            md, res = dbx.files_download(path)
        except dropbox.exceptions.HttpError as err:
            print('*** HTTP error', err)
            return None
    data = res.content
    print(len(data), 'bytes; md:', md)
    return data


def list_folder(dbx, folder, subfolder):
    """List a folder.

    Return a dict mapping unicode filenames to
    FileMetadata|FolderMetadata entries.
    """
    path = '/%s/%s' % (folder, subfolder.replace(os.path.sep, '/'))
    while '//' in path:
        path = path.replace('//', '/')
    path = path.rstrip('/')
    try:
        with stopwatch('list_folder'):
            res = dbx.files_list_folder(path)
    except dropbox.exceptions.ApiError as err:
        print('Folder listing failed for', path, '-- assumed empty:', err)
        return {}
    else:
        rv = {}
        for entry in res.entries:
            rv[entry.name] = entry
        return rv


def extract_all_files(tar_file_path, extract_to):
    with tarfile.open(tar_file_path, 'r') as tar:
        tar.extractall(extract_to)


def download_and_extract(dbx, folder_dbx, file_name_src, folder_dst):

    res = download(dbx, folder_dbx, '', file_name_src)

    if res is not None:
        with open(file_name_dst, "wb") as f:
            f.write(res)

    extract_all_files(file_name_dst, folder_dst)


#https://www.dropbox.com/developers/apps/
token = os.getenv('DROPBOX_TOKEN')

dbx = dropbox.Dropbox(token)

folder_dbx = 'fGPT'
folder_downloads = 'downloads'
file_name_src_data =  'datapipeline.tar'
file_name_src_model =  'model.tar'


for file in [file_name_src_data, file_name_src_model]:
    file_name_src = file
    file_name_dst =  os.path.join(folder_downloads, file_name_src)
    if os.path.exists(file_name_dst):
        print("File {} already exists.".format(file_name_dst))
    else:
        download_and_extract(dbx, folder_dbx, file_name_src, folder_downloads)


#file_to_upload = '/workspaces/fGPT/runs/20231126_081051_GPT-2.tar'
#upload(dbx, file_to_upload, folder_dbx, '', 'model.tar', overwrite=True)


path_data = os.path.join(folder_downloads, 'datapipeline')
path_model = os.path.join(folder_downloads, '20231126_213755_fGPT')

token_to_int, int_to_token = get_token_int_dicts(path_data)
texts_ids_train = read_from_json(os.path.join(path_data, "texts_ids_train.json"))
texts_ids_validation = read_from_json(
    os.path.join(path_data, "texts_ids_validation.json")
)

dataset_info = read_from_json(os.path.join(path_data, "dataset_info.json"))
vocab_size = dataset_info["vocab_size"]
n_positions = dataset_info["n_positions"]

encoder = create_encoder(token_to_int, END_OF_TEXT, TOKEN_TO_REMOVE, UNK)
decoder = create_decoder(int_to_token)

stop_token_id = token_to_int[END_OF_TEXT]

model = get_model(vocab_size, n_positions, device='cpu')

training_result_dict = torch.load(os.path.join(path_model, "last", "model.pt"))
model_state_dict = training_result_dict["model_state_dict"]
model.load_state_dict(model_state_dict)


st.write(
    """
# fGPT 

A language model trained from scratch on [tiny stories](https://arxiv.org/abs/2305.07759)


"""
)

prompt = st.text_input('Enter the beginning of a story...')

if st.button('Generate'):
    output, _ = generate(
            model,
            prompt,
            encoder,
            decoder,
            stop_token_id=stop_token_id,
            max_n=5,
            choices_per_step=3,
        )
    st.text_area("continued story by model", output)
