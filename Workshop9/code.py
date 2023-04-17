# Melvin Moreno
# COMP6710
# Starter code provided by Dr. Akond Rahman
# 04/17/2023

from http import client
from itertools import count
from venv import create
import hvac 
import random 
import warnings

warnings.filterwarnings("ignore", category=DeprecationWarning)

def makeConn():
    hvc_client = client = hvac.Client(url='http://127.0.0.1:8200', token='hvs.TOKEN' ) 
    return hvc_client 

def storeSecret( client,  secr1 , cnt  ):
    secret_path     = 'SECRET_PATH_' + str( cnt  )
    create_response = client.secrets.kv.v2.create_or_update_secret(path=secret_path, secret=dict(password =  secr1 ) )
    print('Secret stored in Vault')

def retrieveSecret(client_, cnt_): 
    secret_path        = 'SECRET_PATH_' + str( cnt_  )
    read_response      = client_.secrets.kv.read_secret_version(path=secret_path) 
    secret_from_vault  = read_response['data']['data']['password']
    print('The secret we have obtained:')
    print(secret_from_vault)

if __name__ == '__main__': 
    clientObj    =  makeConn() 
    secret2store = [
        'root_user',
        'test_password',
        'ghp_ahAyHoRwoQ',
        'MTIzANO=',
        't5f28U'
        ]
    #store the secrets
    for idx, secret in enumerate(secret2store, start=1):
        print('Storing secret: ' + str(idx))
        print("="*50)
        storeSecret(clientObj, secret, idx)
        print('='*50)
    #retrieve the secrets
    for idx, secret in enumerate(secret2store, start=1):
        print('Retrieving secret: ' + str(idx))
        print("="*50)
        retrieveSecret(clientObj, idx)
        print('='*50)
        
        
