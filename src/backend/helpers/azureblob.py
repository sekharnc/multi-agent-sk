from helpers.dutils import decorate_all_methods
from azure.identity import ClientSecretCredential
from azure.storage.blob import BlobServiceClient
from app_config import AppConfig

# from finrobot.utils import decorate_all_methods, get_next_weekday
from functools import wraps

def init_blob_api(func):
    @wraps(func)
    def wrapper(*args, **kwargs):
        global tenantId, clientId, clientSecret, blobAccountName, blobContainerName
        if AppConfig.AZURE_TENANT_ID is None:
            print("Please set the environment variable AZURE_TENANT_ID to use the Blob API.")
            return None
        else:
            tenantId = AppConfig.AZURE_TENANT_ID
            clientId = AppConfig.AZURE_CLIENT_ID
            clientSecret = AppConfig.AZURE_CLIENT_SECRET
            blobAccountName = AppConfig.AZURE_BLOB_STORAGE_NAME
            blobContainerName = AppConfig.AZURE_BLOB_CONTAINER_NAME
            print("Blob api key found successfully.")
            return func(*args, **kwargs)

    return wrapper


@decorate_all_methods(init_blob_api)
class azureBlobApi:

    def copyReport(downloadPath, blobName):
        try:
            with open(downloadPath, "rb") as file:
                readBytes = file.read()
            credentials = ClientSecretCredential(tenantId, clientId, clientSecret)
            blobService = BlobServiceClient(
                    "https://{}.blob.core.windows.net".format(blobAccountName), credential=credentials)
            blobClient = blobService.get_blob_client(container=blobContainerName, blob=blobName)
            blobClient.upload_blob(readBytes,overwrite=True)
            return blobClient.url
        except Exception as e:
            print("Error in copyReport: ", e)
            return None