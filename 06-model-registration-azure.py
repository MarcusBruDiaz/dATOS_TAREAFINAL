 # 06-model-registration-azure.py
from azureml.core import Workspace
from azureml.core import Model
if __name__ == "__main__":
    ws = Workspace.from_config(path='./.azureml',_file_name='config.json')
    model = Model.register(model_name='proyecc_model',tags={'area': 'udea_training','scoring' : 0.97, 'data_set_size': 100000},model_path='outputs/proyecc_model.pkl',workspace = ws)
    print(model.name, model.id, model.version, sep='\t')