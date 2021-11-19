from azureml.core.authentication import InteractiveLoginAuthentication
from azureml.core import Workspace
interactive_auth = InteractiveLoginAuthentication(tenant_id="99e1e721-7184-498e-8aff-b2ad4e53c1c2")
ws = Workspace.get(name='mlw-esp-udea',
            subscription_id='f93ed890-92cb-49de-a5a7-70fa3c14a4a2',
            resource_group='rg-ml-udea',
            location='eastus',
            auth=interactive_auth
            )

ws.write_config(path='.azureml')