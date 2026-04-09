"""
The overarching orchestration loop governing the Communication Rounds.
"""
from .client_node import HospitalClientNode
from .server_aggregator import FederatedServer
from phase_2_local_models.model_factory import create_local_model
from .weight_utils import set_weights
from sklearn.linear_model import SGDClassifier

class FederatedPipelineRunner:
    """
    Executes the standard Federated Averaging algorithmic loop.
    """
    def __init__(self, clients: list[HospitalClientNode], server: FederatedServer):
        self.clients = clients
        self.server = server
        
    def run_training_loop(self, rounds: int = 5, local_epochs: int = 3) -> SGDClassifier:
        """
        Executes N Communication Rounds.
        During each round, all clients train in parallel (simulated here serially),
        then all weights are extracted and aggregated by the server.
        """
        for r in range(rounds):
            print(f"   [SYNC] --- Starting Federated Round {r+1}/{rounds} ---")
            
            client_parameters = []
            
            # Step 1: Client side processing
            for client in self.clients:
                # Client downloads current global weights, trains, and returns new weights
                new_weights = client.train_on_global_weights(
                    global_coef=self.server.global_coef, 
                    global_intercept=self.server.global_intercept, 
                    epochs=local_epochs
                )
                client_parameters.append(new_weights)
                
            # Step 2: Server side processing
            # Server averages the vectors together securely
            self.server.aggregate_and_update(client_parameters)
            
        print("   [SYNC] --- Federated Training Complete ---")
        
        # Finally, we instantiate a usable version of the Global Model so we can test it
        final_global_model = create_local_model()
        # Initialize memory with baby partial fit
        final_global_model.partial_fit([[0]*6], [0], classes=[0,1])
        return set_weights(final_global_model, self.server.global_coef, self.server.global_intercept)
