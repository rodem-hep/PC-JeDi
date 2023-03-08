# PC-JeDi

This is the publically available code repository for Particle-Cloud-Jet-Diffusion (PC-JeDi) {ArXiv link}

To this code you will need to
* Install the libraries listed in the requirements.txt using python > 3.9
* Download the JetNet dataset {ArXiv link}
* Make a free WandB account
* Define following entries in the yaml configs
    * configs/paths/default.yaml
        * data_dir: Path to the downloaded jetnet dataset
        * output_dir: Path save the trained model and associated plots
    * configs/logger/default.yaml
        * wandb/entity: Your username on WandB
    * configs/train.yaml
        * project_name: The desired name of the project, will be used to save the model
        * network_name: The desired name of the run, will be used to save the model

Once the configuration is set, you can run python scripts/train.py
