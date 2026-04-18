# Trains a ConvNeXtV2 encoder on known SAS anomaly classes only, then
# clusters its embeddings with HDBSCAN to evaluate known-class separability.

from sas_open_world.experiments.convnextv2_known_hdbscan import main


if __name__ == "__main__":
    main()
