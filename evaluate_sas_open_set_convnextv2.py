# Evaluates SAS anomaly-only open-set classification with a ConvNeXtV2 backbone
# and max-softmax rejection for unseen anomaly classes.

from sas_open_world.experiments.convnextv2_open_set import main


if __name__ == "__main__":
    main()
