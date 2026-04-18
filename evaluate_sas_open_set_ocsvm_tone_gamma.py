# Sweeps the tone-class OCSVM gamma inside the SAS open-set pipeline to test
# whether a class-specific boundary change improves unknown rejection.

from sas_open_world.experiments.tone_gamma import main


if __name__ == "__main__":
    main()
