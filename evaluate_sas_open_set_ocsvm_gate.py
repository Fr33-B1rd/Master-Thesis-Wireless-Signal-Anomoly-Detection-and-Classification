# Evaluates SAS open-set recognition with PatchCore features, a known-class classifier,
# and per-class One-Class SVM gates for unknown rejection.

from sas_open_world.experiments.open_set_gate import main


if __name__ == "__main__":
    main()
