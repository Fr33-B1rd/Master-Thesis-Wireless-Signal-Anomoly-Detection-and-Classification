# Top-level launcher for the few-shot abnormal-subtype classifier on top of
# PatchCore patch features. Mirrors simulate_sas_convnextv2_open_world.py and
# evaluate_sas_open_set_convnextv2.py — pure import + main() shim.

from sas_open_world.experiments.fewshot_patchcore_classifier import main


if __name__ == "__main__":
    main()
