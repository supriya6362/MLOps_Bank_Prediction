import subprocess
import sys


def run_script(script):

    print("\n======================================")
    print("Running:", script)
    print("======================================")

    result = subprocess.run(
        [
            sys.executable,
            script
        ]
    )

    if result.returncode != 0:

        print(
            f"\nERROR: {script} failed!"
        )

        sys.exit(
            result.returncode
        )


# --------------------------------------------------
# PIPELINE
# --------------------------------------------------

print("======================================")
print(" BANK MARKETING ML PIPELINE")
print("======================================")


print("\n[1] Running preprocessing...")

run_script(
    "src/prepocess.py"
)


print("\n[2] Running model training...")

run_script(
    "src/train.py"
)


print("\n[3] Running model evaluation...")

run_script(
    "src/evaluate.py"
)


print("\n======================================")
print(" PIPELINE COMPLETED SUCCESSFULLY!")
print("======================================")