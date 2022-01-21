#!/bin/bash

# TODO 1: generate some UID that includes the day and time
UID="19-01-2022"

# TODO 2: run array_toy_job and the generated UID as an argument
srun python main.py --uid "$UID"


